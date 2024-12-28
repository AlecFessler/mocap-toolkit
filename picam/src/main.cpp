// © 2024 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <errno.h>
#include <exception>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <sched.h>
#include <signal.h>
#include <sstream>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include "camera_handler.h"
#include "connection.h"
#include "logging.h"
#include "sem_init.h"
#include "videnc.h"

constexpr uint64_t ns_per_s = 1'000'000'000;

volatile static uint64_t timestamp = 0;
volatile static sig_atomic_t running = 1;
volatile static sig_atomic_t stream_end = 0;
volatile static sig_atomic_t frame_rdy = 0;

static std::unique_ptr<sem_t, sem_deleter> loop_ctl_sem;
static std::unique_ptr<camera_handler_t> cam;
static std::unique_ptr<connection> conn;

inline int init_realtime_scheduling(int recording_cpu);
inline int init_timer(timer_t* timerid);
inline int init_signals();
inline int init_sigio(int fd);
inline void arm_timer(
  timer_t timerid,
  uint64_t frame_duration,
  uint64_t& frame_counter
);
inline int flush_encoder(
  videnc& encoder,
  connection& conn
);

int main() {
  try {
    int ret = setup_logging("logs.txt");
    if (ret) {
      std::cout << "Error opening log file: " << strerror(errno) << "\n";
      return -errno;
    }

    config config = parse_config("config.txt");

    uint64_t frame_counter = 0;
    uint64_t frame_duration = ns_per_s / config.fps;
    timer_t timerid;

    loop_ctl_sem = init_semaphore();

    cam = std::make_unique<camera_handler_t>(
      config,
      *loop_ctl_sem.get(),
      frame_rdy
    );
    conn = std::make_unique<connection>(config);
    auto encoder = std::make_unique<videnc>(config);

    if ((ret = init_realtime_scheduling(config.recording_cpu)) < 0) return ret;
    if ((ret = init_timer(&timerid)) < 0) return ret;
    if ((ret = init_signals()) < 0) return ret;
    if ((ret = conn->bind_udp()) < 0) return ret;
    if ((ret = init_sigio(conn->udpfd)) < 0) return ret;

    while (running) {
      if (timestamp) {
        arm_timer(
          timerid,
          frame_duration,
          ++frame_counter
        );
      }

      sem_wait(loop_ctl_sem.get());

      if (stream_end) {
        stream_end = 0;
        frame_rdy = 0;
        frame_counter = 0;
        ret = flush_encoder(*encoder, *conn);
        if (ret == 0)
          conn->end_stream();
        encoder = std::make_unique<videnc>(config);
        continue;
      }

      if (frame_rdy) {
        frame_rdy = 0;
        encoder->encode_frame(cam->frame_buffer);

        int pkt_size = 0;
        uint8_t* ptr = encoder->recv_frame(pkt_size);
        if (ptr == nullptr)
          continue;

        ret = conn->stream_pkt(ptr, pkt_size);
        if (ret != -ECONNRESET)
          continue;

        timestamp = 0;
        frame_counter = 0;
        stream_end = 0;
        conn->discon_tcp();
        encoder = std::make_unique<videnc>(config);
      }
    }

    flush_encoder(*encoder, *conn);
    cleanup_logging();

  } catch (const std::exception& e) {
    char logstr[128];
    snprintf(
      logstr,
      sizeof(logstr),
      "Runtime error: %s",
      e.what()
    );
    LOG(ERROR, logstr);
    return EXIT_FAILURE;
  }

  return 0;
}

void capture_signal_handler(int signo, siginfo_t* info, void* context) {
  (void)signo;
  (void)info;
  (void)context;
  cam->queue_request();
}

void io_signal_handler(int signo, siginfo_t* info, void* context) {
  (void)signo;
  (void)info;
  (void)context;

  size_t buf_size = 8; // bytes
  char buf[buf_size];
  size_t size = conn->recv_msg(buf, buf_size);

  // 8 bytes is our timestamp
  if (size == 8) {
      uint64_t network_timestamp;
      memcpy(&network_timestamp, buf, sizeof(network_timestamp));
      timestamp = network_timestamp;
      sem_post(loop_ctl_sem.get());
      return;
  }

  if (size == 4 && strncmp(buf, "STOP", 4) == 0) {
      LOG(INFO, "Received stop signal, ending stream...");
      timestamp = 0;
      stream_end = 1;
      sem_post(loop_ctl_sem.get());
      return;
  }

  LOG(ERROR, "Unexpected udp message size");
}

void exit_signal_handler(int signo, siginfo_t* info, void* context) {
  (void)signo;
  (void)info;
  (void)context;
  running = 0;
  sem_post(loop_ctl_sem.get());
}

inline int init_realtime_scheduling(int recording_cpu) {
  /**
   * Initializes realtime scheduling for the process
   *
   * Two properties are set:
   *
   * 1. Process is pinned to a specific cpu core to prevent overhead of
   *    the scheduler potentially moving us to a different core.
   *
   * 2. FIFO scheduling with max priority, so any process of equal
   *    priority must wait until this one is blocking on the semaphore
   *    before it can be scheduled on our core. Additionally, any process
   *    of lower priority than max will be preempted as soon as we have
   *    a signal to handle or the semaphore is unblocked.
   */
  char logstr[128];

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(recording_cpu, &cpuset);
  if (sched_setaffinity(0, sizeof(cpuset), &cpuset) < 0) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Failed to set CPU affinity: %s",
      strerror(errno)
    );
    LOG(ERROR, logstr);
    return -errno;
  }

  struct sched_param param;
  param.sched_priority = sched_get_priority_max(SCHED_FIFO);
  if (sched_setscheduler(0, SCHED_FIFO, &param) < 0) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Failed to set real-time scheduling policy: %s",
      strerror(errno)
    );
    LOG(ERROR, logstr);
    return -errno;
  }
  return 0;
}


inline int init_timer(timer_t* timerid) {
  /**
   * Initializes a CLOCK_MONOTONIC timer for precise frame capture timing
   *
   * This timer is fundamental to our multi-camera synchronization system.
   * While our capture timing is ultimately based on PTP-synchronized system
   * clocks (CLOCK_REALTIME), we use CLOCK_MONOTONIC for the actual timer
   * to ensure stable intervals between captures. This is crucial because:
   *
   * 1. CLOCK_MONOTONIC is immune to system time adjustments, preventing
   *    potential frame timing glitches if NTP/PTP adjusts CLOCK_REALTIME
   *
   * 2. When combined with TIMER_ABSTIME in arm_timer(), this provides
   *    precise scheduling of frame captures at specific points in time,
   *    rather than relative delays which could accumulate drift
   *
   * The timer is configured to emit SIGUSR1 signals which trigger frame
   * captures via capture_signal_handler(). The actual timing of these
   * signals is controlled by arm_timer(), which calculates the appropriate
   * monotonic clock targets based on our PTP-synchronized real time targets.
   *
   * Returns 0 on success, -errno on failure
   */
  char logstr[128];

  struct sigevent sev;
  sev.sigev_notify = SIGEV_SIGNAL;
  sev.sigev_signo = SIGUSR1;
  sev.sigev_value.sival_ptr = timerid;

  if (timer_create(CLOCK_MONOTONIC, &sev, timerid) == -1) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Failed to create timer: %s",
      strerror(errno)
    );
    LOG(ERROR, logstr);
    return -errno;
  }
  return 0;
}

inline void arm_timer(timer_t timerid, uint64_t frame_duration, uint64_t& frame_counter) {
    /**
     * Arms the timer to trigger frame captures at precise timestamps
     *
     * This function is the core of our multi-camera synchronization system.
     * It converts PTP-synchronized real time targets into monotonic clock
     * targets that maintain precise timing between frames. The process works
     * as follows:
     *
     * 1. We start with a target timestamp in CLOCK_REALTIME domain, initially
     *    received from the server and shared across all cameras. This timestamp
     *    is incremented by frame_duration (33.33ms at 30fps) before each call.
     *
     * 2. We calculate how far in the future this target is by comparing it
     *    to the current CLOCK_REALTIME. If the target is in the past (which
     *    can happen if we received the initial timestamp late), we adjust
     *    forward by skipping frames until we're back on schedule.
     *
     * 3. We convert this future time delta into the CLOCK_MONOTONIC domain
     *    by adding it to the current monotonic clock value. This maintains
     *    our synchronized timing while protecting against clock adjustments.
     *
     * 4. We set an absolute (TIMER_ABSTIME) timer for this monotonic target.
     *    Using absolute rather than relative timing prevents drift that could
     *    accumulate from processing delays between frames.
     *
     * This system achieves microsecond-level synchronization across multiple
     * cameras by:
     * - Using PTP-synchronized CLOCK_REALTIME for a common time reference
     * - Converting to CLOCK_MONOTONIC for stable intervals
     * - Automatically adjusting for network delays in initial timestamp delivery
     * - Using absolute rather than relative timer targets
     *
     * The timer will emit SIGUSR1 when the target time is reached, triggering
     * capture_signal_handler() to initiate the actual frame capture.
     */
    struct timespec real_time, mono_time;
    clock_gettime(CLOCK_REALTIME, &real_time);
    clock_gettime(CLOCK_MONOTONIC, &mono_time);

    uint64_t current_real_ns = (uint64_t)real_time.tv_sec * ns_per_s + real_time.tv_nsec;
    uint64_t current_mono_ns = (uint64_t)mono_time.tv_sec * ns_per_s + mono_time.tv_nsec;

    uint64_t target = timestamp + frame_duration * frame_counter;
    uint64_t ns_until_target = target - current_real_ns;

    if (ns_until_target <= 0) {
        uint32_t frames_elapsed = (-ns_until_target / frame_duration) + 1;
        uint64_t ns_elapsed = frames_elapsed * frame_duration;
        ns_until_target += ns_elapsed;             // adjust ns_until_target for setting this current timer
        frame_counter += frames_elapsed;           // adjust counter so we're caught up for future frames
        target += frame_duration * frames_elapsed; // adjust the target for the connections timestamp queue
    }

    conn->frame_timestamps.push(target);

    uint64_t mono_target_ns = current_mono_ns + ns_until_target;

    struct itimerspec its;
    its.it_value.tv_sec = mono_target_ns / ns_per_s;
    its.it_value.tv_nsec = mono_target_ns % ns_per_s;
    its.it_interval.tv_sec = 0;
    its.it_interval.tv_nsec = 0;

    timer_settime(timerid, TIMER_ABSTIME, &its, NULL);
}

inline int init_sigio(int fd) {
  /**
   * Binds the udp socket fd to emit SIGIO upon incoming data
   *
   * This enables us to block on the semaphore in the main loop
   * before the initial timestamp comes in. When it does, the main
   * loop will be preempted, the timestamp set, and the semaphore
   * incremented so the main loop can unblock and arm the timer
   * (see io_signal_handler(), arm_timer()).
   *
   * This function also enables us to avoid polling for the "STOP"
   * message since the SIGIO will be emitted upon receiving any
   * data on the port assigned to the udp file descriptor.
   */
  char logstr[128];

  if (fcntl(fd, F_SETFL, O_NONBLOCK | O_ASYNC) < 0) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Failed to set properties for UDP socket: %s",
      strerror(errno)
    );
    LOG(ERROR, logstr);
    return -errno;
  }

  if (fcntl(fd, F_SETOWN, getpid()) < 0) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Failed to set owner process for SIGIO: %s",
      strerror(errno)
    );
    LOG(ERROR, logstr);
    return -errno;
  }

  return 0;
}

inline int init_signals() {
  /**
   * Sets the sa_mask for the process
   *
   * There are 4 signals handled:
   *
   * SIGUSR1 - emitted when the timer (see init_timer(), arm_timer())
   *           reaches the assigned timestamp, handled by enqueueing
   *           a capture request with the camera
   *
   * SIGIO   - emitted whenever data is received on the udp port
   *           (see connection::bind_udp(), io_signal_handler()).
   *           If the data is 8 bytes, it's our starting timestamp,
   *           if it's 4 bytes, it's our "STOP" message, otherwise
   *           it's unexpected and is a server side bug.
   *
   * SIGINT  - emitted by the os to signal for exit
   *
   * SIGTERM - emitted by the os to signal for exit
   *
   * SA_RESTART flag is set, which means any interrupted syscalls
   * will be retried.
   */
  char logstr[128];

  struct sigaction action;
  action.sa_sigaction = capture_signal_handler;
  action.sa_flags = SA_SIGINFO | SA_RESTART;
  sigemptyset(&action.sa_mask);

  struct sigaction io_action;
  io_action.sa_sigaction = io_signal_handler;
  io_action.sa_flags = SA_SIGINFO | SA_RESTART;
  sigemptyset(&io_action.sa_mask);

  struct sigaction exit_action;
  exit_action.sa_sigaction = exit_signal_handler;
  exit_action.sa_flags = SA_SIGINFO | SA_RESTART;
  sigemptyset(&exit_action.sa_mask);

  if (sigaction(SIGUSR1, &action, NULL) < 0 ||
      sigaction(SIGIO, &io_action, NULL) < 0 ||
      sigaction(SIGINT, &exit_action, NULL) < 0 ||
      sigaction(SIGTERM, &exit_action, NULL) < 0) {
      snprintf(
        logstr,
        sizeof(logstr),
        "Failed to set signal handlers: %s",
        strerror(errno)
      );
      LOG(ERROR, logstr);
      return -errno;
    }

  return 0;
}

inline int flush_encoder(videnc& encoder, connection& conn) {
    encoder.flush();
    int pkt_size = 0;
    uint8_t* ptr = nullptr;
    while ((ptr = encoder.recv_frame(pkt_size)) != nullptr) {
      int ret = conn.stream_pkt(ptr, pkt_size);
      if (ret == -ECONNRESET) return ret;
    }
    return 0;
}
