// © 2024 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#include <atomic>
#include <arpa/inet.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
#include "logger.h"
#include "sem_init.h"
#include "videnc.h"

extern char** environ;

constexpr int64_t ns_per_s = 1'000'000'000;

volatile static int64_t timestamp = 0;
volatile static sig_atomic_t running = 1;
volatile static sig_atomic_t stream_end = 0;
volatile static sig_atomic_t frame_rdy = 0;

static std::unique_ptr<sem_t, sem_deleter> loop_ctl_sem;
static std::unique_ptr<camera_handler_t> cam;
static std::unique_ptr<connection> conn;

std::unique_ptr<logger_t> logger;

inline int init_realtime_scheduling(int recording_cpu);
inline int init_timer(timer_t* timerid);
inline int init_signals();
inline int init_sigio(int fd);
inline void arm_timer(
  int64_t timestamp,
  timer_t timerid,
  int64_t frame_duration
);
inline void flush_encoder(
  videnc& encoder,
  connection& conn
);

int main() {
  try {
    logger = std::make_unique<logger_t>("logs.txt");
    config config = parse_config("config.txt");

    int64_t frame_duration = ns_per_s / config.fps;
    timer_t timerid;

    loop_ctl_sem = init_semaphore();

    cam = std::make_unique<camera_handler_t>(
      config,
      *loop_ctl_sem.get(),
      frame_rdy
    );
    conn = std::make_unique<connection>(config, frame_duration);
    videnc encoder{config};

    int ret;
    if ((ret = init_realtime_scheduling(config.recording_cpu)) < 0) return ret;
    if ((ret = init_timer(&timerid)) < 0) return ret;
    if ((ret = init_signals()) < 0) return ret;
    if ((ret = conn->bind_udp()) < 0) return ret;
    if ((ret = init_sigio(conn->udpfd)) < 0) return ret;

    while (running) {
      if (timestamp) {
        timestamp += frame_duration;
        arm_timer(
          timestamp,
          timerid,
          frame_duration
        );
      }

      sem_wait(loop_ctl_sem.get());

      if (frame_rdy) {
        frame_rdy = 0;
        if (!stream_end) {
          encoder.encode_frame(cam->frame_buffer);
          int pkt_size = 0;
          uint8_t* ptr = encoder.recv_frame(pkt_size);
          if (ptr) {
            conn->stream_pkt(ptr, pkt_size);
          }
        }
      }

      if (stream_end) {
        stream_end = 0;
        flush_encoder(encoder, *conn);
        conn->discon_tcp();
        encoder = videnc(config);
      }
    }

    flush_encoder(encoder, *conn);

  } catch (const std::exception& e) {
    if (logger)
      logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, e.what());
    else
      std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return 0;
}

void capture_signal_handler(int signo, siginfo_t* info, void* context) {
  (void)signo;
  (void)info;
  (void)context;
  cam->queue_request();
  logger->log(logger_t::level_t::INFO, __FILE__, __LINE__, "Capture request queued");
}

void io_signal_handler(int signo, siginfo_t* info, void* context) {
  (void)signo;
  (void)info;
  (void)context;

  char buf[8];
  size_t size = conn->recv_msg(buf);

  // 8 bytes is our timestamp
  if (size == 8) {
    logger->log(logger_t::level_t::INFO, __FILE__, __LINE__, "Receied timestamp, starting stream...");
    uint64_t network_timestamp;
    memcpy(&network_timestamp, buf, sizeof(network_timestamp));
    timestamp = be64toh(network_timestamp);
    conn->timestamp = timestamp;
    sem_post(loop_ctl_sem.get());
    return;
  }

  // 4 bytes, "STOP" is our stop signal
  if (size == 4 && strncmp(buf, "STOP", 4) == 0) {
    logger->log(logger_t::level_t::INFO, __FILE__, __LINE__, "Received stop signal, ending stream...");
    timestamp = 0;
    conn->timestamp = 0;
    stream_end = 1;
    sem_post(loop_ctl_sem.get());
    return;
  }

  logger->log(logger_t::level_t::DEBUG, __FILE__, __LINE__, "Unexpected udp message size");
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
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(recording_cpu, &cpuset);
  if (sched_setaffinity(0, sizeof(cpuset), &cpuset) < 0) {
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, "Failed to set CPU affinity");
    return -errno;
  }

  struct sched_param param;
  param.sched_priority = sched_get_priority_max(SCHED_FIFO);
  if (sched_setscheduler(0, SCHED_FIFO, &param) < 0) {
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, "Failed to set real-time scheduling policy");
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
  struct sigevent sev;
  sev.sigev_notify = SIGEV_SIGNAL;
  sev.sigev_signo = SIGUSR1;
  sev.sigev_value.sival_ptr = timerid;

  if (timer_create(CLOCK_MONOTONIC, &sev, timerid) == -1) {
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, "Failed to create timer");
    return -errno;
  }
  return 0;
}

inline void arm_timer(int64_t timestamp, timer_t timerid, int64_t frame_duration) {
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

    int64_t current_real_ns = (int64_t)real_time.tv_sec * ns_per_s + real_time.tv_nsec;
    int64_t current_mono_ns = (int64_t)mono_time.tv_sec * ns_per_s + mono_time.tv_nsec;

    int64_t ns_until_target = timestamp - current_real_ns;

    //char debug_msg[256];
    //snprintf(debug_msg, sizeof(debug_msg),
    //         "Current real: %ld, Target: %ld, Delta: %ld ns (%.2f ms)",
    //         current_real_ns, timestamp, ns_until_target,
    //         ns_until_target / 1000000.0);
    //logger->log(logger_t::level_t::DEBUG, __FILE__, __LINE__, debug_msg);

    if (ns_until_target <= 0) {
        int64_t frames_elapsed = (-ns_until_target / frame_duration) + 1;
        int64_t ns_elapsed = frames_elapsed * frame_duration;
        ns_until_target += ns_elapsed;
        // the adjustment only happens a single time ever
        // and only if the initial timestamp is received late
        timestamp += ns_elapsed;
        conn->timestamp += ns_elapsed;

        //snprintf(debug_msg, sizeof(debug_msg),
        //        "Target in past, adjusted by %ld frames to delta: %ld ns",
        //        frames_elapsed, ns_until_target);
        //logger->log(logger_t::level_t::DEBUG, __FILE__, __LINE__, debug_msg);
    }

    int64_t mono_target_ns = current_mono_ns + ns_until_target;

    struct itimerspec its;
    its.it_value.tv_sec = mono_target_ns / ns_per_s;
    its.it_value.tv_nsec = mono_target_ns % ns_per_s;
    its.it_interval.tv_sec = 0;
    its.it_interval.tv_nsec = 0;

    timer_settime(timerid, TIMER_ABSTIME, &its, NULL);

    //snprintf(debug_msg, sizeof(debug_msg),
    //         "Timer set for monotonic timestamp %ld (current mono: %ld)",
    //         mono_target_ns, current_mono_ns);
    //logger->log(logger_t::level_t::DEBUG, __FILE__, __LINE__, debug_msg);
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
  if (fcntl(fd, F_SETFL, O_NONBLOCK | O_ASYNC) < 0) {
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, "Failed to set O_NONBLOCK | O_ASYNC on UDP socket");
    return -errno;
  }

  if (fcntl(fd, F_SETOWN, getpid()) < 0) {
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, "Failed to set owner process for SIGIO");
    return -errno;
  }

  logger->log(logger_t::level_t::INFO, __FILE__, __LINE__, "Signal-driven I/O enabled");
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
      logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, "Failed to set signal handlers");
      return -errno;
    }

  return 0;
}

inline void flush_encoder(videnc& encoder, connection& conn) {
    encoder.flush();
    int pkt_size = 0;
    uint8_t* ptr = nullptr;
    while ((ptr = encoder.recv_frame(pkt_size)) != nullptr) {
        conn.stream_pkt(ptr, pkt_size);
    }
}
