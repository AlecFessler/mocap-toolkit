// © 2025 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#include <atomic>
#include <chrono>
#include <csignal>
#include <stdexcept>
#include <string>

#include "camera.hpp"
#include "encoder_thread.hpp"
#include "frame_buffer.hpp"
#include "interval_timer.hpp"
#include "logging.hpp"
#include "packet_buffer.hpp"
#include "parse_config.hpp"
#include "scheduling.hpp"
#include "sigsets.hpp"
#include "spsc_queue_wrapper.hpp"
#include "stop_watchdog.hpp"
#include "stream_thread.hpp"
#include "udp_socket.hpp"

constexpr const char* LOG_PATH = "/var/log/picam/picam.log";
constexpr const char* CONFIG_PATH = "/etc/picam/cam_config.txt";
constexpr uint32_t QUEUE_SLOTS = 8;

// terminate_flag: set by SIGTERM, exits the process entirely
static std::atomic<bool> terminate_flag = 0;
static void terminate_handler(int signum) {
  (void)signum;
  terminate_flag.store(1, std::memory_order_relaxed);
}

int main() {
try {

  Logging::setup_logging(LOG_PATH);
  struct config config = parse_config(CONFIG_PATH);
  SPSCQueue<struct frame> frame_queue{QUEUE_SLOTS};
  SPSCQueue<struct packet> packet_queue{QUEUE_SLOTS};
  Camera cam{
    config.resolution,
    config.fps,
    QUEUE_SLOTS + 2, // num frames
    frame_queue
  };

  auto interval = std::chrono::nanoseconds{std::chrono::seconds{1}} / config.fps;
  UdpSocket udpsock{config.udp_port};

  pin_to_core(0);
  set_scheduling_prio(98);
  setup_sig_handler(SIGTERM, terminate_handler);
  setup_sig_handler(SIGRTMIN, SIG_IGN);

  log_(INFO, "Initialization complete, entering session loop");

  // session loop: wait for timestamp, stream, cleanup, repeat
  while (!terminate_flag.load(std::memory_order_relaxed)) {

    // wait for initial timestamp from server
    sigset_t sigset = setup_sigwait({SIGIO, SIGTERM});
    std::chrono::nanoseconds initial_timestamp{0};

    while (initial_timestamp == std::chrono::nanoseconds{0} &&
           !terminate_flag.load(std::memory_order_relaxed)) {
      int signal;
      sigwait(&sigset, &signal);
      if (signal == SIGTERM)
        break;

      initial_timestamp = udpsock.recv_stream_ctl();
      if (initial_timestamp == std::chrono::nanoseconds{0}) {
        log_(WARNING, "Received stop signal while waiting for timestamp, ignoring");
        continue;
      }
    }

    if (terminate_flag.load(std::memory_order_relaxed))
      break;

    log_(INFO, "Received initial timestamp, starting session");

    // session_stop: set by StopWatchdog (UDP STOP) or StreamThread (TCP error)
    // only ends the current session, not the process
    std::atomic<bool> session_stop{0};

    { // session scope — all session resources destroyed on scope exit
      IntervalTimer timer{
        initial_timestamp,
        interval,
        SIGRTMIN
      };
      EncoderThread encoder_thread{
        config.resolution,
        config.fps,
        QUEUE_SLOTS + 2, // num packets
        frame_queue,
        packet_queue,
        session_stop
      };
      StreamThread stream_thread{
        config.tcp_port,
        std::string_view(config.server_ip),
        packet_queue,
        session_stop
      };
      StopWatchdog stop_watcher{session_stop, udpsock};

      stop_watcher.launch();
      encoder_thread.launch();
      stream_thread.launch();

      sigset = setup_sigwait({SIGRTMIN, SIGTERM});
      std::chrono::nanoseconds next_capture = timer.arm_timer();

      while (!session_stop.load(std::memory_order_relaxed) &&
             !terminate_flag.load(std::memory_order_relaxed)) {
        int signal;
        sigwait(&sigset, &signal);
        if (signal == SIGTERM)
          break;

        cam.capture_frame(next_capture);
        next_capture = timer.arm_timer();
      }
    } // destructors fire: threads joined, TCP closed, timer deleted

    // drain queues so they're clean for next session
    frame_queue.drain();
    packet_queue.drain();

    if (terminate_flag.load(std::memory_order_relaxed)) {
      log_(INFO, "Received SIGTERM, shutting down");
      break;
    }

    log_(INFO, "Session ended, waiting for next timestamp...");
  }

  return 0;

} catch (const std::exception& e) {
  log_(ERROR, e.what());
  return -1;
}
}
