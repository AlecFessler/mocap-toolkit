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
#include "stream_thread.hpp"
#include "udp_socket.hpp"

constexpr const char* LOG_PATH = "/var/log/picam/picam.log";
constexpr const char* CONFIG_PATH = "/etc/picam/cam_config.txt";
constexpr uint32_t QUEUE_SLOTS = 8;

int main() {
try {
  Logging::setup_logging(LOG_PATH);
  struct config config = parse_config(CONFIG_PATH);
  // block signals before spawning any threads to ensure they're never delivered to a thread without a handler
  sigset_t sigset = setup_sigwait({SIGIO, SIGRTMIN, SIGTERM});
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

  log_(INFO, "Initialization complete, entering session loop");

  bool terminate = false;
  // session loop: wait for timestamp, stream, cleanup, repeat
  while (!terminate) {

    // wait for initial timestamp from server
    std::chrono::nanoseconds initial_timestamp{0};
    while (initial_timestamp == std::chrono::nanoseconds{0} && !terminate) {
      int signal;
      sigwait(&sigset, &signal);
      if (signal == SIGTERM) {
        terminate = true;
        break;
      } else if (signal == SIGIO) {
        initial_timestamp = udpsock.recv_stream_ctl();
        if (initial_timestamp == std::chrono::nanoseconds{0}) {
          log_(WARNING, "Received stop signal while waiting for timestamp, ignoring");
          continue;
        }
      }
    }
    if (terminate) break;

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

      encoder_thread.launch();
      stream_thread.launch();

      std::chrono::nanoseconds next_capture = timer.arm_timer();

      while (!terminate && !session_stop.load(std::memory_order::acquire)) {
        int signal;
        sigwait(&sigset, &signal);
        if (signal == SIGTERM) {
          terminate = true;
          session_stop.store(true, std::memory_order::release);
          break;
        } else if (signal == SIGIO) {
          std::chrono::nanoseconds stream_ctl = udpsock.recv_stream_ctl();
          if (stream_ctl == std::chrono::nanoseconds{0}) {
            session_stop.store(true, std::memory_order::release);
            break;
          } else {
            std::string warning_msg = "Received unexpected stream control while waiting for stop sentinel";
            log_(WARNING, warning_msg.c_str());
            continue;
          }
        }

        cam.capture_frame(next_capture);
        next_capture = timer.arm_timer();
      }
    } // destructors fire: threads joined, TCP closed, timer deleted

    if (terminate) {
      log_(INFO, "Received SIGTERM, shutting down");
      break;
    }

    // drain queues so they're clean for next session
    frame_queue.drain();
    packet_queue.drain();

    log_(INFO, "Session ended, waiting for next timestamp...");
  }

  return 0;

} catch (const std::exception& e) {
  log_(ERROR, e.what());
  return -1;
}
}
