// © 2025 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#include <atomic>
#include <chrono>
#include <csignal>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>

#include "camera.hpp"
#include "encoder_thread.hpp"
#include "frame_buffer.hpp"
#include "interval_timer.hpp"
#include "logging.hpp"
#include "packet_buffer.hpp"
#include "parse_config.hpp"
#include "protocol.hpp"
#include "scheduling.hpp"
#include "sigsets.hpp"
#include "spsc_queue_wrapper.hpp"
#include "stream_thread.hpp"
#include "tcp_control.hpp"

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
    QUEUE_SLOTS + 2,
    frame_queue
  };

  auto interval = std::chrono::nanoseconds{std::chrono::seconds{1}} / config.fps;

  pin_to_core(0);
  set_scheduling_prio(98);
  setup_sig_handler(SIGTERM, terminate_handler);
  setup_sig_handler(SIGRTMIN, SIG_IGN);
  setup_sig_handler(SIGPIPE, SIG_IGN); // ignore SIGPIPE for TCP write failures

  // TCP control channel — connects to server and stays connected
  TcpControl control{config.ctrl_port, std::string_view(config.server_ip), config.camera_id};

  log_(INFO, "Initialization complete, entering session loop");

  // session loop: connect, wait for START, stream, cleanup, repeat
  while (!terminate_flag.load(std::memory_order_relaxed)) {

    // ensure connected to server (retries with backoff)
    control.ensure_connected(terminate_flag);
    if (!control.is_connected())
      continue;

    log_(INFO, "Connected to server, waiting for START command");

    // wait for START command from server
    std::chrono::nanoseconds initial_timestamp{0};
    ctrl_msg_header header;
    uint64_t timestamp_payload;

    while (initial_timestamp == std::chrono::nanoseconds{0} &&
           !terminate_flag.load(std::memory_order_relaxed)) {

      // poll the control fd with timeout for heartbeat
      struct pollfd pfd;
      pfd.fd = control.fd();
      pfd.events = POLLIN;

      int ret = poll(&pfd, 1, HEARTBEAT_INTERVAL_MS);
      if (ret < 0) {
        if (errno == EINTR) continue;
        break;
      }

      if (ret == 0) {
        // timeout — send heartbeat
        control.send_msg(CTRL_HEARTBEAT);
        if (!control.is_connected()) break;
        continue;
      }

      int msg_type = control.recv_msg(&header, &timestamp_payload, sizeof(timestamp_payload));
      if (msg_type < 0) {
        log_(WARNING, "Lost connection to server");
        break;
      }

      if (msg_type == CTRL_HEARTBEAT) {
        continue;
      } else if (msg_type == CTRL_START) {
        initial_timestamp = std::chrono::nanoseconds{timestamp_payload};
        control.send_msg(CTRL_ACK);
        log_(INFO, "Received START command");
      } else if (msg_type == CTRL_STOP) {
        control.send_msg(CTRL_ACK);
        continue;
      }
    }

    if (terminate_flag.load(std::memory_order_relaxed))
      break;

    if (initial_timestamp == std::chrono::nanoseconds{0}) {
      // lost connection — loop back to reconnect
      continue;
    }

    // session_stop: set by StreamThread error or server STOP
    std::atomic<bool> session_stop{0};

    { // session scope
      IntervalTimer timer{
        initial_timestamp,
        interval,
        SIGRTMIN
      };
      EncoderThread encoder_thread{
        config.resolution,
        config.fps,
        QUEUE_SLOTS + 2,
        frame_queue,
        packet_queue,
        session_stop
      };
      StreamThread stream_thread{
        config.stream_port,
        std::string_view(config.server_ip),
        packet_queue,
        session_stop
      };

      encoder_thread.launch();
      stream_thread.launch();

      // capture loop — also check for STOP command via control channel
      sigset_t sigset = setup_sigwait({SIGRTMIN, SIGTERM});
      std::chrono::nanoseconds next_capture = timer.arm_timer();
      uint32_t frame_counter = 0;

      while (!session_stop.load(std::memory_order_relaxed) &&
             !terminate_flag.load(std::memory_order_relaxed)) {
        int signal;
        struct timespec sig_timeout = {.tv_sec = 0, .tv_nsec = 33000000}; // 33ms
        signal = sigtimedwait(&sigset, nullptr, &sig_timeout);

        if (signal == SIGTERM)
          break;

        if (signal == SIGRTMIN) {
          cam.capture_frame(next_capture);
          next_capture = timer.arm_timer();
          frame_counter++;
        }

        // every 30 frames (~1s), send heartbeat on control channel
        // a write to a dead connection fails immediately with EPIPE
        if (frame_counter % 30 == 0 && control.fd() >= 0) {
          control.send_msg(CTRL_HEARTBEAT);
          if (!control.is_connected()) {
            log_(WARNING, "Server disconnected (heartbeat write failed)");
            session_stop.store(1, std::memory_order_relaxed);
            continue;
          }
        }

        // also check for incoming STOP (non-blocking peek)
        if (control.fd() >= 0) {
          char peek_buf;
          ssize_t peek = recv(control.fd(), &peek_buf, 1, MSG_PEEK | MSG_DONTWAIT);
          if (peek == 0) {
            log_(WARNING, "Server disconnected (EOF)");
            control.disconnect();
            session_stop.store(1, std::memory_order_relaxed);
          } else if (peek > 0) {
            ctrl_msg_header hdr;
            uint8_t buf[16];
            int msg = control.recv_msg(&hdr, buf, sizeof(buf));
            if (msg == CTRL_STOP) {
              control.send_msg(CTRL_ACK);
              log_(INFO, "Received STOP from server");
              session_stop.store(1, std::memory_order_relaxed);
            } else if (msg < 0) {
              log_(WARNING, "Lost connection to server");
              session_stop.store(1, std::memory_order_relaxed);
            }
          } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
            log_(WARNING, "Control channel error, disconnecting");
            control.disconnect();
            session_stop.store(1, std::memory_order_relaxed);
          }
        } else {
          log_(WARNING, "Control channel disconnected");
          session_stop.store(1, std::memory_order_relaxed);
        }
      }
    } // destructors fire

    // unblock SIGTERM so it can be delivered normally outside capture loop
    // (setup_sigwait blocks it for sigtimedwait, but that persists across sessions)
    sigset_t unblock;
    sigemptyset(&unblock);
    sigaddset(&unblock, SIGTERM);
    sigprocmask(SIG_UNBLOCK, &unblock, nullptr);

    frame_queue.drain();
    packet_queue.drain();

    if (terminate_flag.load(std::memory_order_relaxed)) {
      log_(INFO, "Received SIGTERM, shutting down");
      break;
    }

    log_(INFO, "Session ended, waiting for next START...");
  }

  return 0;

} catch (const std::exception& e) {
  log_(ERROR, e.what());
  return -1;
}
}
