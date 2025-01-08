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
#include "scheduling.hpp"
#include "sigsets.hpp"
#include "spsc_queue_wrapper.hpp"
#include "stop_watchdog.hpp"
#include "stream_thread.hpp"
#include "udp_socket.hpp"

constexpr const char* LOG_PATH = "/var/log/picam/picam.log";
constexpr uint32_t QUEUE_SLOTS = 8;

// NOTE: TEMPORARY UNTIL CONFIG PARSER IS BUILT
constexpr const char* IP = "192.168.86.100";
constexpr uint16_t TCP_PORT = 12345;
constexpr uint16_t UDP_PORT = 22345;
constexpr uint32_t FPS = 30;
constexpr std::pair<uint32_t, uint32_t> RESOLUTION = {1280, 720};

static std::atomic<bool> stop_flag = 0;
static void stop_handler(int signum) {
  (void)signum;
  stop_flag.store(1, std::memory_order_relaxed);
}

int main() {
  Logging::setup_logging(LOG_PATH);
  SPSCQueue<struct frame> frame_queue{QUEUE_SLOTS};
  SPSCQueue<struct packet> packet_queue{QUEUE_SLOTS};
  Camera cam{
    RESOLUTION,
    FPS,
    QUEUE_SLOTS + 2, // num frames
    frame_queue
  };
  EncoderThread encoder_thread{
    RESOLUTION,
    FPS,
    QUEUE_SLOTS + 2, // num packets
    frame_queue,
    packet_queue,
    stop_flag
  };
  StreamThread stream_thread{
    TCP_PORT,
    std::string_view(IP),
    packet_queue,
    stop_flag
  };
  UdpSocket udpsock{UDP_PORT};
  StopWatchdog stop_watcher_thread{stop_flag, udpsock};

  pin_to_core(0);
  set_scheduling_prio(99);
  setup_sig_handler(SIGTERM, stop_handler);
  setup_sig_handler(SIGRTMIN, SIG_IGN);
  sigset_t sigset = setup_sigwait({SIGIO, SIGTERM});

  std::chrono::nanoseconds initial_timestamp{0};
  while (
    initial_timestamp == std::chrono::nanoseconds{0} &&
    !stop_flag.load(std::memory_order_relaxed)
  ) {
    int signal;
    sigwait(&sigset, &signal);
    if (signal == SIGTERM)
      break;

    initial_timestamp = udpsock.recv_stream_ctl();
    if (initial_timestamp == std::chrono::nanoseconds{0}) {
      std::string warning_msg = "Expected timestamp but received stop message";
      log_(WARNING, warning_msg.c_str());
      continue;
    }

    auto interval = std::chrono::nanoseconds{std::chrono::seconds{1}} / FPS;
    IntervalTimer timer{
      initial_timestamp,
      interval,
      SIGRTMIN
    };

    stop_watcher_thread.launch();
    encoder_thread.launch();
    stream_thread.launch();

    sigset = setup_sigwait({SIGRTMIN, SIGTERM});
    std::chrono::nanoseconds next_capture = timer.arm_timer();
    while (!stop_flag.load(std::memory_order_relaxed)) {
      sigwait(&sigset, &signal);
      if (signal == SIGTERM)
        break;

      cam.capture_frame(next_capture);
      next_capture = timer.arm_timer();
    }
  }

  return 0;
}
