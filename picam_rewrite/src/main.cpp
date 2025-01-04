// © 2025 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#include <chrono>
#include <csignal>
#include <stdexcept>
#include <string>

#include "interval_timer.hpp"
#include "logging.hpp"
#include "sigsets.hpp"
#include "stop_watchdog.hpp"
#include "tcp_socket.hpp"
#include "udp_socket.hpp"

constexpr const char* LOG_PATH = "/var/log/picam/picam.log";

// NOTE: TEMPORARY UNTIL CONFIG PARSER IS BUILT
constexpr const char* IP = "192.168.86.100";
constexpr uint16_t TCP_PORT = 12345;
constexpr uint16_t UDP_PORT = 22345;
constexpr uint32_t FPS = 30;

static volatile sig_atomic_t stop_flag = 0;
static void stop_handler(int signum) {
  (void)signum;
  stop_flag = 1;
}

int main() {
  Logging::setup_logging(LOG_PATH);
  setup_sig_handler(SIGTERM, stop_handler);
  setup_sig_handler(SIGRTMIN, SIG_IGN);
  UdpSocket udpsock{UDP_PORT};
  TcpSocket tcpsock{TCP_PORT, std::string_view(IP)};

  // setup camera - resolution and fps

  // setup encoder - zerolatency and ultrafast

  // launch worker threads 2x - share ptr to encoder
  // and to tcpsock, have locks to sync access

  sigset_t sigset = setup_sigwait({SIGIO, SIGTERM});
  std::chrono::nanoseconds initial_timestamp{0};

  while (initial_timestamp == std::chrono::nanoseconds{0} && !stop_flag) {
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

    std::string recvd_timestamp_msg =
      "Received timestamp "
      + std::to_string(initial_timestamp.count());
    log_(INFO, recvd_timestamp_msg.c_str());

    StopWatchdog::launch_watchdog(
      pthread_self(),
      std::move(udpsock)
    );

    auto interval = std::chrono::nanoseconds{std::chrono::seconds{1}} / FPS;
    IntervalTimer timer{
      initial_timestamp,
      interval,
      SIGRTMIN
    };
    std::chrono::nanoseconds next_capture = timer.arm_timer();

    std::string armed_timer_msg =
      "Armed timer for "
      + std::to_string(next_capture.count());
    log_(INFO, armed_timer_msg.c_str());

    sigset = setup_sigwait({SIGRTMIN, SIGTERM});

    while (!stop_flag) {
      sigwait(&sigset, &signal);
      if (signal == SIGTERM)
        break;

      // queue capture request to camera

      next_capture = timer.arm_timer();
      armed_timer_msg =
        "Armed timer for "
        + std::to_string(next_capture.count());
      log_(INFO, armed_timer_msg.c_str());
    }
  }

  return 0;
}
