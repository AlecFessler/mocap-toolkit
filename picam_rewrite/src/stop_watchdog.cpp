// © 2025 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#include <atomic>
#include <chrono>
#include <cstring>
#include <pthread.h>
#include <stdexcept>
#include <string>

#include "logging.hpp"
#include "sigsets.hpp"
#include "stop_watchdog.hpp"
#include "udp_socket.hpp"

static volatile sig_atomic_t stop_flag = 0;
static void stop_handler(int signum) {
  (void)signum;
  stop_flag = 1;
}

void* stop_watchdog_fn(void* ptr) {
  auto this = static_cast<StopWatchdog*>(ptr);
  setup_sig_handler(SIGTERM, stop_handler);

  int signal;
  sigset_t sigset;
  sigset = setup_sigwait({SIGIO, SIGTERM});

  while (!stop_flag) {
    sigwait(&sigset, &signal);
    if (signal == SIGTERM)
      break;

    std::chrono::nanoseconds stream_ctl = this->m_udpsock.recv_stream_ctl();
    if (stream_ctl != std::chrono::nanoseconds{0}) {
      std::string warning_msg = "Received unexpected stream control while waiting for stop sentinel";
      log_(WARNING, warning_msg.c_str());
      continue;
    }

    std::string info_msg = "Received stop signal, stopping main thread";
    log_(INFO, info_msg.c_str());

    this->m_main_stop_flag.store(1, std::memory_order_relaxed);
    break;
  }

  return nullptr;
}

StopWatchdog::StopWatchdog(std::atomic<bool>& main_stop_flag) :
  m_main_stop_flag(main_stop_flag) {}

StopWatchdog::~StopWatchdog() {
  pthread_kill(m_this_thread, SIGTERM);
  pthread_join(m_this_thread, nullptr);
}

void StopWatchdog::launch(UdpSocket&& udpsock) {
  m_udpsock = std::move(udpsock);
  int status = pthread_create(
    &m_this_thread,
    nullptr,
    stop_watchdog_fn,
    static_cast<void*>(this)
  );
  if (status != 0) {
    std::string err_msg =
      "Failed to spawn stop watchdog thread: "
      + std::string(strerror(errno));
    log_(ERROR, err_msg.c_str());
    throw std::runtime_error(err_msg);
  }
}
