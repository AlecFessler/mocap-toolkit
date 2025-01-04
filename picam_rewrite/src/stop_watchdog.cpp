// © 2025 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#include <chrono>
#include <csignal>
#include <cstring>
#include <pthread.h>
#include <stdexcept>
#include <string>

#include "logging.hpp"
#include "sigsets.hpp"
#include "stop_watchdog.hpp"
#include "udp_socket.hpp"

constexpr int MAIN_THREAD_STOP_SIGNAL = SIGTERM;
constexpr int WATCHDOG_THREAD_STOP_SIGNAL = SIGUSR1;

static volatile sig_atomic_t stop_flag = 0;

static void stop_handler(int signum) {
  (void)signum;
  stop_flag = 1;
}

void* stop_watchdog_fn(void* ptr) {
  struct stop_watchdog_ctx* ctx = static_cast<struct stop_watchdog_ctx*>(ptr);
  setup_sig_handler(WATCHDOG_THREAD_STOP_SIGNAL, stop_handler);

  int signal;
  sigset_t sigset;
  sigset = setup_sigwait({SIGIO, WATCHDOG_THREAD_STOP_SIGNAL});

  while (!stop_flag) {
    sigwait(&sigset, &signal);
    if (signal == WATCHDOG_THREAD_STOP_SIGNAL)
      break;

    std::chrono::nanoseconds stream_ctl = ctx->udpsock.recv_stream_ctl();
    if (stream_ctl != std::chrono::nanoseconds{0}) {
      std::string warning_msg = "Received unexpected stream control while waiting for stop sentinel";
      log_(WARNING, warning_msg.c_str());
      continue;
    }

    std::string info_msg = "Received stop signal, killing main thread";
    log_(INFO, info_msg.c_str());

    pthread_kill(ctx->main_thread, MAIN_THREAD_STOP_SIGNAL);
    break;
  }

  return nullptr;
}

StopWatchdog::StopWatchdog(
  pthread_t main_thread,
  UdpSocket&& udpsock
) : ctx{main_thread, std::move(udpsock)} {
  int status = pthread_create(
    &tid,
    nullptr,
    stop_watchdog_fn,
    (void*)&ctx
  );
  if (status != 0) {
    std::string err_msg =
      "Failed to spawn stop watchdog thread: "
      + std::string(strerror(errno));
    log_(ERROR, err_msg.c_str());
    throw std::runtime_error(err_msg);
  }
}

void StopWatchdog::launch_watchdog(
  pthread_t main_thread,
  UdpSocket&& udpsock
) {
  static StopWatchdog instance{
    main_thread,
    std::move(udpsock)
  };
}

StopWatchdog::~StopWatchdog() {
  pthread_kill(tid, WATCHDOG_THREAD_STOP_SIGNAL);
  pthread_join(tid, nullptr);
}
