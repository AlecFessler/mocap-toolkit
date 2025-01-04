// © 2025 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#ifndef STOP_WATCHDOG_HPP
#define STOP_WATCHDOG_HPP

#include <pthread.h>
#include <sys/types.h>

#include "udp_socket.hpp"

struct stop_watchdog_ctx {
  pthread_t main_thread;
  UdpSocket udpsock;
};

class StopWatchdog{
private:
  pthread_t tid;
  stop_watchdog_ctx ctx;

  StopWatchdog(
    pthread_t main_thread,
    UdpSocket&& udpsock
  );

public:
  ~StopWatchdog();
  static void launch_watchdog(
    pthread_t main_thread,
    UdpSocket&& udpsock
  );
};

#endif // STOP_WATCHDOG_HPP
