// © 2025 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#ifndef STOP_WATCHDOG_HPP
#define STOP_WATCHDOG_HPP

#include <atomic>
#include <pthread.h>

#include "udp_socket.hpp"

class StopWatchdog{
private:
  std::atomic<bool>& m_main_stop_flag;
  pthread_t m_this_thread;
  UdpSocket m_udpsock;

public:
  StopWatchdog(std::atomic<bool>& main_stop_flag);
  ~StopWatchdog();
  void launch(UdpSocket&& udpsock);
};

#endif // STOP_WATCHDOG_HPP
