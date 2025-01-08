// © 2025 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#ifndef STREAM_THREAD_HPP
#define STREAM_THREAD_HPP

#include <atomic>
#include <cstdint>
#include <pthread.h>
#include <string_view>

#include "packet_buffer.hpp"
#include "spsc_queue_wrapper.hpp"
#include "tcp_socket.hpp"

class StreamThread {
public:
  // everything is public for the stream thread function to access
  TcpSocket m_tcpsock;
  SPSCQueue<struct packet>& m_packet_queue;
  std::atomic<bool>& m_main_stop_flag;

  pthread_t m_this_thread;

  StreamThread(
    uint16_t port,
    std::string_view ip,
    SPSCQueue<struct packet>& packet_queue,
    std::atomic<bool>& main_stop_flag
  );
  ~StreamThread();
  void launch();
};

#endif // STREAM_THREAD_HPP
