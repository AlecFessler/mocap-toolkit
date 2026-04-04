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
#include "tcp_stream.hpp"

class StreamThread {
public:
  TcpStream m_tcp_stream;
  SPSCQueue<struct packet>& m_packet_queue;
  std::atomic<bool>& m_main_stop_flag;

  pthread_t m_this_thread;

  StreamThread(
    uint16_t stream_port,
    std::string_view server_ip,
    uint8_t camera_id,
    SPSCQueue<struct packet>& packet_queue,
    std::atomic<bool>& main_stop_flag
  );
  ~StreamThread();
  void launch();
};

#endif // STREAM_THREAD_HPP
