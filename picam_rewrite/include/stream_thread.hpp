#ifndef STREAM_THREAD_HPP
#define STREAM_THREAD_HPP

#include <cstdint>
#include <string_view>

#include "spsc_queue.hpp"
#include "tcp_socket.hpp"

class StreamThread {
private:
  TcpSocket m_tcpsock;
  struct consumer_q* m_packet_consumer_q_ptr;

public:
  StreamThread(
    uint16_t port,
    std::string_view ip,
    struct consumer_q* packet_consumer_q_ptr
  );
  ~StreamThread();
  void launch();
};

#endif // STREAM_THREAD_HPP
