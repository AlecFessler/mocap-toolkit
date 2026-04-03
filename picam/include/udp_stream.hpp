// © 2025 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#ifndef UDP_STREAM_HPP
#define UDP_STREAM_HPP

#include <arpa/inet.h>
#include <chrono>
#include <cstdint>
#include <span>
#include <string_view>

#include "protocol.hpp"

class UdpStream {
private:
  int m_fd;
  struct sockaddr_in m_server_addr;
  uint32_t m_sequence;
  uint8_t m_send_buffer[UDP_MTU];

public:
  UdpStream(uint16_t port, std::string_view ip);
  ~UdpStream() noexcept;

  // Fragment and send an encoded frame via UDP
  void stream_packet(
    std::chrono::nanoseconds timestamp,
    std::span<uint8_t> data
  );
};

#endif // UDP_STREAM_HPP
