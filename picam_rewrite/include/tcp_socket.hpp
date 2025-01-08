// © 2025 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#ifndef TCP_SOCKET_HPP
#define TCP_SOCKET_HPP

#include <chrono>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

constexpr uint32_t PKT_MAX_SIZE = 262144; // 256KB

class TcpSocket {
private:
  int m_fd;
  uint16_t m_port;
  std::string_view m_ip;
  std::vector<uint8_t> m_send_buffer;

  void make_connection();

public:
  TcpSocket(
    uint16_t port,
    std::string_view ip
  );
  ~TcpSocket() noexcept;
  void stream_packet(
    std::chrono::nanoseconds timestamp,
    std::span<uint8_t> data
  );
};

#endif // TCP_SOCKET_HPP
