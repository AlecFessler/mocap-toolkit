// © 2025 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#ifndef TCP_STREAM_HPP
#define TCP_STREAM_HPP

#include <chrono>
#include <cstdint>
#include <span>
#include <string_view>

class TcpStream {
private:
  int m_fd;
  uint16_t m_port;
  std::string_view m_ip;
  uint8_t m_camera_id;
  uint32_t m_sequence;

  bool write_all(const void* buf, uint32_t size);

public:
  TcpStream(uint16_t port, std::string_view ip, uint8_t camera_id);
  ~TcpStream() noexcept;

  void connect_to_server();
  void disconnect();
  bool is_connected() const { return m_fd >= 0; }

  // Send an encoded frame over TCP
  void stream_packet(
    std::chrono::nanoseconds timestamp,
    std::span<uint8_t> data
  );
};

#endif // TCP_STREAM_HPP
