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
  int fd;
  uint16_t port;
  std::string_view ip;
  std::vector<uint8_t> send_buffer;

  void make_connection();

public:
  TcpSocket(
    uint16_t port,
    std::string_view ip
  );
  ~TcpSocket() noexcept;
  void stream_pkt(
    std::chrono::nanoseconds timestamp,
    std::span<uint8_t> data
  );
};

#endif // TCP_SOCKET_HPP
