#ifndef UDP_SOCKET_HPP
#define UDP_SOCKET_HPP

#include <chrono>
#include <cstdint>

class UdpSocket {
private:
  int fd;

public:
  UdpSocket(uint16_t port);
  UdpSocket(const UdpSocket& other) = delete;
  UdpSocket& operator=(const UdpSocket& other) = delete;
  UdpSocket(UdpSocket&& other) noexcept;
  UdpSocket& operator=(UdpSocket&& other) noexcept;
  ~UdpSocket() noexcept;

  std::chrono::nanoseconds recv_stream_ctl();
};

#endif // UDP_SOCKET_HPP
