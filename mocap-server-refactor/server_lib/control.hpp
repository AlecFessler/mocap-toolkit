#ifndef MOCAP_CONTROL_HPP
#define MOCAP_CONTROL_HPP

#include <cstdint>
#include <expected>
#include <netinet/in.h>

#include "error.hpp"

namespace mocap {

// UDP socket for the session control plane. Every camera binds the same
// control port, so one datagram to the subnet broadcast address starts or
// stops the whole rig.
class control_socket {
public:
  static std::expected<control_socket, error> open(in_addr broadcast_addr, uint16_t port);

  control_socket(control_socket&& other) noexcept;
  control_socket& operator=(control_socket&& other) noexcept;
  control_socket(const control_socket&) = delete;
  control_socket& operator=(const control_socket&) = delete;
  ~control_socket();

  std::expected<void, error> broadcast_start(uint64_t timestamp) const;
  std::expected<void, error> broadcast_stop() const;

private:
  control_socket(int fd, sockaddr_in dest);

  std::expected<void, error> broadcast(const void* msg, size_t len) const;

  int m_fd;
  sockaddr_in m_dest;
};

} // namespace mocap

#endif // MOCAP_CONTROL_HPP
