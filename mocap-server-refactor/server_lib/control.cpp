#include <cstring>
#include <sys/socket.h>
#include <unistd.h>
#include <utility>

#include "control.hpp"

namespace mocap {

constexpr const char* STOP_SENTINEL = "STOP";

control_socket::control_socket(int fd, sockaddr_in dest)
  : m_fd(fd), m_dest(dest) {}

control_socket::control_socket(control_socket&& other) noexcept
  : m_fd(std::exchange(other.m_fd, -1)), m_dest(other.m_dest) {}

control_socket& control_socket::operator=(control_socket&& other) noexcept {
  if (this != &other) {
    if (m_fd >= 0)
      close(m_fd);
    m_fd = std::exchange(other.m_fd, -1);
    m_dest = other.m_dest;
  }
  return *this;
}

control_socket::~control_socket() {
  if (m_fd >= 0)
    close(m_fd);
}

std::expected<control_socket, error> control_socket::open(in_addr broadcast_addr, uint16_t port) {
  int fd = socket(AF_INET, SOCK_DGRAM, 0);
  if (fd < 0)
    return std::unexpected(errno_error("failed to create control socket"));

  int enable = 1;
  if (setsockopt(fd, SOL_SOCKET, SO_BROADCAST, &enable, sizeof(enable)) < 0) {
    error err = errno_error("failed to enable broadcast on control socket");
    close(fd);
    return std::unexpected(err);
  }

  sockaddr_in dest{};
  dest.sin_family = AF_INET;
  dest.sin_port = htons(port);
  dest.sin_addr = broadcast_addr;

  return control_socket(fd, dest);
}

std::expected<void, error> control_socket::broadcast(const void* msg, size_t len) const {
  ssize_t sent = sendto(
    m_fd,
    msg,
    len,
    0,
    reinterpret_cast<const sockaddr*>(&m_dest),
    sizeof(m_dest)
  );
  if (sent < 0)
    return std::unexpected(errno_error("failed to broadcast control message"));

  if (static_cast<size_t>(sent) != len)
    return std::unexpected(invalid("control message truncated in transit"));

  return {};
}

std::expected<void, error> control_socket::broadcast_start(uint64_t timestamp) const {
  return broadcast(&timestamp, sizeof(timestamp));
}

std::expected<void, error> control_socket::broadcast_stop() const {
  return broadcast(STOP_SENTINEL, std::strlen(STOP_SENTINEL));
}

} // namespace mocap
