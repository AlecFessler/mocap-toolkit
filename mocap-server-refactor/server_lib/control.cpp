#include <cstring>
#include <sys/socket.h>
#include <unistd.h>
#include <utility>

#include "control.hpp"

namespace mocap {

constexpr const char* STOP_SENTINEL = "STOP";

ControlSocket::ControlSocket(int fd, sockaddr_in dest)
  : m_fd(fd), m_dest(dest) {}

ControlSocket::ControlSocket(ControlSocket&& other) noexcept
  : m_fd(std::exchange(other.m_fd, -1)), m_dest(other.m_dest) {}

ControlSocket& ControlSocket::operator=(ControlSocket&& other) noexcept {
  if (this != &other) {
    if (m_fd >= 0)
      close(m_fd);
    m_fd = std::exchange(other.m_fd, -1);
    m_dest = other.m_dest;
  }
  return *this;
}

ControlSocket::~ControlSocket() {
  if (m_fd >= 0)
    close(m_fd);
}

std::expected<ControlSocket, Error> ControlSocket::open(in_addr broadcast_addr, uint16_t port) {
  int fd = socket(AF_INET, SOCK_DGRAM, 0);
  if (fd < 0)
    return std::unexpected(errno_error("failed to create control socket"));

  int enable = 1;
  if (setsockopt(fd, SOL_SOCKET, SO_BROADCAST, &enable, sizeof(enable)) < 0) {
    Error err = errno_error("failed to enable broadcast on control socket");
    close(fd);
    return std::unexpected(err);
  }

  sockaddr_in dest{};
  dest.sin_family = AF_INET;
  dest.sin_port = htons(port);
  dest.sin_addr = broadcast_addr;

  return ControlSocket(fd, dest);
}

std::expected<void, Error> ControlSocket::broadcast(const void* msg, size_t len) const {
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

std::expected<void, Error> ControlSocket::broadcast_start(uint64_t timestamp) const {
  return broadcast(&timestamp, sizeof(timestamp));
}

std::expected<void, Error> ControlSocket::broadcast_stop() const {
  return broadcast(STOP_SENTINEL, std::strlen(STOP_SENTINEL));
}

} // namespace mocap
