#include <netinet/in.h>
#include <sys/socket.h>
#include <utility>

#include "stream.hpp"

namespace mocap {

namespace {
constexpr int LISTEN_BACKLOG = 1;
}

stream_listener::stream_listener(unique_fd fd, std::string name)
  : m_fd(std::move(fd)), m_name(std::move(name)) {}

std::expected<stream_listener, error> stream_listener::open(uint16_t port, std::string name) {
  unique_fd fd{socket(AF_INET, SOCK_STREAM, 0)};
  if (!fd.valid())
    return std::unexpected(errno_error("failed to create listen socket for " + name));

  // without this a crashed session leaves the port in TIME_WAIT and the next
  // session_start fails to bind
  int enable = 1;
  if (setsockopt(fd.get(), SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable)) < 0)
    return std::unexpected(errno_error("failed to set SO_REUSEADDR for " + name));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(port);

  if (bind(fd.get(), reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) < 0)
    return std::unexpected(errno_error("failed to bind listen socket for " + name));

  if (listen(fd.get(), LISTEN_BACKLOG) < 0)
    return std::unexpected(errno_error("failed to listen for " + name));

  return stream_listener(std::move(fd), std::move(name));
}

std::expected<unique_fd, error> stream_listener::accept() const {
  unique_fd conn{::accept(m_fd.get(), nullptr, nullptr)};
  if (!conn.valid())
    return std::unexpected(errno_error("failed to accept connection for " + m_name));

  return conn;
}

} // namespace mocap
