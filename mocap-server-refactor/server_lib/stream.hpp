#ifndef MOCAP_STREAM_HPP
#define MOCAP_STREAM_HPP

#include <cstdint>
#include <expected>
#include <string>

#include "error.hpp"
#include "fd.hpp"

namespace mocap {

// Listening socket for one camera. Cameras dial out to the server, so each
// camera's identity is established by which listener accepted it.
class stream_listener {
public:
  static std::expected<stream_listener, error> open(uint16_t port, std::string name);

  std::expected<unique_fd, error> accept() const;

  int fd() const { return m_fd.get(); }
  const std::string& name() const { return m_name; }

private:
  stream_listener(unique_fd fd, std::string name);

  unique_fd m_fd;
  std::string m_name;
};

} // namespace mocap

#endif // MOCAP_STREAM_HPP
