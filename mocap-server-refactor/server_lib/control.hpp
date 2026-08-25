#ifndef MOCAP_CONTROL_HPP
#define MOCAP_CONTROL_HPP

#include <cstdint>
#include <expected>
#include <netinet/in.h>

#include "error.hpp"
#include "fd.hpp"

namespace mocap {

// UDP socket for the session control plane. Every camera binds the same
// control port, so one datagram to the subnet broadcast address starts or
// stops the whole rig.
class ControlSocket {
public:
  static std::expected<ControlSocket, Error> open(in_addr broadcast_addr, uint16_t port);

  // cameras arm their interval timer against this absolute time, so delivery
  // jitter between cameras does not affect capture alignment
  std::expected<void, Error> broadcast_start(uint64_t timestamp) const;
  std::expected<void, Error> broadcast_stop() const;

private:
  ControlSocket(UniqueFd fd, sockaddr_in dest);

  std::expected<void, Error> broadcast(const void* msg, size_t len) const;

  UniqueFd m_fd;
  sockaddr_in m_dest;
};

} // namespace mocap

#endif // MOCAP_CONTROL_HPP
