#include <array>
#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unistd.h>

#include "logging.hpp"
#include "udp_socket.hpp"

UdpSocket::UdpSocket(uint16_t port) {
  /**
   * Binds a a UDP socket to the provided port
   * and sets up for SIGIO to be emitted to
   * this process when data is ready in the
   * kernels socket buffer. Threads are
   * responsible for setting a sigmask with
   * SIGIO included if they with to handle
   * this event.
   */
  m_fd = socket(AF_INET, SOCK_DGRAM, 0);
  if (m_fd == -1) {
    std::string err_msg =
      "Failed to create UDP socket: "
      + std::string(strerror(errno));
    log_(ERROR, err_msg.c_str());
    throw std::runtime_error(err_msg);
  }

  struct sockaddr_in addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(port);

  int status = bind(
    m_fd,
    (struct sockaddr*)&addr,
    sizeof(addr)
  );
  if (status == -1) {
    close(m_fd);
    std::string err_msg =
      "Failed to bind UDP socket: "
      + std::string(strerror(errno));
    log_(ERROR, err_msg.c_str());
    throw std::runtime_error(err_msg);
  }

  status = fcntl(
    m_fd,
    F_SETFL,
    O_NONBLOCK | O_ASYNC
  );
  if (status == -1) {
    close(m_fd);
    std::string err_msg =
      "Failed to set properties for UDP socket: "
      + std::string(strerror(errno));
    log_(ERROR, err_msg.c_str());
    throw std::runtime_error(err_msg);
  }

  status = fcntl(
    m_fd,
    F_SETOWN,
    getpid()
  );
  if (status == -1) {
    close(m_fd);
    std::string err_msg =
      "Failed to set owner process for SIGIO: "
      + std::string(strerror(errno));
    log_(ERROR, err_msg.c_str());
    throw std::runtime_error(err_msg);
  }
}

UdpSocket::UdpSocket(UdpSocket&& other) noexcept :
  m_fd(other.m_fd) {
  other.m_fd = -1;
}

UdpSocket::~UdpSocket() {
  if (m_fd >= 0)
    close(m_fd);
}

std::chrono::nanoseconds UdpSocket::recv_stream_ctl() {
  /**
   * This function is expecting the server to
   * send one of two possible messages:
   *  1. A timestamp in the form of the number
   *     of nanoseconds since the unix epoch
   *  2. "STOP" indicating that it's done
   *     with our stream and we can proceed
   *     with clean up and exit
   *
   * In the first case, the timestamp will be
   * returned unformatted, and if it's the second
   * case then 0 is returned. For anything else
   * being received, an exception is thrown.
   *
   * There is the possibility that if an 8 byte
   * non timestamp message is received, it will
   * corrupt the program state. In practice, this
   * just means debug the server and move on.
   */
  constexpr uint32_t MAX_SIZE = 8; // bytes
  std::array<char, MAX_SIZE> buffer{};
  int32_t size = recvfrom(
    m_fd,
    buffer.data(),
    MAX_SIZE,
    0,
    nullptr,
    nullptr
  );
  if (size == -1) {
    std::string err_msg =
      "Failed to receive stream control message on UDP socket: "
      + std::string(strerror(errno));
    log_(ERROR, err_msg.c_str());
    throw std::runtime_error(err_msg);
  }

  if (size == 8) {
    uint64_t timestamp;
    std::memcpy(&timestamp, buffer.data(), sizeof(timestamp));
    return std::chrono::nanoseconds{timestamp};

  } else if (size == 4 && std::string_view(buffer.data(), 4) == "STOP") {
    return std::chrono::nanoseconds{0};

  } else {
    std::string err_msg =
      "Unexpected stream control received of size: "
      + std::to_string(size);
    log_(ERROR, err_msg.c_str());
    throw std::runtime_error(err_msg);
  }
}
