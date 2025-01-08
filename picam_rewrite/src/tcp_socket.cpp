// © 2025 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <span>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

#include "logging.hpp"
#include "tcp_socket.hpp"

TcpSocket::TcpSocket(
  uint16_t port,
  std::string_view ip
) :
  m_fd(-1),
  m_port(port),
  m_ip(ip),
  m_send_buffer(PKT_MAX_SIZE) {}

TcpSocket::~TcpSocket() noexcept {
  if (m_fd >= 0)
    close(m_fd);
}

void TcpSocket::make_connection() {
  m_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (m_fd == -1) {
    std::string err_msg =
      "Failed to create TCP socket: "
      + std::string(strerror(errno));
    log_(ERROR, err_msg.c_str());
    throw std::runtime_error(err_msg);
  }

  struct sockaddr_in server_addr;
  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(m_port);
  int status = inet_pton(
    AF_INET,
    m_ip.data(),
    &server_addr.sin_addr
  );
  if (status == -1) {
    std::string err_msg =
      "Error converting IP address: "
      + std::string(strerror(errno));
    log_(ERROR, err_msg.c_str());
    throw std::runtime_error(err_msg);
  }

  status = connect(
    m_fd,
    reinterpret_cast<struct sockaddr*>(&server_addr),
    sizeof(server_addr)
  );
  if (status == -1) {
    std::string err_msg =
      "Failed to connect to server: "
      + std::string(strerror(errno));
    log_(ERROR, err_msg.c_str());
    throw std::runtime_error(err_msg);
  }

  log_(INFO, "Connected to the server");
}

void TcpSocket::stream_pkt(
  std::chrono::nanoseconds timestamp,
  std::span<uint8_t> data
) {
  if (m_fd == -1)
    make_connection();

  constexpr uint32_t HEADER_SIZE = 12; // bytes
  if (data.size() > PKT_MAX_SIZE - HEADER_SIZE) {
    std::string err_msg =
      "Received a packet of size "
      + std::to_string(data.size())
      + ", max size is 262144 bytes (256KB)";
    log_(ERROR, err_msg.c_str());
    throw std::runtime_error(err_msg);
  }

  m_send_buffer.clear();

  uint64_t pkt_timestamp = timestamp.count();
  uint32_t pkt_size = data.size();

  std::memcpy( // timestamp
    m_send_buffer.data(),
    &pkt_timestamp,
    sizeof(pkt_timestamp)
  );
  std::memcpy( // packet size
    m_send_buffer.data() + sizeof(pkt_timestamp),
    &pkt_size,
    sizeof(pkt_size)
  );
  std::memcpy( // packet data
    m_send_buffer.data() + HEADER_SIZE,
    data.data(),
    pkt_size
  );

  const uint64_t total_bytes = pkt_size + HEADER_SIZE;
  uint64_t bytes_written = 0;
  while (bytes_written < total_bytes) {
    int64_t result = write(
      m_fd,
      m_send_buffer.data() + bytes_written,
      total_bytes - bytes_written
    );
    if (result == -1) {
      std::string err_msg =
        "Error while streaming packet: "
        + std::string(strerror(errno));
      log_(ERROR, err_msg.c_str());
      throw std::runtime_error(err_msg);
    }

    bytes_written += result;
  }
}
