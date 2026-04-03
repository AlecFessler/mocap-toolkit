// © 2025 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unistd.h>

#include "logging.hpp"
#include "udp_stream.hpp"

UdpStream::UdpStream(uint16_t port, std::string_view ip)
  : m_fd(-1), m_sequence(0) {

  m_fd = socket(AF_INET, SOCK_DGRAM, 0);
  if (m_fd == -1) {
    std::string err_msg = "Failed to create UDP socket: " + std::string(strerror(errno));
    log_(ERROR, err_msg.c_str());
    throw std::runtime_error(err_msg);
  }

  // increase send buffer for burst sending of fragments
  int sndbuf = 512 * 1024; // 512KB
  setsockopt(m_fd, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));

  std::memset(&m_server_addr, 0, sizeof(m_server_addr));
  m_server_addr.sin_family = AF_INET;
  m_server_addr.sin_port = htons(port);
  inet_pton(AF_INET, ip.data(), &m_server_addr.sin_addr);
}

UdpStream::~UdpStream() noexcept {
  if (m_fd >= 0)
    close(m_fd);
}

void UdpStream::stream_packet(
  std::chrono::nanoseconds timestamp,
  std::span<uint8_t> data
) {
  uint32_t total_size = data.size();
  uint16_t frag_total = (total_size + UDP_MAX_PAYLOAD - 1) / UDP_MAX_PAYLOAD;
  uint64_t ts = timestamp.count();

  for (uint16_t frag = 0; frag < frag_total; frag++) {
    uint32_t offset = frag * UDP_MAX_PAYLOAD;
    uint16_t frag_size = std::min(
      static_cast<uint32_t>(UDP_MAX_PAYLOAD),
      total_size - offset
    );

    // build header
    udp_frag_header* header = reinterpret_cast<udp_frag_header*>(m_send_buffer);
    header->sequence = m_sequence;
    header->timestamp = ts;
    header->frag_idx = frag;
    header->frag_total = frag_total;
    header->frag_size = frag_size;

    // copy payload
    std::memcpy(m_send_buffer + UDP_HEADER_SIZE, data.data() + offset, frag_size);

    ssize_t sent = sendto(
      m_fd,
      m_send_buffer,
      UDP_HEADER_SIZE + frag_size,
      0,
      reinterpret_cast<struct sockaddr*>(&m_server_addr),
      sizeof(m_server_addr)
    );

    if (sent < 0) {
      // UDP send failure is non-fatal — frame will just be incomplete
      std::string err_msg = "UDP send failed: " + std::string(strerror(errno));
      log_(WARNING, err_msg.c_str());
      break;
    }
  }

  m_sequence++;
}
