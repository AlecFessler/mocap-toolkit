// © 2025 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <netinet/tcp.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <unistd.h>

#include "logging.hpp"
#include "protocol.hpp"
#include "tcp_stream.hpp"

TcpStream::TcpStream(uint16_t port, std::string_view ip, uint8_t camera_id)
  : m_fd(-1), m_port(port), m_ip(ip), m_camera_id(camera_id), m_sequence(0) {}

TcpStream::~TcpStream() noexcept {
  disconnect();
}

void TcpStream::disconnect() {
  if (m_fd >= 0) {
    close(m_fd);
    m_fd = -1;
  }
}

void TcpStream::connect_to_server() {
  m_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (m_fd == -1) {
    std::string err_msg = "Failed to create stream TCP socket: " + std::string(strerror(errno));
    log_(ERROR, err_msg.c_str());
    throw std::runtime_error(err_msg);
  }

  // disable Nagle — we want frames sent immediately
  int flag = 1;
  setsockopt(m_fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

  // keepalive to detect dead server
  setsockopt(m_fd, SOL_SOCKET, SO_KEEPALIVE, &flag, sizeof(flag));
  int idle = 1, intvl = 1, cnt = 2;
  setsockopt(m_fd, IPPROTO_TCP, TCP_KEEPIDLE, &idle, sizeof(idle));
  setsockopt(m_fd, IPPROTO_TCP, TCP_KEEPINTVL, &intvl, sizeof(intvl));
  setsockopt(m_fd, IPPROTO_TCP, TCP_KEEPCNT, &cnt, sizeof(cnt));

  // large send buffer for frame bursts
  int sndbuf = 512 * 1024;
  setsockopt(m_fd, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));

  // send timeout — if server stalls, don't block forever
  struct timeval snd_timeout = {.tv_sec = 1, .tv_usec = 0};
  setsockopt(m_fd, SOL_SOCKET, SO_SNDTIMEO, &snd_timeout, sizeof(snd_timeout));

  struct sockaddr_in server_addr;
  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(m_port);
  inet_pton(AF_INET, m_ip.data(), &server_addr.sin_addr);

  int status = connect(m_fd, reinterpret_cast<struct sockaddr*>(&server_addr), sizeof(server_addr));
  if (status == -1) {
    std::string err_msg = "Stream TCP connect failed: " + std::string(strerror(errno));
    log_(ERROR, err_msg.c_str());
    close(m_fd);
    m_fd = -1;
    throw std::runtime_error(err_msg);
  }

  // identify this stream connection with our camera ID (reuse control message framing)
  ctrl_msg_header hdr;
  hdr.type = CTRL_IDENTIFY;
  hdr.length = sizeof(m_camera_id);
  if (!write_all(&hdr, CTRL_HEADER_SIZE) || !write_all(&m_camera_id, sizeof(m_camera_id))) {
    disconnect();
    throw std::runtime_error("Failed to send stream identify");
  }

  log_(INFO, "Connected to server stream channel");
}

bool TcpStream::write_all(const void* buf, uint32_t size) {
  const uint8_t* ptr = static_cast<const uint8_t*>(buf);
  uint32_t sent = 0;
  while (sent < size) {
    ssize_t n = write(m_fd, ptr + sent, size - sent);
    if (n <= 0) {
      if (n < 0 && errno == EINTR) continue;
      return false;
    }
    sent += n;
  }
  return true;
}

void TcpStream::stream_packet(
  std::chrono::nanoseconds timestamp,
  std::span<uint8_t> data
) {
  if (m_fd < 0) return;

  tcp_frame_header header;
  header.sequence = m_sequence;
  header.timestamp = timestamp.count();
  header.size = data.size();

  if (!write_all(&header, TCP_FRAME_HEADER_SIZE) ||
      !write_all(data.data(), data.size())) {
    std::string err_msg = "Stream write failed: " + std::string(strerror(errno));
    log_(WARNING, err_msg.c_str());
    disconnect();
    return;
  }

  m_sequence++;
}
