// © 2025 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#include <arpa/inet.h>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <netinet/tcp.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

#include "logging.hpp"
#include "tcp_control.hpp"

TcpControl::TcpControl(uint16_t port, std::string_view ip, uint8_t camera_id)
  : m_fd(-1), m_port(port), m_ip(ip), m_camera_id(camera_id) {}

TcpControl::~TcpControl() noexcept {
  disconnect();
}

void TcpControl::disconnect() {
  if (m_fd >= 0) {
    close(m_fd);
    m_fd = -1;
  }
}

void TcpControl::connect_to_server() {
  m_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (m_fd == -1) {
    std::string err_msg = "Failed to create TCP socket: " + std::string(strerror(errno));
    log_(ERROR, err_msg.c_str());
    throw std::runtime_error(err_msg);
  }

  // disable Nagle for low-latency control messages
  int flag = 1;
  setsockopt(m_fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

  // aggressive TCP keepalive: detect dead server within 3 seconds
  setsockopt(m_fd, SOL_SOCKET, SO_KEEPALIVE, &flag, sizeof(flag));
  int idle = 1;   // start keepalive probes after 1s idle
  int intvl = 1;  // 1s between probes
  int cnt = 2;    // 2 failed probes = dead
  setsockopt(m_fd, IPPROTO_TCP, TCP_KEEPIDLE, &idle, sizeof(idle));
  setsockopt(m_fd, IPPROTO_TCP, TCP_KEEPINTVL, &intvl, sizeof(intvl));
  setsockopt(m_fd, IPPROTO_TCP, TCP_KEEPCNT, &cnt, sizeof(cnt));

  struct sockaddr_in server_addr;
  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(m_port);
  inet_pton(AF_INET, m_ip.data(), &server_addr.sin_addr);

  int status = connect(m_fd, reinterpret_cast<struct sockaddr*>(&server_addr), sizeof(server_addr));
  if (status == -1) {
    close(m_fd);
    m_fd = -1;
    return; // caller will retry
  }

  // send IDENTIFY message with camera ID
  send_msg(CTRL_IDENTIFY, &m_camera_id, sizeof(m_camera_id));

  log_(INFO, "Connected to server control channel");
}

void TcpControl::ensure_connected(const std::atomic<bool>& stop_flag) {
  uint32_t retry_ms = 500;
  constexpr uint32_t max_retry_ms = 5000;

  while (m_fd < 0 && !stop_flag.load(std::memory_order_relaxed)) {
    connect_to_server();
    if (m_fd >= 0)
      return;

    std::string msg = "Control connection failed, retrying in " + std::to_string(retry_ms) + "ms";
    log_(WARNING, msg.c_str());
    std::this_thread::sleep_for(std::chrono::milliseconds(retry_ms));
    retry_ms = std::min(retry_ms * 2, max_retry_ms);
  }
}

void TcpControl::send_msg(ctrl_msg_type type, const void* payload, uint32_t length) {
  if (m_fd < 0)
    return;

  ctrl_msg_header header;
  header.type = type;
  header.length = length;

  // send header
  ssize_t sent = write(m_fd, &header, CTRL_HEADER_SIZE);
  if (sent != CTRL_HEADER_SIZE) {
    disconnect();
    return;
  }

  // send payload if any
  if (length > 0 && payload) {
    sent = write(m_fd, payload, length);
    if (sent != static_cast<ssize_t>(length)) {
      disconnect();
      return;
    }
  }
}

int TcpControl::recv_msg(ctrl_msg_header* header, void* payload, uint32_t max_payload) {
  if (m_fd < 0)
    return -1;

  // receive header
  ssize_t received = 0;
  while (received < static_cast<ssize_t>(CTRL_HEADER_SIZE)) {
    ssize_t n = read(m_fd, reinterpret_cast<uint8_t*>(header) + received, CTRL_HEADER_SIZE - received);
    if (n <= 0) {
      disconnect();
      return -1;
    }
    received += n;
  }

  // receive payload
  if (header->length > 0) {
    if (header->length > max_payload) {
      disconnect();
      return -1;
    }
    received = 0;
    while (received < static_cast<ssize_t>(header->length)) {
      ssize_t n = read(m_fd, reinterpret_cast<uint8_t*>(payload) + received, header->length - received);
      if (n <= 0) {
        disconnect();
        return -1;
      }
      received += n;
    }
  }

  return header->type;
}
