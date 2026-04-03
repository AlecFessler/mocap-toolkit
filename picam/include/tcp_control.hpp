// © 2025 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#ifndef TCP_CONTROL_HPP
#define TCP_CONTROL_HPP

#include <atomic>
#include <chrono>
#include <cstdint>
#include <string_view>

#include "protocol.hpp"

class TcpControl {
private:
  int m_fd;
  uint16_t m_port;
  std::string_view m_ip;
  uint8_t m_camera_id;

  void connect_to_server();

public:
  TcpControl(uint16_t port, std::string_view ip, uint8_t camera_id);
  ~TcpControl() noexcept;

  // Block until connected to server (retries with backoff)
  // Returns early if stop_flag is set
  void ensure_connected(const std::atomic<bool>& stop_flag);

  // Send a control message
  void send_msg(ctrl_msg_type type, const void* payload = nullptr, uint32_t length = 0);

  // Receive a control message (blocks until message or timeout)
  // Returns the message type, fills payload buffer
  // Returns -1 on error/disconnect
  int recv_msg(ctrl_msg_header* header, void* payload, uint32_t max_payload);

  // Check if connected
  bool is_connected() const { return m_fd >= 0; }

  // Close and reset
  void disconnect();

  // Get fd for poll/select
  int fd() const { return m_fd; }
};

#endif // TCP_CONTROL_HPP
