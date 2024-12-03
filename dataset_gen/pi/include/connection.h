// © 2024 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#ifndef CONNECTION_H
#define CONNECTION_H

#include <string>
#include "config.h"

class connection {
public:
  connection() noexcept;
  connection(config& config, int64_t frame_duration) noexcept;
  ~connection() noexcept;

  int tcpfd;
  int conn_tcp();
  int stream_pkt(const uint8_t* data, size_t size);
  int end_stream();
  void discon_tcp();

  int udpfd;
  int bind_udp();
  size_t recv_msg(char* msg_buf, size_t size);

  int64_t timestamp;
  int64_t frame_duration;

private:
  std::string server_ip;
  std::string tcp_port;
  std::string udp_port;
};

#endif
