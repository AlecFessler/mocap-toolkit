// © 2024 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#ifndef CONNECTION_H
#define CONNECTION_H

#include <queue>
#include <string>
#include "config.h"

class connection {
public:
  connection() noexcept;
  connection(config& config) noexcept;
  ~connection() noexcept;

  int tcpfd;
  int conn_tcp();
  int stream_pkt(const uint8_t* data, uint32_t size);
  int end_stream();
  void discon_tcp();

  int udpfd;
  int bind_udp();
  size_t recv_msg(char* msg_buf, size_t size);

  std::queue<uint64_t> frame_timestamps;

private:
  std::string server_ip;
  std::string tcp_port;
  std::string udp_port;
};

#endif
