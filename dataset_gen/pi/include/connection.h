// © 2024 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#ifndef CONNECTION_H
#define CONNECTION_H

#include <string>

class connection {
public:
  connection() noexcept;
  connection(
    std::string& server_ip,
    std::string& tcp_port,
    std::string& udp_port
  ) noexcept;
  ~connection() noexcept;

  int tcpfd;
  int conn_tcp();
  int stream_pkt(const uint8_t* data, size_t size);
  void discon_tcp();

  int udpfd;
  int bind_udp();
  size_t recv_msg(char* msg_buf);

private:
  std::string server_ip;
  std::string tcp_port;
  std::string udp_port;
};

#endif
