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

  int conn_tcp();
  int tcpfd;

  int bind_udp();
  int udpfd;

private:
  std::string server_ip;
  std::string tcp_port;
  std::string udp_port;
};

#endif
