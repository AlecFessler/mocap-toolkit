// © 2024 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#ifndef CONNECTION_H
#define CONNECTION_H

#include <string>


class connection {
public:
  connection(
    const std::string& server_ip,
    const std::string& port
  ) noexcept;
  ~connection() noexcept;

  int conn_sock();
  void disconn_sock() noexcept;
  int sockfd;

private:
  const std::string server_ip;
  const std::string port;
};

#endif
