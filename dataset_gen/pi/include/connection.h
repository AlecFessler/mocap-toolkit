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
    std::string& port
  ) noexcept;
  connection(connection&& other) noexcept;
  connection& operator=(connection&& other) noexcept;
  connection(const connection&) = delete;
  connection& operator=(const connection&) = delete;
  ~connection() noexcept;

  int conn_sock();
  void disconn_sock() noexcept;
  int sockfd;

private:
  std::string server_ip;
  std::string port;
};

#endif
