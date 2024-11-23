// © 2024 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.


#include <arpa/inet.h>
#include <stdexcept>
#include <string>
#include <memory>
#include <unistd.h>
#include "connection.h"
#include "logger.h"

extern std::unique_ptr<logger_t> logger;

connection::connection(
  const std::string& server_ip,
  const std::string& port
) noexcept :
  server_ip(server_ip),
  port(port),
  sockfd(-1) {}

connection::~connection() noexcept {
  disconn_sock();
}

int connection::conn_sock() {
  if (sockfd >= 0) return 0;

  sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd < 0) {
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, "Failed to create socket");
    return -errno;
  }

  struct sockaddr_in server_addr;
  server_addr.sin_family = AF_INET;

  int port_num = std::stoi(port);
  if (port_num < 1 || port_num > 65535) {
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, "Invalid port number");
    return -EINVAL;
  }

  server_addr.sin_port = htons(port_num);

  if (inet_pton(AF_INET, server_ip.c_str(), &server_addr.sin_addr) <= 0) {
    if (errno == 0) {
      logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, "Invalid IP address format");
      return -EINVAL;
    } else {
      logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, "System error during IP address conversion");
      return -errno;
    }
  }

  if (connect(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, "Failed to connect to server");
    return -errno;
  }

  return 0;
}

void connection::disconn_sock() noexcept {
  if (sockfd >= 0) {
    close(sockfd);
    sockfd = -1;
  }
}
