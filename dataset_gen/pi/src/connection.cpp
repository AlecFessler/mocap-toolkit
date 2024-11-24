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

connection::connection()
  noexcept :
  sockfd(-1),
  server_ip("UNSET_SERVER"),
  port("UNSET_PORT") {}


connection::connection(
  std::string& server_ip,
  std::string& port
) noexcept :
  sockfd(-1),
  server_ip(server_ip),
  port(port) {}

connection::connection(connection&& other) noexcept
    : sockfd(other.sockfd), server_ip(std::move(other.server_ip)), port(std::move(other.port)) {
    other.sockfd = -1;
}

connection& connection::operator=(connection&& other) noexcept {
  if (this != &other) {
    if (sockfd >= 0) close(sockfd);

    server_ip = std::move(other.server_ip);
    port = std::move(other.port);
    sockfd = other.sockfd;

    other.sockfd = -1;
  }
  return *this;
}

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

  while (connect(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
    if (errno == EINTR) continue;
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, "Failed to connect to server");
    return -errno;
  }

  logger->log(logger_t::level_t::DEBUG, __FILE__, __LINE__, "Connected to server");
  return 0;
}

void connection::disconn_sock() noexcept {
  if (sockfd >= 0) {
    close(sockfd);
    sockfd = -1;
  }
}
