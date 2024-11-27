// © 2024 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#include <arpa/inet.h>
#include <cstring>
#include <stdexcept>
#include <string>
#include <memory>
#include <unistd.h>
#include "connection.h"
#include "logger.h"

extern std::unique_ptr<logger_t> logger;

connection::connection()
  noexcept :
  tcpfd(-1),
  server_ip("UNSET_SERVER"),
  tcp_port("UNSET_PORT"),
  udp_port("UNSET_PORT") {}


connection::connection(
  std::string& server_ip,
  std::string& tcp_port,
  std::string& udp_port
) noexcept :
  tcpfd(-1),
  udpfd(-1),
  server_ip(server_ip),
  tcp_port(tcp_port),
  udp_port(udp_port) {}

connection::~connection() noexcept {
  if (tcpfd >= 0) {
    close(tcpfd);
    tcpfd = -1;
  }
  if (udpfd >= 0) {
    close(udpfd);
    udpfd = -1;
  }
}

int connection::conn_tcp() {
  if (tcpfd >= 0) return 0;

  tcpfd = socket(AF_INET, SOCK_STREAM, 0);
  if (tcpfd < 0) {
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, "Failed to create socket");
    return -errno;
  }

  struct sockaddr_in server_addr;
  server_addr.sin_family = AF_INET;

  int tcp_port_num = std::stoi(tcp_port);
  if (tcp_port_num < 1 || tcp_port_num > 65535) {
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, "Invalid tcp_port number");
    return -EINVAL;
  }

  server_addr.sin_port = htons(tcp_port_num);

  if (inet_pton(AF_INET, server_ip.c_str(), &server_addr.sin_addr) <= 0) {
    if (errno == 0) {
      logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, "Invalid IP address format");
      return -EINVAL;
    } else {
      logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, "System error during IP address conversion");
      return -errno;
    }
  }

  while (connect(tcpfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
    if (errno == EINTR) continue;
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, "Failed to connect to server");
    return -errno;
  }

  logger->log(logger_t::level_t::DEBUG, __FILE__, __LINE__, "Connected to server");
  return 0;
}

int connection::bind_udp() {
  if (udpfd >= 0) return 0;

  udpfd = socket(AF_INET, SOCK_DGRAM, 0);
  if (udpfd < 0) {
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, "Failed to create UDP socket");
    return -errno;
  }

  struct sockaddr_in udp_addr;
  memset(&udp_addr, 0, sizeof(udp_addr));
  udp_addr.sin_family = AF_INET;
  udp_addr.sin_addr.s_addr = htonl(INADDR_ANY);

  if (udp_port.empty() || udp_port.find_first_not_of("0123456789") != std::string::npos) {
      std::string error_message = "Invalid UDP_PORT value: " + udp_port;
      logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, error_message.c_str());
      close(udpfd);
      udpfd = -1;
      return -EINVAL;
  }

  int udp_port_num;
  try {
      udp_port_num = std::stoi(udp_port);
  } catch (const std::exception& e) {
      std::string error_message = "Error parsing UDP_PORT: " + udp_port + " - " + e.what();
      logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, error_message.c_str());
      close(udpfd);
      udpfd = -1;
      return -EINVAL;
  }

  if (udp_port_num < 1 || udp_port_num > 65535) {
      std::string error_message = "UDP_PORT out of range: " + std::to_string(udp_port_num);
      logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, error_message.c_str());
      close(udpfd);
      udpfd = -1;
      return -EINVAL;
  }

  udp_addr.sin_port = htons(udp_port_num);

  if (bind(udpfd, (struct sockaddr*)&udp_addr, sizeof(udp_addr)) < 0) {
      logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, "Failed to bind UDP socket");
      return -errno;
  }

  logger->log(logger_t::level_t::INFO, __FILE__, __LINE__, "UDP socket bound successfully");
  return 0;
}
