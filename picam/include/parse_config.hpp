// © 2025 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#ifndef PARSE_CONFIG_HPP
#define PARSE_CONFIG_HPP

#include <cstdint>
#include <string>

struct config {
  std::string server_ip;
  uint16_t tcp_port;
  uint16_t udp_port;
  std::pair<uint32_t, uint32_t> resolution;
  uint32_t fps;
};

config parse_config(const std::string& filename);

#endif // PARSE_CONFIG_HPP
