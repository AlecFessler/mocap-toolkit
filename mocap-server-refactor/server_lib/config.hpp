#ifndef MOCAP_CONFIG_HPP
#define MOCAP_CONFIG_HPP

#include <cstdint>
#include <expected>
#include <netinet/in.h>
#include <filesystem>
#include <string>
#include <vector>

#include "error.hpp"

namespace mocap {

struct stream_params {
  uint32_t frame_width;
  uint32_t frame_height;
  uint32_t fps;
};

struct camera {
  std::string name;
  uint8_t id;
  in_addr eth_ip;
  uint16_t tcp_port;
};

struct config {
  stream_params stream;
  in_addr control_broadcast;
  uint16_t control_port;
  std::vector<camera> cameras;
};

std::expected<config, error> parse_config(const std::filesystem::path& path);

} // namespace mocap

#endif // MOCAP_CONFIG_HPP
