// © 2025 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#include <fstream>
#include <sstream>
#include <stdexcept>

#include "logging.hpp"
#include "parse_config.hpp"

constexpr uint32_t hash_string(std::string_view str) {
  uint32_t hash = 5381;
  for (char c : str) {
    hash = ((hash << 5) + hash) + c;
  }
  return hash;
}

static void trim(std::string& str) {
  while (
    !str.empty() &&
    (str.back() == '\n' || str.back() == '\r' || str.back() == ' ')
  ) {
    str.pop_back();
  }

  uint32_t start = 0;
  while (
    start < str.length() &&
    (str[start] == '\n' || str[start] == '\r' || str[start] == ' ')
  ) {
    start++;
  }

  if (start > 0)
    str.erase(0, start);
}

config parse_config(const std::string& filename) {
  config config;
  std::ifstream file(filename);
  if (!file)
    log_(ERROR, "Failed to open config file");

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;

    std::istringstream iss(line);
    std::string key;
    std::string value;
    if (std::getline(iss, key, '=') && std::getline(iss, value)) {
      trim(key);
      trim(value);

      switch (hash_string(key)) {
        case hash_string("SERVER_IP"):
          config.server_ip = value;
          break;

        case hash_string("TCP_PORT"):
          config.tcp_port = std::stoi(value);
          break;

        case hash_string("UDP_PORT"):
          config.udp_port = std::stoi(value);
          break;

        case hash_string("FRAME_WIDTH"):
          config.resolution.first = std::stoi(value);
          break;

        case hash_string("FRAME_HEIGHT"):
          config.resolution.second = std::stoi(value);
          break;

        case hash_string("FPS"):
          config.fps = std::stoi(value);
          break;

        default:
          std::string err_msg =
            "Invalid key value pair in config: "
            + key + "=" + value;
          log_(ERROR, err_msg.c_str());
          throw std::runtime_error(err_msg);
      }
    }
  }

  return config;
}
