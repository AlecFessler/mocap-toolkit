// © 2024 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#include <fstream>
#include <sstream>
#include <stdexcept>
#include "config.h"

static void trim(std::string& str) {
  while (!str.empty() && (str.back() == '\n' || str.back() == '\r' || str.back() == ' '))
    str.pop_back();

  size_t start = 0;
  while (start < str.length() && (str[start] == ' ' || str[start] == '\n' || str[start] == '\r'))
    start++;

  if (start > 0)
    str.erase(0, start);
}

config parse_config(const std::string& filename) {
  config config;
  std::ifstream file(filename);
  if (!file)
    throw std::runtime_error("Could not open config file: " + filename);

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;  // Allow comments with #

    std::istringstream iss(line);
    std::string key, value;
    if (std::getline(iss, key, '=') && std::getline(iss, value)) {
      trim(key);
      trim(value);

      if (key == "SERVER_IP")
        config.server_ip = value;
      else if (key == "TCP_PORT")
        config.tcp_port = value;
      else if (key == "UDP_PORT")
         config.udp_port = value;
      else if (key == "ENC_SPEED")
        config.enc_speed = value;
      else if (key == "ENC_QUALITY")
        config.enc_quality = value;
      else if (key == "RECORDING_CPU")
        config.recording_cpu = std::stoi(value);
      else if (key == "DMA_BUFFERS")
        config.dma_buffers = std::stoi(value);
      else if (key == "FRAME_WIDTH")
        config.frame_width = std::stoi(value);
      else if (key == "FRAME_HEIGHT")
        config.frame_height = std::stoi(value);
      else if (key == "FRAME_DURATION_MIN")
        config.frame_duration_min = std::stoi(value);
      else if (key == "FRAME_DURATION_MAX")
        config.frame_duration_max = std::stoi(value);
      else if (key == "FPS")
        config.fps = std::stoi(value);
      else
        throw std::runtime_error("Unknown config key: " + key);
    }
  }

  return config;
}
// © 2024 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#include <fstream>
#include <sstream>
#include <stdexcept>
#include "config.h"

static void trim(std::string& str) {
  while (!str.empty() && (str.back() == '\n' || str.back() == '\r' || str.back() == ' '))
    str.pop_back();

  size_t start = 0;
  while (start < str.length() && (str[start] == ' ' || str[start] == '\n' || str[start] == '\r'))
    start++;

  if (start > 0)
    str.erase(0, start);
}

config parse_config(const std::string& filename) {
  config config;
  std::ifstream file(filename);
  if (!file)
    throw std::runtime_error("Could not open config file: " + filename);

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;  // Allow comments with #

    std::istringstream iss(line);
    std::string key, value;
    if (std::getline(iss, key, '=') && std::getline(iss, value)) {
      trim(key);
      trim(value);

      if (key == "SERVER_IP")
        config.server_ip = value;
      else if (key == "TCP_PORT")
        config.tcp_port = value;
      else if (key == "UDP_PORT")
         config.udp_port = value;
      else if (key == "ENC_SPEED")
        config.enc_speed = value;
      else if (key == "ENC_QUALITY")
        config.enc_quality = value;
      else if (key == "RECORDING_CPU")
        config.recording_cpu = std::stoi(value);
      else if (key == "DMA_BUFFERS")
        config.dma_buffers = std::stoi(value);
      else if (key == "FRAME_WIDTH")
        config.frame_width = std::stoi(value);
      else if (key == "FRAME_HEIGHT")
        config.frame_height = std::stoi(value);
      else if (key == "FRAME_DURATION_MIN")
        config.frame_duration_min = std::stoi(value);
      else if (key == "FRAME_DURATION_MAX")
        config.frame_duration_max = std::stoi(value);
      else if (key == "FPS")
        config.fps = std::stoi(value);
      else
        throw std::runtime_error("Unknown config key: " + key);
    }
  }

  return config;
}
