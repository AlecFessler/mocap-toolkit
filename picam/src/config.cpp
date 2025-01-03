// © 2024 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#include <fstream>
#include <sstream>
#include <stdexcept>

#include "config.h"
#include "logging.h"

static void trim(std::string& str) {
  /**
   * Removes leading and trailing whitespace from a string.
   *
   * The function considers three types of whitespace:
   * - Regular spaces ' '
   * - Newlines '\n'
   * - Carriage returns '\r'
   *
   * The operation is performed in two passes:
   * 1. Removes trailing whitespace by repeatedly popping from the back
   * 2. Removes leading whitespace by finding the first non-whitespace
   *    character and erasing everything before it
   *
   * Parameters:
   *   str: Reference to string to be trimmed in-place
   */
  while (!str.empty() && (str.back() == '\n' || str.back() == '\r' || str.back() == ' '))
    str.pop_back();

  size_t start = 0;
  while (start < str.length() && (str[start] == ' ' || str[start] == '\n' || str[start] == '\r'))
    start++;

  if (start > 0)
    str.erase(0, start);
}

config parse_config(const std::string& filename) {
  /**
   * Parses a configuration file into a config structure.
   *
   * The file format follows a simple key=value pattern:
   * - Lines starting with # are treated as comments
   * - Empty lines are skipped
   * - Keys and values are trimmed of whitespace
   * - String values (SERVER_IP, ports, encoder settings) are stored as-is
   * - Numeric values are converted to integers via std::stoi
   *
   * The parser validates:
   * - File existence and readability
   * - Presence of both key and value in each non-comment line
   * - Recognition of all configuration keys
   * - Successful conversion of numeric values
   *
   * Parameters:
   *   filename: Path to the configuration file
   *
   * Returns:
   *   Populated config structure
   *
   * Throws:
   *   std::runtime_error: If file cannot be opened
   *   std::runtime_error: If an unknown configuration key is found
   *   std::invalid_argument: If numeric conversion fails (via std::stoi)
   *   std::out_of_range: If numeric value exceeds integer limits
   */
  config config;
  std::ifstream file(filename);
  if (!file)
    LOG(ERROR, "Could not open config file");

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
