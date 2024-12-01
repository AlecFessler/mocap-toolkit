#ifndef CONFIG_H
#define CONFIG_H

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

struct config {
  std::string server_ip;
  std::string tcp_port;
  std::string udp_port;
  std::string enc_speed;
  std::string enc_quality;
  int recording_cpu;
  int dma_buffers;
  int frame_width;
  int frame_height;
  int frame_duration_min;
  int frame_duration_max;
  int fps;
};

config parse_config(const std::string& filename);

#endif // CONFIG_H
