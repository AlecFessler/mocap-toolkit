#ifndef PARSE_CONF_H
#define PARSE_CONF_H

#include <arpa/inet.h>
#include <stdint.h>

#define CAM_NAME_LEN 9

struct stream_conf {
  uint32_t frame_width;
  uint32_t frame_height;
  uint32_t fps;
};

struct cam_conf {
  struct in_addr eth_ip;
  struct in_addr wifi_ip;
  uint16_t tcp_port;
  uint16_t udp_port;
  uint8_t id;
  char name[CAM_NAME_LEN]; // expected name format rpicamXX\0 where XX is a counter from 00-99
};

int count_cameras(const char* fpath);
int parse_conf(
  struct stream_conf* stream_conf,
  struct cam_conf* confs,
  int count
);

#endif
