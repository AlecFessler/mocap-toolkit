#ifndef PARSE_CONF_H
#define PARSE_CONF_H

#include <arpa/inet.h>
#include <stdint.h>

#define CAM_NAME_LEN 9

typedef struct cam_conf {
  struct in_addr eth_ip;
  struct in_addr wifi_ip;
  uint16_t tcp_port;
  uint16_t udp_port;
  uint8_t id;
  char name[CAM_NAME_LEN]; // expected name format rpicamXX\0 where XX is a counter from 00-99
} cam_conf;

int8_t count_cameras(const char* fpath);
int8_t parse_conf(cam_conf* confs, int8_t count);

#endif
