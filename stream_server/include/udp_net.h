#ifndef UDP_NET_H
#define UDP_NET_H

#include "parse_conf.h"

int broadcast_msg(cam_conf* confs, int confs_size, const char* msg, size_t msg_size);

#endif // UDP_NET_H
