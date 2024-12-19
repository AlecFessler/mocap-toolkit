#ifndef NETWORK_H
#define NETWORK_H

#include "parse_conf.h"

int broadcast_msg(cam_conf* confs, int confs_size, const char* msg, size_t msg_size);
int setup_stream(cam_conf* conf);
int accept_conn(int sockfd);
ssize_t recv_from_stream(int clientfd, char* buf, size_t size);

#endif // NETWORK_H
