#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <cstddef>
#include <sys/types.h>

#include "parse_conf.h"

int broadcast_msg(struct cam_conf* confs, int confs_size, const char* msg, size_t msg_size);
int setup_stream(struct cam_conf* conf);
int accept_conn(int sockfd);
ssize_t recv_from_stream(int clientfd, char* buf, size_t size);

#endif // NETWORK_HPP
