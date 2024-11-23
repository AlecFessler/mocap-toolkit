// © 2024 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#ifndef CONNECTION_H
#define CONNECTION_H
#include <string>

typedef struct {
    int& sockfd;
    const std::string& server_ip;
    const std::string& port;
} conn_info_t;

#endif
