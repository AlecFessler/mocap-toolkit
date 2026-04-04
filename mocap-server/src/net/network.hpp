#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <cstddef>
#include <cstdint>
#include <sys/types.h>

#include "core/parse_conf.h"
#include "core/protocol.hpp"

// TCP control channel
int setup_ctrl_listener(uint16_t port);
int accept_ctrl_conn(int listenfd);
int send_ctrl_msg(int fd, ctrl_msg_type type, const void* payload = nullptr, uint32_t length = 0);
int recv_ctrl_msg(int fd, ctrl_msg_header* header, void* payload, uint32_t max_payload);

// TCP video stream
int setup_stream_listener(uint16_t port);
int accept_stream_conn(int listenfd);

// Receive a complete encoded frame over TCP.
// Returns frame size on success, -1 on error/disconnect, 0 on timeout.
int recv_tcp_frame(
  int fd,
  uint8_t* buf,
  uint32_t buf_size,
  uint64_t* out_timestamp,
  uint32_t* out_sequence
);

#endif // NETWORK_HPP
