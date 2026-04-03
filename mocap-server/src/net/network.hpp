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

// UDP frame receive
int setup_udp_recv(uint16_t port);

// UDP frame reassembly buffer
constexpr uint32_t MAX_FRAGMENTS = 256;

struct reassembly_buf {
  uint32_t sequence;
  uint64_t timestamp;
  uint16_t frag_total;
  uint16_t frags_received;
  uint8_t frag_mask[MAX_FRAGMENTS / 8]; // bitmask of received fragments
  uint8_t data[262144]; // max encoded frame (256KB)
  uint32_t total_size;
  bool active;
};

void reassembly_init(reassembly_buf* buf);
// Returns complete frame size if all fragments received, 0 if incomplete
int reassembly_add_fragment(
  reassembly_buf* buf,
  const uint8_t* packet,
  uint32_t packet_size,
  uint64_t* out_timestamp
);

#endif // NETWORK_HPP
