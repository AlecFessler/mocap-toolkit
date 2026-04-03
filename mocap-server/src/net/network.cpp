#include <arpa/inet.h>
#include <cerrno>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include "core/logging.h"
#include "net/network.hpp"

#define ACCEPT_TIMEOUT 30 // 30 sec for initial connection
#define CTRL_RECV_TIMEOUT 5 // 5 sec for control messages

// ========== TCP Control Channel ==========

int setup_ctrl_listener(uint16_t port) {
  char logstr[128];

  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    snprintf(logstr, sizeof(logstr), "Error creating TCP socket: %s", strerror(errno));
    log_write(ERROR, logstr);
    return -errno;
  }

  int enable = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int));

  // disable Nagle
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &enable, sizeof(int));

  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  addr.sin_addr.s_addr = INADDR_ANY;

  if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
    snprintf(logstr, sizeof(logstr), "Error binding control socket: %s", strerror(errno));
    log_write(ERROR, logstr);
    close(fd);
    return -errno;
  }

  if (listen(fd, 1) < 0) {
    snprintf(logstr, sizeof(logstr), "Error listening on control socket: %s", strerror(errno));
    log_write(ERROR, logstr);
    close(fd);
    return -errno;
  }

  return fd;
}

int accept_ctrl_conn(int listenfd) {
  char logstr[128];

  struct timeval timeout = {.tv_sec = ACCEPT_TIMEOUT, .tv_usec = 0};
  setsockopt(listenfd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

  struct sockaddr_in addr;
  socklen_t addr_len = sizeof(addr);
  int fd;
  do {
    fd = accept(listenfd, (struct sockaddr*)&addr, &addr_len);
  } while (fd < 0 && errno == EINTR);

  if (fd < 0) {
    if (errno == EWOULDBLOCK || errno == EAGAIN)
      log_write(ERROR, "Control accept timed out, no camera connected");
    else {
      snprintf(logstr, sizeof(logstr), "Error accepting control connection: %s", strerror(errno));
      log_write(ERROR, logstr);
    }
    return -errno;
  }

  int enable = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &enable, sizeof(int));

  // set recv timeout for control messages
  struct timeval ctrl_timeout = {.tv_sec = CTRL_RECV_TIMEOUT, .tv_usec = 0};
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &ctrl_timeout, sizeof(ctrl_timeout));

  return fd;
}

int send_ctrl_msg(int fd, ctrl_msg_type type, const void* payload, uint32_t length) {
  ctrl_msg_header header;
  header.type = type;
  header.length = length;

  ssize_t sent = write(fd, &header, CTRL_HEADER_SIZE);
  if (sent != CTRL_HEADER_SIZE)
    return -1;

  if (length > 0 && payload) {
    sent = write(fd, payload, length);
    if (sent != static_cast<ssize_t>(length))
      return -1;
  }

  return 0;
}

int recv_ctrl_msg(int fd, ctrl_msg_header* header, void* payload, uint32_t max_payload) {
  ssize_t received = 0;
  while (received < static_cast<ssize_t>(CTRL_HEADER_SIZE)) {
    ssize_t n = read(fd, reinterpret_cast<uint8_t*>(header) + received, CTRL_HEADER_SIZE - received);
    if (n <= 0)
      return -1;
    received += n;
  }

  if (header->length > 0) {
    if (header->length > max_payload)
      return -1;
    received = 0;
    while (received < static_cast<ssize_t>(header->length)) {
      ssize_t n = read(fd, reinterpret_cast<uint8_t*>(payload) + received, header->length - received);
      if (n <= 0)
        return -1;
      received += n;
    }
  }

  return header->type;
}

// ========== UDP Frame Receive ==========

int setup_udp_recv(uint16_t port) {
  char logstr[128];

  int fd = socket(AF_INET, SOCK_DGRAM, 0);
  if (fd < 0) {
    snprintf(logstr, sizeof(logstr), "Error creating UDP socket: %s", strerror(errno));
    log_write(ERROR, logstr);
    return -errno;
  }

  // large receive buffer for burst arrivals
  int rcvbuf = 1024 * 1024; // 1MB
  setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));

  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  addr.sin_addr.s_addr = INADDR_ANY;

  if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
    snprintf(logstr, sizeof(logstr), "Error binding UDP socket: %s", strerror(errno));
    log_write(ERROR, logstr);
    close(fd);
    return -errno;
  }

  // non-blocking for polling
  struct timeval timeout = {.tv_sec = 0, .tv_usec = 100000}; // 100ms
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

  return fd;
}

// ========== UDP Reassembly ==========

void reassembly_init(reassembly_buf* buf) {
  memset(buf, 0, sizeof(*buf));
  buf->active = false;
}

int reassembly_add_fragment(
  reassembly_buf* buf,
  const uint8_t* packet,
  uint32_t packet_size,
  uint64_t* out_timestamp
) {
  if (packet_size < UDP_HEADER_SIZE)
    return 0;

  const udp_frag_header* header = reinterpret_cast<const udp_frag_header*>(packet);
  const uint8_t* payload = packet + UDP_HEADER_SIZE;
  uint32_t payload_size = header->frag_size;

  if (payload_size + UDP_HEADER_SIZE > packet_size)
    return 0;

  // new sequence — reset buffer
  if (!buf->active || header->sequence != buf->sequence) {
    memset(buf->frag_mask, 0, sizeof(buf->frag_mask));
    buf->sequence = header->sequence;
    buf->timestamp = header->timestamp;
    buf->frag_total = header->frag_total;
    buf->frags_received = 0;
    buf->total_size = 0;
    buf->active = true;
  }

  // skip if old sequence
  if (header->sequence < buf->sequence)
    return 0;

  // skip if fragment index out of range
  if (header->frag_idx >= buf->frag_total || header->frag_idx >= MAX_FRAGMENTS)
    return 0;

  // skip if already received this fragment
  uint32_t byte_idx = header->frag_idx / 8;
  uint8_t bit_mask = 1 << (header->frag_idx % 8);
  if (buf->frag_mask[byte_idx] & bit_mask)
    return 0;

  // copy payload into correct position
  uint32_t offset = header->frag_idx * UDP_MAX_PAYLOAD;
  if (offset + payload_size > sizeof(buf->data))
    return 0;

  memcpy(buf->data + offset, payload, payload_size);
  buf->frag_mask[byte_idx] |= bit_mask;
  buf->frags_received++;
  buf->total_size += payload_size;

  // check if all fragments received
  if (buf->frags_received == buf->frag_total) {
    *out_timestamp = buf->timestamp;
    buf->active = false;
    return buf->total_size;
  }

  return 0;
}
