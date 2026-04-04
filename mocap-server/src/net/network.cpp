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

// ========== TCP Video Stream ==========

int setup_stream_listener(uint16_t port) {
  return setup_ctrl_listener(port); // same TCP listener setup
}

int accept_stream_conn(int listenfd) {
  return accept_ctrl_conn(listenfd); // same accept logic
}

// Read exactly `size` bytes from fd, handling partial reads.
// Returns 0 on success, -1 on error/disconnect, -2 on timeout.
static int read_exact(int fd, void* buf, uint32_t size) {
  uint8_t* ptr = static_cast<uint8_t*>(buf);
  uint32_t received = 0;
  while (received < size) {
    ssize_t n = read(fd, ptr + received, size - received);
    if (n < 0) {
      if (errno == EINTR) continue;
      if (errno == EAGAIN || errno == EWOULDBLOCK) return -2;
      return -1;
    }
    if (n == 0) return -1; // connection closed
    received += n;
  }
  return 0;
}

int recv_tcp_frame(
  int fd,
  uint8_t* buf,
  uint32_t buf_size,
  uint64_t* out_timestamp,
  uint32_t* out_sequence
) {
  // read frame header
  tcp_frame_header header;
  int ret = read_exact(fd, &header, TCP_FRAME_HEADER_SIZE);
  if (ret == -2) return 0;  // timeout
  if (ret < 0) return -1;   // error/disconnect

  if (header.size > buf_size) return -1; // frame too large

  // read frame payload
  ret = read_exact(fd, buf, header.size);
  if (ret < 0) return -1;

  *out_timestamp = header.timestamp;
  *out_sequence = header.sequence;
  return static_cast<int>(header.size);
}
