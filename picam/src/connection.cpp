// © 2024 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#include <arpa/inet.h>
#include <cstring>
#include <errno.h>
#include <stdexcept>
#include <string>
#include <memory>
#include <unistd.h>

#include "connection.h"
#include "logging.h"

static const int MAX_RETRIES = 3;
static constexpr char END_STREAM[] = "EOSTREAM";

connection::connection()
  noexcept :
  /**
   * Creates a connection object with default values.
   *
   * Initializes a connection in an unconfigured state with invalid
   * file descriptors (-1) and placeholder network settings. This
   * constructor is primarily used for declaring connection objects
   * that will be configured later.
   */
  tcpfd(-1),
  udpfd(-1),
  server_ip("UNSET_SERVER"),
  tcp_port("UNSET_PORT"),
  udp_port("UNSET_PORT") {}


connection::connection(
  config& config
) noexcept :
  /**
   * Creates a connection object with specific network settings.
   *
   * Initializes a connection with server address and port information
   * while maintaining invalid file descriptors (-1). The actual network
   * connections are established later via conn_tcp() and bind_udp() to
   * allow for error handling and reconnection attempts.
   *
   * Parameters:
   *   server_ip: IPv4 address of the streaming server
   *   tcp_port:  Port number for streaming video data
   *   udp_port:  Port number for receiving control messages
   */
  tcpfd(-1),
  udpfd(-1),
  server_ip(config.server_ip),
  tcp_port(config.tcp_port),
  udp_port(config.udp_port) {}

connection::~connection() noexcept {
  /**
   * Safely closes any open network connections.
   *
   * Ensures proper cleanup of system resources by closing both TCP
   * and UDP sockets if they were opened. The file descriptors are
   * set to -1 after closing to maintain a consistent invalid state,
   * though this is technically unnecessary in a destructor.
   */
  if (tcpfd >= 0) {
    close(tcpfd);
    tcpfd = -1;
  }
  if (udpfd >= 0) {
    close(udpfd);
    udpfd = -1;
  }
}

int connection::conn_tcp() {
  /**
   * Establishes a TCP connection to the streaming server.
   *
   * Creates and connects a TCP socket using the configured server_ip
   * and tcp_port. The connection process includes:
   * 1. Socket creation with IPv4 and TCP protocol
   * 2. Port number validation (1-65535)
   * 3. IP address parsing and validation
   * 4. Connection establishment with retry on EINTR
   *
   * The method is idempotent - if a connection exists, it returns
   * success without creating a new one. This allows repeated calls
   * in retry loops without resource leaks.
   *
   * Returns:
   *   0 on success
   *   -errno on system calls failures
   *   -EINVAL on invalid port or IP address
   */
  if (tcpfd >= 0) return 0;
  char logstr[128];

  tcpfd = socket(AF_INET, SOCK_STREAM, 0);
  if (tcpfd < 0) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Failed to create socket: %s",
      strerror(errno)
    );
    LOG(ERROR, logstr);
    return -errno;
  }

  struct sockaddr_in server_addr;
  server_addr.sin_family = AF_INET;

  int tcp_port_num = std::stoi(tcp_port);
  if (tcp_port_num < 1 || tcp_port_num > 65535) {
    LOG(ERROR, "Invalid tcp_port number");
    return -EINVAL;
  }

  server_addr.sin_port = htons(tcp_port_num);

  if (inet_pton(AF_INET, server_ip.c_str(), &server_addr.sin_addr) <= 0) {
    if (errno == 0) {
      LOG(ERROR, "Invalid IP address format");
      return -EINVAL;
    } else {
      snprintf(
        logstr,
        sizeof(logstr),
        "Error during IP address conversion: %s",
        strerror(errno)
      );
      LOG(ERROR, logstr);
      return -errno;
    }
  }

  while (connect(tcpfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
    if (errno == EINTR) continue;
    LOG(ERROR, "Failed to connect to server");
    return -errno;
  }

  LOG(DEBUG, "Connected to server");
  return 0;
}

int connection::stream_pkt(const uint8_t* data, uint32_t size) {
  char logstr[128];

  uint64_t timestamp = frame_timestamps.front();
  frame_timestamps.pop();
  uint64_t pkt_size = size + sizeof(size) + sizeof(timestamp);
  uint8_t pkt[pkt_size];

  memcpy(
    pkt,
    (const uint8_t*)&timestamp,
    sizeof(timestamp)
  );
  memcpy(
    pkt + sizeof(timestamp),
    (const uint8_t*)&size,
    sizeof(size)
  );
  memcpy(
    pkt + sizeof(timestamp) + sizeof(size),
    data,
    size
  );

  int retries = 0;
  size_t total_written = 0;
  while (total_written < pkt_size) {
    if (tcpfd < 0) {
      LOG(WARNING, "Not connected to server, trying to connect");
      while (tcpfd < 0) {
        int ret = conn_tcp();
        if (ret < 0) {
          if (retries++ == MAX_RETRIES) {
            LOG(WARNING, "No more connection retries");
            return ret;
          }
          LOG(WARNING, "Failed to connect, retrying");
          continue;
        }
      }
    }

    ssize_t result = write(
      tcpfd,
      pkt + total_written,
      pkt_size - total_written
    );

    if (result < 0) {
      if (errno == EINTR) continue;
      if (errno == EPIPE || errno == ECONNRESET) {
        LOG(WARNING, "Server disconnected while streaming a frame packet");
        close(tcpfd);
        tcpfd = -1;
        continue;
      }
      snprintf(
        logstr,
        sizeof(logstr),
        "Error transmitting frame packet: %s",
        strerror(errno)
      );
      LOG(ERROR, logstr);
      return -errno;
    }

    total_written += result;
  }
  return 0;
}

int connection::end_stream() {
  char logstr[128];
  size_t end_stream_size = sizeof(END_STREAM) - 1;

  int retries = 0;
  size_t total_written = 0;
  while (total_written < end_stream_size) {
    if (tcpfd < 0) {
      LOG(WARNING, "Not connected to server, trying to connect");
      while (tcpfd < 0) {
        int ret = conn_tcp();
        if (ret < 0) {
          if (retries++ == MAX_RETRIES) {
            LOG(WARNING, "No more connection retries");
            return ret;
          }
          LOG(WARNING, "Failed to connect, retrying");
          continue;
        }
      }
    }

    ssize_t result = write(
      tcpfd,
      END_STREAM + total_written,
      end_stream_size - total_written
    );

    if (result < 0) {
      if (errno == EINTR) continue;
      if (errno == EPIPE || errno == ECONNRESET) {
        LOG(WARNING, "Server disconnected while notifying end of stream");
        close(tcpfd);
        tcpfd = -1;
        continue;
      }
      snprintf(
        logstr,
        sizeof(logstr),
        "Error transmitting end of stream packet: %s",
        strerror(errno)
      );
      LOG(ERROR, logstr);
      return -errno;
    }

    total_written += result;
  }

  return 0;
}

void connection::discon_tcp() {
  /**
   * Disconnects from the tcp socket
   */
  if (tcpfd >= 0) {
    close(tcpfd);
    tcpfd = -1;
  }
}

int connection::bind_udp() {
  /**
   * Creates and binds a UDP socket for receiving control messages.
   *
   * Sets up a UDP socket bound to all interfaces (INADDR_ANY) on
   * the configured udp_port. The initialization process includes:
   * 1. Socket creation with IPv4 and UDP protocol
   * 2. Port number validation (1-65535)
   * 3. Socket binding with retry on EINTR
   *
   * The method is idempotent - if a socket is already bound, it
   * returns success without creating a new one. This supports
   * safe retry attempts.
   *
   * Returns:
   *   0 on success
   *   -errno on system call failures
   *   -EINVAL on invalid port configuration
   */
  if (udpfd >= 0) return 0;
  char logstr[128];

  udpfd = socket(AF_INET, SOCK_DGRAM, 0);
  if (udpfd < 0) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Failed to create UDP socket: %s",
      strerror(errno)
    );
    LOG(ERROR, logstr);
    return -errno;
  }

  struct sockaddr_in udp_addr;
  memset(&udp_addr, 0, sizeof(udp_addr));
  udp_addr.sin_family = AF_INET;
  udp_addr.sin_addr.s_addr = htonl(INADDR_ANY);

  int udp_port_num = std::stoi(udp_port);
  if (udp_port_num < 1 || udp_port_num > 65535) {
    LOG(ERROR, "Invalid udp_port number");
    return -EINVAL;
  }
  udp_addr.sin_port = htons(udp_port_num);

  while (bind(udpfd, (struct sockaddr*)&udp_addr, sizeof(udp_addr)) < 0) {
    if (errno == EINTR) continue;
    snprintf(
      logstr,
      sizeof(logstr),
      "Failed to bind UDP socket: %s",
      strerror(errno)
    );
    LOG(ERROR, logstr);
    return -errno;
  }

  return 0;
}

size_t connection::recv_msg(char* msg_buf, size_t size) {
  return recvfrom(
    udpfd,
    msg_buf,
    size,
    0,
    NULL,
    NULL
  );
}
