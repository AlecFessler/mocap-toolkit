// © 2024 Alec Fessler
// MIT License
// See LICENSE file in the project root for full license information.

#include <arpa/inet.h>
#include <cstring>
#include <stdexcept>
#include <string>
#include <memory>
#include <unistd.h>
#include "connection.h"
#include "logger.h"

extern std::unique_ptr<logger_t> logger;

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
  std::string& server_ip,
  std::string& tcp_port,
  std::string& udp_port
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
  server_ip(server_ip),
  tcp_port(tcp_port),
  udp_port(udp_port) {}

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

  tcpfd = socket(AF_INET, SOCK_STREAM, 0);
  if (tcpfd < 0) {
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, "Failed to create socket");
    return -errno;
  }

  struct sockaddr_in server_addr;
  server_addr.sin_family = AF_INET;

  int tcp_port_num = std::stoi(tcp_port);
  if (tcp_port_num < 1 || tcp_port_num > 65535) {
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, "Invalid tcp_port number");
    return -EINVAL;
  }

  server_addr.sin_port = htons(tcp_port_num);

  if (inet_pton(AF_INET, server_ip.c_str(), &server_addr.sin_addr) <= 0) {
    if (errno == 0) {
      logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, "Invalid IP address format");
      return -EINVAL;
    } else {
      logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, "System error during IP address conversion");
      return -errno;
    }
  }

  while (connect(tcpfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
    if (errno == EINTR) continue;
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, "Failed to connect to server");
    return -errno;
  }

  logger->log(logger_t::level_t::DEBUG, __FILE__, __LINE__, "Connected to server");
  return 0;
}

int connection::stream_pkt(const uint8_t* data, size_t size) {
  /**
   * Streams encoded video frames via tcp to the server
   *
   * This function is passed as a callback to the encoder
   * which calls it whenever it has a frame to output. This
   * is done because the encoder does not give an output
   * upon every input, but instead buffers frames for some
   * time before any are ready. Rather than having the encoder
   * communicate externally about the outcome of encoding the
   * video frame, we simply pass this function and main isn't
   * concerned with whether it's called or not.
   *
   * The tcp socket is checked before use, and if it's not
   * connected, we connect. This means the first connection to
   * the tcp socket does not occur until sometime after recording
   * begins, once the first encoded frame is ready for transmission.
   */
  size_t total_bytes_written = 0;
  while (total_bytes_written < size) {
    if (tcpfd < 0) {
      int ret = conn_tcp();
      if (ret < 0) return ret;
    }

    ssize_t result = write(
      tcpfd,
      data + total_bytes_written,
      size - total_bytes_written
    );

    if (result < 0) {
      if (errno == EINTR) continue;
      logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, "Error transmitting frame");
      return -1;
    }

    total_bytes_written += result;
  }

  logger->log(logger_t::level_t::INFO, __FILE__, __LINE__, "Transmitted frame");
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

  udpfd = socket(AF_INET, SOCK_DGRAM, 0);
  if (udpfd < 0) {
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, "Failed to create UDP socket");
    return -errno;
  }

  struct sockaddr_in udp_addr;
  memset(&udp_addr, 0, sizeof(udp_addr));
  udp_addr.sin_family = AF_INET;
  udp_addr.sin_addr.s_addr = htonl(INADDR_ANY);

  int udp_port_num = std::stoi(udp_port);
  if (udp_port_num < 1 || udp_port_num > 65535) {
    logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, "Invalid udp_port number");
    return -EINVAL;
  }
  udp_addr.sin_port = htons(udp_port_num);

  while (bind(udpfd, (struct sockaddr*)&udp_addr, sizeof(udp_addr)) < 0) {
      if (errno == EINTR) continue;
      logger->log(logger_t::level_t::ERROR, __FILE__, __LINE__, "Failed to bind UDP socket");
      return -errno;
  }

  logger->log(logger_t::level_t::INFO, __FILE__, __LINE__, "UDP socket bound successfully");
  return 0;
}

size_t connection::recv_msg(char* msg_buf) {
  return recvfrom(
    udpfd,
    msg_buf,
    sizeof(msg_buf),
    0,
    NULL,
    NULL
  );
}
