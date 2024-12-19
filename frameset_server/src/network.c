#include <arpa/inet.h>
#include <errno.h>
#include <net/if.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

#include "parse_conf.h"
#include "logging.h"
#include "network.h"

static bool is_eth_conn(int sockfd) {
  struct ifreq ifr;
  strncpy(ifr.ifr_name, "eno1", IFNAMSIZ);
  bool eth_conn = (ioctl(sockfd, SIOCGIFFLAGS, &ifr) >= 0) &&
                 !(ifr.ifr_flags & IFF_RUNNING);

  if (eth_conn) {
    log(INFO, "Not connected to ethernet network, falling back to wifi");
    return false;
  }

  log(INFO, "Connected to ethernet network");
  return true;
}

int broadcast_msg(cam_conf* confs, int confs_size, const char* msg, size_t msg_size) {
  int ret = 0;
  char logstr[128];

  int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
  if (sockfd < 0) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Error creating udp socket: %s",
      strerror(errno)
    );
    log(ERROR, logstr);
    return -errno;
  }

  bool eth_conn = is_eth_conn(sockfd);

  for (int i = 0; i < confs_size; i++) {
    struct sockaddr_in rcvr_addr;
    memset(&rcvr_addr, 0, sizeof(rcvr_addr));
    rcvr_addr.sin_family = AF_INET;
    rcvr_addr.sin_port = htons(confs[i].udp_port);
    rcvr_addr.sin_addr = eth_conn ?
                         confs[i].eth_ip :
                         confs[i].wifi_ip;
    ret = sendto(
      sockfd,
      msg,
      msg_size,
      0,
      (struct sockaddr*)&rcvr_addr,
      sizeof(rcvr_addr)
    );

    if (ret < 0) {
      snprintf(
        logstr,
        sizeof(logstr),
        "Error broadcasting msg: %s",
        strerror(errno)
      );
      log(ERROR, logstr);
      ret = -errno;
      break;

    } else {
      snprintf(
        logstr,
        sizeof(logstr),
        "Broadcast msg to %s",
        confs[i].name
      );
      log(INFO, logstr);
    }
  }

  log(INFO, "Broadcast msg to all cameras successfully");
  close(sockfd);
  return ret;
}

int setup_stream(cam_conf* conf) {
  int ret = 0;
  char logstr[128];

  int sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd < 0) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Error creating tcp socket: %s",
      strerror(errno)
    );
    log(ERROR, logstr);
    return -errno;
  }

  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(conf->tcp_port);
  addr.sin_addr.s_addr = INADDR_ANY;

  ret = bind(sockfd, (struct sockaddr*)&addr, sizeof(addr));
  if (ret < 0) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Error binding tcp socket: %s",
      strerror(errno)
    );
    log(ERROR, logstr);
    ret = -errno;
    goto err_cleanup;
  }

  int backlog = 1; // allow one client
  ret = listen(sockfd, backlog);
  if (ret < 0) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Error listening on tcp socket: %s",
      strerror(errno)
    );
    log(ERROR, logstr);
    ret = -errno;
    goto err_cleanup;
  }

  return sockfd;

  err_cleanup:
  if (sockfd >= 0) {
    close(sockfd);
  }
  return ret;
}

int accept_conn(int sockfd) {
  char logstr[128];

  struct sockaddr_in rcvr_addr;
  socklen_t addr_len = sizeof(rcvr_addr);

  int clientfd = accept(sockfd, (struct sockaddr*)&rcvr_addr, &addr_len);
  if (clientfd < 0) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Error accepting connection: %s",
      strerror(errno)
    );
    log(ERROR, logstr);
    return -errno;
  }

  log(INFO, "Accepted connection");
  return clientfd;
}

ssize_t recv_from_stream(int clientfd, char* buf, size_t size) {
  char logstr[128];

  ssize_t bytes = recv(clientfd, buf, size, MSG_WAITALL);
  if (bytes == 0) {
    log(INFO, "Client has disconnected");
    return 0;
  } else if (bytes < 0) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Error receiving packet from stream: %s",
      strerror(errno)
    );
    log(ERROR, logstr);
    return -errno;
  }

  return bytes;
}
