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
#include "udp_net.h"

static bool is_eth_conn(int sockfd) {
  struct ifreq ifr;
  strncpy(ifr.ifr_name, "eno1", IFNAMSIZ);
  if ((ioctl(sockfd, SIOCGIFFLAGS, &ifr) >= 0) &&
      (!(ifr.ifr_flags & IFF_RUNNING))
  ) {
    log(INFO, "Not connected to ethernet network, falling back to wifi");
    return false;
  }
  log(INFO, "Connected to ethernet network");
  return true;
}

int broadcast_msg(cam_conf* confs, int confs_size, const char* msg, size_t msg_size) {
  int ret = 0;
  char logstr[128];

  int sockfd = -1;
  if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
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

    if (sendto(
      sockfd,
      msg,
      msg_size,
      0,
      (struct sockaddr*)&rcvr_addr,
      sizeof(rcvr_addr)) < 0
    ) {
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
