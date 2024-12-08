#include <errno.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#include "parse_conf.h"
#include "logging.h"
#include "udp_net.h"

#define TIMESTAMP_DELAY 3 // seconds



int main() {
  int ret = 0;
  char logstr[128];

  char* log_path = "/var/log/mocap-toolkit/server.log";
  if (setup_logging(log_path)) {
    printf("Error opening log file: %s\n", strerror(errno));
    return -errno;
  }

  char* cams_conf_path = "/etc/mocap-toolkit/cams.yaml";
  int cam_count = count_cameras(cams_conf_path);
  if (cam_count <= 0) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Error getting camera count: %s",
      strerror(cam_count)
    );
    log(ERROR, logstr);
    cleanup_logging();
    return cam_count;
  }

  struct cam_conf confs[cam_count];

  ret = parse_conf(confs, cam_count);
  if (ret) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Error parsing camera confs %s",
      strerror(ret)
    );
    log(ERROR, logstr);
    cleanup_logging();
    return ret;
  }

  // broadcast timestamp
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  uint64_t timestamp = (ts.tv_sec + TIMESTAMP_DELAY) * 1000000000ULL + ts.tv_nsec;
  const char* ts_msg = (char*)&timestamp;
  broadcast_msg(confs, cam_count, ts_msg, sizeof(timestamp))

  // start stream mananger threads

  // broadcast stop
  const char* stop_msg = "STOP";
  broadcast_msg(confs, cam_count, stop_msg, strlen(stop_msg));

  cleanup_logging();
  return ret;
}
