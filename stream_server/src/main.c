#include <errno.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "parse_conf.h"
#include "logging.h"

int main() {
  int ret = 0;
  char logstr[128];

  char* log_path = "/var/log/mocap-toolkit/server.log";
  if (setup_logging(log_path)) {
    printf("Error opening log file: %s\n", strerror(errno));
    return -errno;
  }

  char* cams_conf_path = "/etc/mocap-toolkit/cams.yaml";
  int8_t cam_count = count_cameras(cams_conf_path);
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

  return ret;
}
