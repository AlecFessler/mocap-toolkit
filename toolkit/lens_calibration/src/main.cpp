#include <errno.h>
#include <iostream>
#include <opencv2/core.hpp>

#include "logging.h"
#include "stream_controller.h"

#define LOG_PATH "/var/log/mocap-toolkit/lens_calibration.log"

int main() {
  int ret = 0;
  char logstr[128];

  ret = setup_logging(LOG_PATH);
  if (ret) {
    std::cout << "Error opening log file: " << strerror(errno) << "\n";
    return -errno;
  }

  StreamController stream_ctlr = StreamController(
    1280,
    720,
    3
  );

  cv::Mat frames[3];
  uint64_t timestamp;
  uint32_t counter = 0;
  while(counter++ < 10) {
    stream_ctlr.recv_frameset(frames, &timestamp);
    snprintf(
      logstr,
      sizeof(logstr),
      "Received frameset with timestamp %lu",
      timestamp
    );
    LOG(DEBUG, logstr);
  }

  cleanup_logging();
  return 0;
}
