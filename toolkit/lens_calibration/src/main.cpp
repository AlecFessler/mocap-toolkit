#include <cstdint>
#include <cstring>
#include <errno.h>
#include <spsc_queue.hpp>
#include <iostream>
#include <unistd.h>

#include "logging.h"
#include "stream_ctl.h"

constexpr const char* LOG_PATH = "/var/log/mocap-toolkit/lens_calibration.log";

int main() {
  int32_t ret = 0;
  char logstr[128];

  ret = setup_logging(LOG_PATH);
  if (ret) {
    std::cout << "Error opening log file: " << strerror(errno) << "\n";
    return -errno;
  }

  struct stream_ctx stream_ctx;
  ret = start_streams(stream_ctx, 3); // pass temp cam count until we can parse the conf file
  if (ret < 0) {
    cleanup_streams(stream_ctx);
    cleanup_logging();
    return ret;
  }

  uint32_t received_framesets = 0;
  while (received_framesets < 100) {
    struct ts_frame_buf** frameset = static_cast<ts_frame_buf**>(
      spsc_dequeue(stream_ctx.filled_frameset_q)
    );
    if (frameset == nullptr) {
      usleep(1000); // 1ms
      continue;
    }

    snprintf(
      logstr,
      sizeof(logstr),
      "Received frameset with timestamp %lu",
      frameset[0]->timestamp
    );
    LOG(DEBUG, logstr);

    spsc_enqueue(stream_ctx.empty_frameset_q, frameset);

    received_framesets++;
  }

  cleanup_streams(stream_ctx);
  cleanup_logging();
  return 0;
}
