#include <cstdint>
#include <cstring>
#include <errno.h>
#include <opencv2/opencv.hpp>
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

  int32_t cam_count = 3; // pass temp cam count until we can parse the conf file

  struct stream_ctx stream_ctx;
  ret = start_streams(stream_ctx, cam_count);
  if (ret < 0) {
    cleanup_streams(stream_ctx);
    cleanup_logging();
    return ret;
  }

  uint32_t received_framesets = 0;
  while (received_framesets < 1000) {
    struct ts_frame_buf** frameset = static_cast<ts_frame_buf**>(
      spsc_dequeue(stream_ctx.filled_frameset_q)
    );
    if (frameset == nullptr) {
      usleep(1000); // 1ms
      continue;
    }

    for (int i = 0; i < cam_count; i++) {
      cv::Mat nv12_frame(
        FRAME_HEIGHT * 3/2,
        FRAME_WIDTH,
        CV_8UC1,
        frameset[i]->frame_buf
      );

      cv::Mat bgr_frame;
      cv::cvtColor(
        nv12_frame,
        bgr_frame,
        cv::COLOR_YUV2BGR_NV12
      );

      char name[8];
      snprintf(
        name,
        sizeof(name),
        "Cam %d",
        i
      );
      cv::imshow(name, bgr_frame);
      cv::waitKey(1);
    }

    spsc_enqueue(stream_ctx.empty_frameset_q, frameset);

    received_framesets++;
  }

  cleanup_streams(stream_ctx);
  cleanup_logging();
  return 0;
}
