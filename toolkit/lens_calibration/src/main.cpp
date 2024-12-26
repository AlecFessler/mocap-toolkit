#include <cstdint>
#include <cstring>
#include <errno.h>
#include <opencv2/opencv.hpp>
#include <spsc_queue.hpp>
#include <string>
#include <iostream>
#include <unistd.h>

#include "logging.h"
#include "parse_conf.h"
#include "stream_ctl.h"

constexpr const char* LOG_PATH = "/var/log/mocap-toolkit/lens_calibration.log";
constexpr const char* CAM_CONF_PATH = "/etc/mocap-toolkit/cams.yaml";

int main(int argc, char* argv[]) {
  int32_t ret = 0;
  char logstr[128];

  ret = setup_logging(LOG_PATH);
  if (ret) {
    std::cout << "Error opening log file: " << strerror(errno) << "\n";
    return -errno;
  }

  int cam_count = count_cameras(CAM_CONF_PATH);
  if (cam_count <= 0) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Error getting camera count: %s",
      strerror(cam_count)
    );
    LOG(ERROR, logstr);
    cleanup_logging();
    return cam_count;
  }

  struct stream_conf stream_conf;
  struct cam_conf cam_confs[cam_count];
  ret = parse_conf(&stream_conf, cam_confs, cam_count);
  if (ret) {
    snprintf(
      logstr,
      sizeof(logstr),
      "Error parsing camera confs %s",
      strerror(ret)
    );
    LOG(ERROR, logstr);
    cleanup_logging();
    return ret;
  }

  int target_cam_id = -1;
  if (argc == 2) {
    target_cam_id = std::stoi(argv[1]);

    bool found = false;
    for (int i = 0; i < cam_count; i++) {
      if (cam_confs[i].id != target_cam_id)
        continue;

      cam_confs[0] = cam_confs[i];
      cam_count = 1;
      found = true;
      break;
    }

    if (!found) {
      snprintf(
        logstr,
        sizeof(logstr),
        "Camera ID %d not found in config",
        target_cam_id
      );
      LOG(ERROR, logstr);
      cleanup_logging();
      return -EINVAL;
    }
  } else {
    LOG(ERROR, "One camera ID required for lens calibration");
    cleanup_logging();
    return -EINVAL;
  }

  struct stream_ctx stream_ctx;
  char* target_id = target_cam_id >= 0 ? argv[1] : nullptr;
  ret = start_streams(stream_ctx, cam_count, target_id);
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
        stream_conf.frame_height * 3/2,
        stream_conf.frame_width,
        CV_8UC1,
        frameset[i]->frame_buf
      );

      cv::Mat bgr_frame;
      cv::cvtColor(
        nv12_frame,
        bgr_frame,
        cv::COLOR_YUV2BGR_NV12
      );

      char name[9];
      snprintf(
        name,
        sizeof(name),
        "%s",
        cam_confs[i].name
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
