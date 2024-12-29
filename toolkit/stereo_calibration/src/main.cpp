#include <csignal>
#include <cstdint>
#include <cstring>
#include <errno.h>
#include <opencv2/opencv.hpp>
#include <spsc_queue.hpp>
#include <string>
#include <iostream>
#include <vector>
#include <unistd.h>

#include "lens_calibration.hpp"
#include "logging.h"
#include "parse_conf.h"
#include "stereo_calibration.hpp"
#include "stream_ctl.h"

constexpr const char* LOG_PATH = "/var/log/mocap-toolkit/stereo_calibration.log";
constexpr const char* CAM_CONF_PATH = "/etc/mocap-toolkit/cams.yaml";
constexpr const char* CALIBRATION_PARAMS_PATH = "/etc/mocap-toolkit/";

constexpr uint32_t BOARD_WIDTH = 9;
constexpr uint32_t BOARD_HEIGHT = 6;
constexpr float SQUARE_SIZE = 25.0; // mm

volatile sig_atomic_t stop_flag = 0;

void stop_handler(int signum) {
  (void)signum;
  stop_flag = 1;
}

int main() {
  int32_t ret = 0;
  char logstr[128];

  struct sigaction sa;
  sa.sa_handler = stop_handler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;
  sigaction(SIGINT, &sa, nullptr);

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

  struct calibration_params calib_params[cam_count];
  for (int i = 0; i < cam_count; i++) {
    std::string filename =
      std::string(CALIBRATION_PARAMS_PATH) +
      std::string(cam_confs[i].name) +
      "_calibration.yaml";

    bool loaded = load_calibration_params(
      filename,
      calib_params[i]
    );

    if (loaded) continue;

    snprintf(
      logstr,
      sizeof(logstr),
      "Failed to load %s",
      filename.c_str()
    );
    LOG(ERROR, logstr);
    cleanup_logging();
    return -EINVAL;
  }

  StereoCalibration calibrator{
    calib_params,
    cam_count,
    stream_conf.frame_width,
    stream_conf.frame_height,
    BOARD_WIDTH,
    BOARD_HEIGHT,
    SQUARE_SIZE
  };

  struct stream_ctx stream_ctx;
  ret = start_streams(
    stream_ctx,
    stream_conf.frame_width,
    stream_conf.frame_height,
    cam_count,
    nullptr
  );
  if (ret < 0) {
    cleanup_streams(stream_ctx);
    cleanup_logging();
    return ret;
  }

  const uint32_t cooldown_limit = stream_conf.fps / 3;
  uint32_t cooldown = 0;
  uint32_t cooldown_counter = 0;

  bool done_calibrating = false;
  cv::Mat gray_frames[cam_count];
  cv::Mat bgr_frames[cam_count];
  while (!stop_flag && !done_calibrating) {
    struct ts_frame_buf** frameset = static_cast<ts_frame_buf**>(
      spsc_dequeue(stream_ctx.filled_frameset_q)
    );
    if (frameset == nullptr) {
      usleep(100); // 0.1ms
      continue;
    }

    for (int i = 0; i < cam_count; i++) {
      cv::Mat nv12_frame(
        stream_conf.frame_height * 3/2,
        stream_conf.frame_width,
        CV_8UC1,
        frameset[i]->frame_buf
      );

      cv::cvtColor(
        nv12_frame,
        gray_frames[i],
        cv::COLOR_YUV2GRAY_NV12
      );

      cv::cvtColor(
        nv12_frame,
        bgr_frames[i],
        cv::COLOR_YUV2BGR_NV12
      );
    }

    spsc_enqueue(stream_ctx.empty_frameset_q, frameset);

    for (int i = 0; i < cam_count; i++) {
      cv::imshow(cam_confs[i].name, bgr_frames[i]);
      cv::waitKey(1);
    }

    if (cooldown > 0) {
      if (++cooldown_counter >= cooldown)
        cooldown = 0;
      continue;
    }
    cooldown = cooldown_limit;
    cooldown_counter = 0;

    calibrator.try_frames(gray_frames);
    done_calibrating = calibrator.check_status();
  }

  calibrator.calibrate();
  calibrator.save_params(cam_confs);

  cleanup_streams(stream_ctx);
  cleanup_logging();
  return 0;
}
