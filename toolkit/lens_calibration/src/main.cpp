#include <csignal>
#include <cstdint>
#include <cstring>
#include <errno.h>
#include <opencv2/opencv.hpp>
#include <spsc_queue.hpp>
#include <string>
#include <iostream>
#include <unistd.h>

#include "img_processing.hpp"
#include "logging.h"
#include "lens_calibration.hpp"
#include "parse_conf.h"
#include "stream_ctl.h"

constexpr const char* LOG_PATH = "/var/log/mocap-toolkit/lens_calibration.log";
constexpr const char* CAM_CONF_PATH = "/etc/mocap-toolkit/cams.yaml";

constexpr uint32_t BOARD_WIDTH = 9;
constexpr uint32_t BOARD_HEIGHT = 6;
constexpr float SQUARE_SIZE = 25.0; // mm

volatile sig_atomic_t stop_flag = 0;

void stop_handler(int signum) {
  (void)signum;
  stop_flag = 1;
}

int main(int argc, char* argv[]) {
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

  LensCalibration calibrator(
    PROCESSED_WIDTH,
    PROCESSED_HEIGHT,
    BOARD_WIDTH,
    BOARD_HEIGHT,
    SQUARE_SIZE
  );

  struct stream_ctx stream_ctx;
  char* target_id = target_cam_id >= 0 ? argv[1] : nullptr;
  ret = start_streams(
    stream_ctx,
    stream_conf.frame_width,
    stream_conf.frame_height,
    cam_count,
    target_id
  );
  if (ret < 0) {
    cleanup_streams(stream_ctx);
    cleanup_logging();
    return ret;
  }

  /*
   * In either a success or a failure case, we want to wait
   * some cooldown before retrying a detection. This is because
   * on a success case, we want to allow some time to pass to
   * reposition the board before. For the failure case, this
   * is because a detectChessboardPatterns on the failure case
   * is quite costly, and will slow down the live stream, so
   * we wait a few frames to prevent from putting backpressure
   * on the stream
   */

  const uint32_t detection_cooldown = stream_conf.fps / 3;
  const uint32_t failure_cooldown = stream_conf.fps / 5;
  uint32_t cooldown = 0;
  uint32_t cooldown_counter = 0;

  bool calibration_complete = false;
  while (!stop_flag && !calibration_complete) {
    struct ts_frame_buf** frameset = static_cast<ts_frame_buf**>(
      spsc_dequeue(stream_ctx.filled_frameset_q)
    );
    if (frameset == nullptr) {
      usleep(100); // 0.1ms
      continue;
    }

    cv::Mat nv12_frame(
      stream_conf.frame_height * 3/2,
      stream_conf.frame_width,
      CV_8UC1,
      frameset[0]->frame_buf
    );

    cv::Mat unprocessed_bgr;
    cv::cvtColor(
      nv12_frame,
      unprocessed_bgr,
      cv::COLOR_YUV2BGR_NV12
    );
    cv::Mat bgr_frame = wide_to_3_4_ar(unprocessed_bgr);

    if (cooldown > 0) {
      spsc_enqueue(stream_ctx.empty_frameset_q, frameset);

      cv::imshow("stream", bgr_frame);
      cv::waitKey(1);

      if (++cooldown_counter >= cooldown) {
        cooldown_counter = 0;
        cooldown = 0;
      }

      continue;
    }

    cv::Mat unprocessed_gray;
    cv::cvtColor(
      nv12_frame,
      unprocessed_gray,
      cv::COLOR_YUV2GRAY_NV12
    );
    cv::Mat gray_frame = wide_to_3_4_ar(unprocessed_gray);

    spsc_enqueue(stream_ctx.empty_frameset_q, frameset);

    bool found_corners = calibrator.try_frame(gray_frame);
    if (!found_corners) {
      cv::imshow("stream", bgr_frame);
      cv::waitKey(1);
      cooldown = failure_cooldown;
      continue;
    }

    cooldown = detection_cooldown;
    calibrator.display_corners(bgr_frame);

    double err = calibrator.calibrate();
    calibration_complete = calibrator.check_status();
  }

  std::string filename = std::string(cam_confs[0].name) + "_calibration.yaml";
  calibrator.save_params(filename);

  cleanup_streams(stream_ctx);
  cleanup_logging();
  return 0;
}
