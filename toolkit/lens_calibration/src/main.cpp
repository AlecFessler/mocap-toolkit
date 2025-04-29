#include <csignal>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <errno.h>
#include <opencv2/opencv.hpp>
#include <sched.h>
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
constexpr uint32_t BOARD_HEIGHT = 6; constexpr float SQUARE_SIZE = 25.0; // mm

constexpr uint32_t CORES_PER_CCD = 8;

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
    log_write(ERROR, logstr);
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
    log_write(ERROR, logstr);
    cleanup_logging();
    return ret;
  }

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(cam_count % CORES_PER_CCD, &cpuset);
  pid_t pid = getpid();
  sched_setaffinity(
    pid,
    sizeof(cpu_set_t),
    &cpuset
  );

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
      log_write(ERROR, logstr);
      cleanup_logging();
      return -EINVAL;
    }
  } else {
    log_write(ERROR, "One camera ID required for lens calibration");
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
    cam_count,
    target_id
  );
  if (ret < 0) {
    cleanup_streams(stream_ctx);
    cleanup_logging();
    return ret;
  }

  uint32_t frame_pitch = ((stream_conf.frame_width + 511) / 512) * 512;
  uint64_t frame_size = stream_conf.frame_width * stream_conf.frame_height * 3 / 2;
  uint8_t* host_frame = static_cast<uint8_t*>(malloc(frame_size));
  if (host_frame == nullptr) {
    cleanup_streams(stream_ctx);
    cleanup_logging();
    return -ENOMEM;
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
    void* dev_ptr = nullptr;
    cudaIpcMemHandle_t* ipc_handle = static_cast<cudaIpcMemHandle_t*>(
      spsc_dequeue(stream_ctx.ipc_handles_cq)
    );
    if (ipc_handle == nullptr) {
      usleep(100); // 0.1ms
      continue;
    }

    cudaError_t cudaErr = cudaIpcOpenMemHandle(&dev_ptr, *ipc_handle, cudaIpcMemLazyEnablePeerAccess);
    if (cudaErr != cudaSuccess) {
      snprintf(
        logstr,
        sizeof(logstr),
        "cudaIpcOpenMemHandle failed: %s",
        cudaGetErrorString(cudaErr)
      );
      log_write(ERROR, logstr);
    }
    cudaErr = cudaMemcpy2D(
      host_frame,
      stream_conf.frame_width,
      dev_ptr,
      frame_pitch,
      stream_conf.frame_width,
      stream_conf.frame_height * 3 / 2,
      cudaMemcpyDeviceToHost
    );
    if (cudaErr != cudaSuccess) {
      snprintf(
        logstr,
        sizeof(logstr),
        "cudaMemcpy failed: %s",
        cudaGetErrorString(cudaErr)
      );
      log_write(ERROR, logstr);
    }

    cv::Mat nv12_frame(
      stream_conf.frame_height * 3/2,
      stream_conf.frame_width,
      CV_8UC1,
      host_frame
    );

    cv::Mat unprocessed_bgr;
    cv::cvtColor(
      nv12_frame,
      unprocessed_bgr,
      cv::COLOR_YUV2BGR_NV12
    );
    cv::Mat bgr_frame = wide_to_3_4_ar(unprocessed_bgr);

    if (cooldown > 0) {
      cudaErr = cudaIpcCloseMemHandle(dev_ptr);
      if (cudaErr != cudaSuccess) {
        snprintf(
          logstr,
          sizeof(logstr),
          "cudaIpcCloseMemHandle failed: %s",
          cudaGetErrorString(cudaErr)
        );
        log_write(ERROR, logstr);
      }
      for (uint32_t i = 0; i < cam_count + 1; i++) {
        stream_ctx.counters[i].fetch_sub(1, std::memory_order_relaxed);
      }

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

    cudaErr = cudaIpcCloseMemHandle(dev_ptr);
    if (cudaErr != cudaSuccess) {
      snprintf(
        logstr,
        sizeof(logstr),
        "cudaIpcCloseMemHandle failed: %s",
        cudaGetErrorString(cudaErr)
      );
      log_write(ERROR, logstr);
    }
    for (uint32_t i = 0; i < cam_count + 1; i++) {
      stream_ctx.counters[i].fetch_sub(1, std::memory_order_relaxed);
    }

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

  if (calibration_complete) {
    std::string filename = std::string(cam_confs[0].name) + "_calibration.yaml";
    calibrator.save_params(filename);
  }

  cleanup_streams(stream_ctx);
  cleanup_logging();
  free(host_frame);
  return 0;
}
