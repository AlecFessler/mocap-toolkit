#include <csignal>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <errno.h>
#include <opencv2/opencv.hpp>
#include <spsc_queue.hpp>
#include <string>
#include <iostream>
#include <vector>
#include <unistd.h>

#include "img_processing.hpp"
#include "lens_calibration.hpp"
#include "logging.h"
#include "parse_conf.h"
#include "stereo_calibration.hpp"
#include "pose_predictor.hpp"
#include "stream_ctl.h"

constexpr const char* LOG_PATH = "/var/log/mocap-toolkit/mocap.log";
constexpr const char* CAM_CONF_PATH = "/etc/mocap-toolkit/cams.yaml";
constexpr const char* CALIBRATION_PARAMS_PATH = "/etc/mocap-toolkit/";
constexpr const char* MODEL_PATH = "/var/lib/mocap-toolkit/sapiens_0.3b_goliath_best_goliath_AP_573_torchscript.pt2";

constexpr uint32_t CORES_PER_CCD = 8;

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
    log_write(ERROR, logstr);
    cleanup_logging();
    return -EINVAL;
  }

  PosePredictor predictor{std::string(MODEL_PATH)};
  std::vector<cv::Mat> bgr_frames;
  bgr_frames.resize(cam_count);
  std::vector<std::pair<std::vector<float>, std::vector<float>>> keypoints;
  std::vector<std::vector<float>> confidence_scores;

  struct stream_ctx stream_ctx;
  ret = start_streams(
    stream_ctx,
    cam_count,
    nullptr
  );
  if (ret < 0) {
    cleanup_streams(stream_ctx);
    cleanup_logging();
    return ret;
  }

  uint32_t frame_pitch = ((stream_conf.frame_width + 511) / 512) * 512;
  uint64_t frame_size = stream_conf.frame_width * stream_conf.frame_height * 3 / 2;
  uint8_t* host_frames = static_cast<uint8_t*>(malloc(frame_size * cam_count));
  if (host_frames == nullptr) {
    cleanup_streams(stream_ctx);
    cleanup_logging();
    return -ENOMEM;
  }

  while (!stop_flag) {
    void* dev_ptrs[cam_count];
    cudaIpcMemHandle_t* ipc_handles = static_cast<cudaIpcMemHandle_t*>(
      spsc_dequeue(stream_ctx.ipc_handles_cq)
    );
    if (ipc_handles == nullptr) {
      usleep(100); // 0.1ms
      continue;
    }

    for (int i = 0; i < cam_count; i++) {
      cudaError_t cudaErr = cudaIpcOpenMemHandle(&dev_ptrs[i], ipc_handles[i], cudaIpcMemLazyEnablePeerAccess);
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
        host_frames + i * frame_size,
        stream_conf.frame_width,
        dev_ptrs[i],
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
    }

    for (int i = 0; i < cam_count; i++) {
      cv::Mat nv12_frame(
        stream_conf.frame_height * 3/2,
        stream_conf.frame_width,
        CV_8UC1,
        host_frames + i * frame_size
      );

      cv::Mat unprocessed_bgr;
      cv::cvtColor(
        nv12_frame,
        unprocessed_bgr,
        cv::COLOR_YUV2BGR_NV12
      );
      bgr_frames[i] = wide_to_3_4_ar(unprocessed_bgr);
    }

    for (int i = 0; i < cam_count; i++) {
      cudaError_t cudaErr = cudaIpcCloseMemHandle(dev_ptrs[i]);
      if (cudaErr != cudaSuccess) {
        snprintf(
          logstr,
          sizeof(logstr),
          "cudaIpcCloseMemHandle failed: %s",
          cudaGetErrorString(cudaErr)
        );
        log_write(ERROR, logstr);
      }
    }
    for (int i = 0; i < cam_count + 1; i++) {
      stream_ctx.counters[i].fetch_sub(1, std::memory_order_relaxed);
    }

    predictor.predict(bgr_frames, keypoints, confidence_scores);

    for (int i = 0; i < cam_count; i++) {
      for (int j = 0; j < NUM_KEYPOINTS; j++) {
        if (confidence_scores[i][j] < 0.5)
          continue;

        int x = static_cast<int>(keypoints[i].first[j] * bgr_frames[i].cols);
        int y = static_cast<int>(keypoints[i].second[j] * bgr_frames[i].rows);

        cv::circle(bgr_frames[i], cv::Point(x, y), 3, cv::Scalar(0, 0, 225), -1);
      }
      cv::imshow(cam_confs[i].name, bgr_frames[i]);
    }
    cv::waitKey(1);
  }

  cleanup_streams(stream_ctx);
  cleanup_logging();
  return 0;
}
