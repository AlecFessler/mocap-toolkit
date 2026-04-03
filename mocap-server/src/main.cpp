#include <atomic>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <pthread.h>
#include <sched.h>
#include <spsc_queue.hpp>
#include <string>
#include <time.h>
#include <unistd.h>
#include <vector>
#include <fcntl.h>

extern "C" {
#include <libavutil/hwcontext.h>
}

#include "img_processing.hpp"
#include "lens_calibration.hpp"
#include "logging.h"
#include "network.hpp"
#include "parse_conf.h"
#include "pose_predictor.hpp"
#include "stereo_calibration.hpp"
#include "stream_mgr.hpp"
#include "triangulator.hpp"

constexpr const char* LOG_PATH = "/var/log/mocap-toolkit/toolkit.log";
constexpr const char* CAM_CONF_PATH = "/etc/mocap-toolkit/cams.yaml";
constexpr const char* CALIBRATION_PARAMS_PATH = "/etc/mocap-toolkit/";
constexpr const char* MODEL_PATH = "/var/lib/mocap-toolkit/sapiens.plan";

constexpr uint32_t CORES_PER_CCD = 8;
constexpr uint32_t TIMESTAMP_DELAY = 1; // seconds
constexpr uint32_t EMPTY_QS_WAIT = 10000; // 0.01 ms
constexpr uint32_t DEV_PTRS_PER_THREAD = 16;
constexpr uint32_t DEV_PTR_ACQUIRE_RETRY_LIMIT = 10;

constexpr const char* KEYPOINTS_OUTPUT_PATH = "/tmp/mocap_keypoints_alec.bin";
// binary format: each frame is NUM_KEYPOINTS * 3 floats (x,y,z), NaN for invalid
// frame size = 308 * 3 * 4 = 3696 bytes
constexpr uint32_t FRAME_BYTES = NUM_KEYPOINTS * 3 * sizeof(float);

constexpr uint32_t BOARD_WIDTH = 9;
constexpr uint32_t BOARD_HEIGHT = 6;
constexpr float SQUARE_SIZE = 25.0f; // mm

enum mode {
  MODE_LENS_CALIBRATION,
  MODE_STEREO_CALIBRATION,
  MODE_MOCAP
};

static volatile sig_atomic_t running = 1;

static void shutdown_handler(int signum) {
  (void)signum;
  running = 0;
}

// frameset passed from main thread to inference thread
constexpr uint32_t INFERENCE_Q_SLOTS = 4;

struct frameset {
  void* dev_ptrs[MAX_CAMERAS];
  std::atomic<uint32_t>* surface_counters[MAX_CAMERAS]; // decrement these when done
  int cam_count;
};

struct inference_ctx {
  consumer_q* frameset_cq;
  PosePredictor* predictor;
  Triangulator* triangulator;
  stream_conf* strm_conf;
  cam_conf* cam_confs;
  int cam_count;
  int kp_fd;
  uint8_t* host_frames; // for display
  volatile sig_atomic_t* running;
};

static void* inference_thread_fn(void* ptr) {
  inference_ctx* ctx = static_cast<inference_ctx*>(ptr);
  char logstr[128];

  uint32_t frame_pitch = ((ctx->strm_conf->frame_width + 511) / 512) * 512;
  uint64_t frame_size = ctx->strm_conf->frame_width * ctx->strm_conf->frame_height * 3 / 2;

  std::vector<std::vector<std::pair<float, float>>> keypoints_2d(
    ctx->cam_count, std::vector<std::pair<float, float>>(NUM_KEYPOINTS)
  );
  std::vector<std::vector<float>> confidence_scores(
    ctx->cam_count, std::vector<float>(NUM_KEYPOINTS)
  );
  std::vector<cv::Point3f> keypoints_3d(NUM_KEYPOINTS);

  struct timespec sleep_ts = { .tv_sec = 0, .tv_nsec = 100000 }; // 0.1ms

  // create windows once, side by side (resizable, scaled to fit)
  for (int i = 0; i < ctx->cam_count; i++) {
    cv::namedWindow(ctx->cam_confs[i].name, cv::WINDOW_NORMAL);
    cv::resizeWindow(ctx->cam_confs[i].name, 384, 512);
    cv::moveWindow(ctx->cam_confs[i].name, i * 390, 0);
  }

  while (*ctx->running) {
    frameset* fs = static_cast<frameset*>(spsc_dequeue(ctx->frameset_cq));
    if (fs == nullptr) {
      nanosleep(&sleep_ts, nullptr);
      continue;
    }

    // GPU->host copy for display
    for (int i = 0; i < ctx->cam_count; i++) {
      cudaMemcpy2D(
        ctx->host_frames + i * frame_size,
        ctx->strm_conf->frame_width,
        fs->dev_ptrs[i],
        frame_pitch,
        ctx->strm_conf->frame_width,
        ctx->strm_conf->frame_height * 3 / 2,
        cudaMemcpyDeviceToHost
      );
    }

    // TensorRT inference directly on decoder surfaces
    ctx->predictor->predict_gpu(
      fs->dev_ptrs,
      ctx->strm_conf->frame_width,
      ctx->strm_conf->frame_height,
      frame_pitch,
      keypoints_2d,
      confidence_scores
    );

    // release decoder surfaces now
    for (int i = 0; i < ctx->cam_count; i++)
      fs->surface_counters[i]->fetch_sub(1, std::memory_order_relaxed);

    // 3D triangulation
    ctx->triangulator->triangulate(
      keypoints_2d,
      confidence_scores,
      keypoints_3d
    );

    // visualize
    cv::Mat bgr_frames[MAX_CAMERAS];
    for (int i = 0; i < ctx->cam_count; i++) {
      cv::Mat nv12_frame(
        ctx->strm_conf->frame_height * 3 / 2,
        ctx->strm_conf->frame_width,
        CV_8UC1,
        ctx->host_frames + i * frame_size
      );
      cv::Mat unprocessed_bgr;
      cv::cvtColor(nv12_frame, unprocessed_bgr, cv::COLOR_YUV2BGR_NV12);
      bgr_frames[i] = wide_to_3_4_ar(unprocessed_bgr);

      for (int k = 0; k < NUM_KEYPOINTS; k++) {
        if (confidence_scores[i][k] < 0.5f)
          continue;
        int x = static_cast<int>(keypoints_2d[i][k].first);
        int y = static_cast<int>(keypoints_2d[i][k].second);
        cv::circle(bgr_frames[i], cv::Point(x, y), 3, cv::Scalar(0, 0, 255), -1);
      }
      cv::imshow(ctx->cam_confs[i].name, bgr_frames[i]);
    }
    cv::waitKey(1);

    // write 3D keypoints to file
    if (ctx->kp_fd >= 0)
      write(ctx->kp_fd, keypoints_3d.data(), FRAME_BYTES);

    // debug: count 2D detections per camera and confidence stats
    for (int i = 0; i < ctx->cam_count; i++) {
      int det_count = 0;
      float max_conf = -1e9f;
      float min_conf = 1e9f;
      for (int k = 0; k < NUM_KEYPOINTS; k++) {
        if (confidence_scores[i][k] >= 0.5f)
          det_count++;
        if (confidence_scores[i][k] > max_conf)
          max_conf = confidence_scores[i][k];
        if (confidence_scores[i][k] < min_conf)
          min_conf = confidence_scores[i][k];
      }
      snprintf(logstr, sizeof(logstr), "Cam %d: %d/308 detections (conf range %.3f to %.3f)",
        i, det_count, min_conf, max_conf);
      log_write(INFO, logstr);
    }

    int valid_count = 0;
    for (int k = 0; k < NUM_KEYPOINTS; k++) {
      if (!std::isnan(keypoints_3d[k].x))
        valid_count++;
    }
    snprintf(logstr, sizeof(logstr), "Triangulated %d/%d keypoints", valid_count, NUM_KEYPOINTS);
    log_write(INFO, logstr);
  }

  return nullptr;
}

static void print_usage(const char* name) {
  printf("Usage: %s <mode> [options]\n", name);
  printf("Modes:\n");
  printf("  lens_calibration <cam_id>   Intrinsic calibration for a single camera\n");
  printf("  stereo_calibration          Stereo calibration for all camera pairs\n");
  printf("  mocap                       3D pose reconstruction\n");
}

int main(int argc, char* argv[]) {
  int ret = 0;
  char logstr[128];

  if (argc < 2) {
    print_usage(argv[0]);
    return -EINVAL;
  }

  enum mode mode;
  if (strcmp(argv[1], "lens_calibration") == 0) {
    if (argc < 3) {
      printf("lens_calibration requires a camera ID argument\n");
      return -EINVAL;
    }
    mode = MODE_LENS_CALIBRATION;
  } else if (strcmp(argv[1], "stereo_calibration") == 0) {
    mode = MODE_STEREO_CALIBRATION;
  } else if (strcmp(argv[1], "mocap") == 0) {
    mode = MODE_MOCAP;
  } else {
    print_usage(argv[0]);
    return -EINVAL;
  }

  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sa.sa_handler = shutdown_handler;
  sa.sa_flags = 0;
  sigemptyset(&sa.sa_mask);
  sigaction(SIGTERM, &sa, nullptr);
  sigaction(SIGINT, &sa, nullptr);

  ret = setup_logging(LOG_PATH);
  if (ret) {
    printf("Error opening log file: %s\n", strerror(errno));
    return -errno;
  }

  int cam_count = count_cameras(CAM_CONF_PATH);
  if (cam_count <= 0) {
    snprintf(logstr, sizeof(logstr), "Error getting camera count: %s", strerror(cam_count));
    log_write(ERROR, logstr);
    cleanup_logging();
    return cam_count;
  }
  if (cam_count > MAX_CAMERAS) {
    log_write(ERROR, "Camera count exceeds MAX_CAMERAS");
    cleanup_logging();
    return -EINVAL;
  }

  struct stream_conf stream_conf;
  struct cam_conf cam_confs[MAX_CAMERAS];
  ret = parse_conf(&stream_conf, cam_confs, cam_count);
  if (ret) {
    snprintf(logstr, sizeof(logstr), "Error parsing camera confs: %s", strerror(ret));
    log_write(ERROR, logstr);
    cleanup_logging();
    return ret;
  }

  // filter to single camera for lens calibration
  int target_cam_id = -1;
  if (mode == MODE_LENS_CALIBRATION) {
    target_cam_id = atoi(argv[2]);
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
      snprintf(logstr, sizeof(logstr), "Camera ID %d not found in config", target_cam_id);
      log_write(ERROR, logstr);
      cleanup_logging();
      return -EINVAL;
    }
  }

  // pin main thread
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(cam_count % CORES_PER_CCD, &cpuset);
  sched_setaffinity(getpid(), sizeof(cpu_set_t), &cpuset);

  // create single shared CUDA hw device context
  AVBufferRef* hw_device_ctx = nullptr;
  ret = av_hwdevice_ctx_create(
    &hw_device_ctx,
    AV_HWDEVICE_TYPE_CUDA,
    nullptr,
    nullptr,
    0
  );
  if (ret < 0) {
    snprintf(logstr, sizeof(logstr), "Failed to create CUDA device context: %d", ret);
    log_write(ERROR, logstr);
    cleanup_logging();
    return ret;
  }

  // allocate per-camera resources (all statically sized)
  std::atomic<uint32_t> dev_ptrs_used[MAX_CAMERAS];
  ts_dev_ptr ts_dev_ptrs[MAX_CAMERAS * DEV_PTRS_PER_THREAD];
  producer_q dev_ptr_pqs[MAX_CAMERAS];
  consumer_q dev_ptr_cqs[MAX_CAMERAS];
  void* dev_ptr_q_bufs[MAX_CAMERAS * DEV_PTRS_PER_THREAD];

  for (int i = 0; i < cam_count; i++) {
    dev_ptrs_used[i].store(0, std::memory_order_relaxed);
    spsc_queue_init(
      &dev_ptr_pqs[i],
      &dev_ptr_cqs[i],
      &dev_ptr_q_bufs[i * DEV_PTRS_PER_THREAD],
      DEV_PTRS_PER_THREAD
    );
  }

  // broadcast initial timestamp to start cameras
  // the Pi takes ~1s after receiving the timestamp to connect via TCP,
  // so we broadcast first, then spawn worker threads which will accept
  log_write(INFO, "Broadcasting timestamp to cameras...");
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  uint64_t timestamp = (ts.tv_sec + TIMESTAMP_DELAY) * 1000000000ULL + ts.tv_nsec;
  broadcast_msg(cam_confs, cam_count, reinterpret_cast<const char*>(&timestamp), sizeof(timestamp));
  log_write(INFO, "Broadcast sent, spawning worker threads");

  // spawn worker threads
  thread_ctx ctxs[MAX_CAMERAS];
  pthread_t threads[MAX_CAMERAS];
  int thread_count = 0;

  for (int i = 0; i < cam_count; i++) {
    ctxs[i].conf = &cam_confs[i];
    ctxs[i].strm_conf = &stream_conf;
    ctxs[i].dev_ptr_queue = &dev_ptr_pqs[i];
    ctxs[i].dev_ptrs_used = &dev_ptrs_used[i];
    ctxs[i].dev_ptrs_total = DEV_PTRS_PER_THREAD;
    ctxs[i].dev_ptrs = &ts_dev_ptrs[i * DEV_PTRS_PER_THREAD];
    ctxs[i].core = i % CORES_PER_CCD;
    ctxs[i].main_running = &running;
    ctxs[i].hw_device_ctx = hw_device_ctx;

    ret = pthread_create(&threads[i], nullptr, stream_mgr_fn, static_cast<void*>(&ctxs[i]));
    if (ret) {
      log_write(ERROR, "Error spawning thread");
      running = 0;
      break;
    }
    thread_count++;
  }

  // mode-specific initialization
  LensCalibration* lens_calibrator = nullptr;
  StereoCalibration* stereo_calibrator = nullptr;
  PosePredictor* predictor = nullptr;
  Triangulator* triangulator = nullptr;

  calibration_params calib_params[MAX_CAMERAS];
  uint32_t frame_pitch = ((stream_conf.frame_width + 511) / 512) * 512;
  uint64_t frame_size = stream_conf.frame_width * stream_conf.frame_height * 3 / 2;

  // host frame buffer for calibration modes (GPU->host copy needed for OpenCV)
  uint8_t* host_frames = nullptr;

  if (mode == MODE_LENS_CALIBRATION) {
    lens_calibrator = new LensCalibration(
      PROCESSED_WIDTH, PROCESSED_HEIGHT,
      BOARD_WIDTH, BOARD_HEIGHT, SQUARE_SIZE
    );
    host_frames = static_cast<uint8_t*>(malloc(frame_size));
    if (!host_frames) {
      log_write(ERROR, "Failed to allocate host frame buffer");
      running = 0;
    }
  } else if (mode == MODE_STEREO_CALIBRATION) {
    for (int i = 0; i < cam_count; i++) {
      std::string filename =
        std::string(CALIBRATION_PARAMS_PATH) +
        std::string(cam_confs[i].name) +
        "_calibration.yaml";
      if (!load_calibration_params(filename, calib_params[i])) {
        snprintf(logstr, sizeof(logstr), "Failed to load %s", filename.c_str());
        log_write(ERROR, logstr);
        running = 0;
        break;
      }
    }
    if (running) {
      stereo_calibrator = new StereoCalibration(
        calib_params, cam_count,
        PROCESSED_WIDTH, PROCESSED_HEIGHT,
        BOARD_WIDTH, BOARD_HEIGHT, SQUARE_SIZE
      );
      host_frames = static_cast<uint8_t*>(malloc(frame_size * cam_count));
      if (!host_frames) {
        log_write(ERROR, "Failed to allocate host frame buffer");
        running = 0;
      }
    }
  } else if (mode == MODE_MOCAP) {
    for (int i = 0; i < cam_count; i++) {
      std::string filename =
        std::string(CALIBRATION_PARAMS_PATH) +
        std::string(cam_confs[i].name) +
        "_calibration.yaml";
      if (!load_calibration_params(filename, calib_params[i])) {
        snprintf(logstr, sizeof(logstr), "Failed to load %s", filename.c_str());
        log_write(ERROR, logstr);
        running = 0;
        break;
      }
    }
    if (running) {
      predictor = new PosePredictor(std::string(MODEL_PATH), cam_count);
      triangulator = new Triangulator(
        calib_params, cam_count,
        std::string(CALIBRATION_PARAMS_PATH), cam_confs
      );
    }
  }

  // inference thread setup for mocap mode
  int kp_fd = -1;
  uint8_t* mocap_host_frames = nullptr;
  producer_q frameset_pq;
  consumer_q frameset_cq;
  void* frameset_q_buf[INFERENCE_Q_SLOTS];
  frameset frameset_pool[INFERENCE_Q_SLOTS];
  uint32_t frameset_pool_idx = 0;
  pthread_t inference_thread;
  bool inference_thread_started = false;
  inference_ctx inf_ctx;

  if (mode == MODE_MOCAP && running) {
    kp_fd = open(KEYPOINTS_OUTPUT_PATH, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (kp_fd < 0) {
      snprintf(logstr, sizeof(logstr), "Failed to open keypoints output: %s", strerror(errno));
      log_write(ERROR, logstr);
    }

    mocap_host_frames = static_cast<uint8_t*>(malloc(frame_size * cam_count));
    if (!mocap_host_frames) {
      log_write(ERROR, "Failed to allocate mocap display buffer");
      running = 0;
    }

    spsc_queue_init(&frameset_pq, &frameset_cq, frameset_q_buf, INFERENCE_Q_SLOTS);

    inf_ctx.frameset_cq = &frameset_cq;
    inf_ctx.predictor = predictor;
    inf_ctx.triangulator = triangulator;
    inf_ctx.strm_conf = &stream_conf;
    inf_ctx.cam_confs = cam_confs;
    inf_ctx.cam_count = cam_count;
    inf_ctx.kp_fd = kp_fd;
    inf_ctx.host_frames = mocap_host_frames;
    inf_ctx.running = &running;

    ret = pthread_create(&inference_thread, nullptr, inference_thread_fn, &inf_ctx);
    if (ret) {
      log_write(ERROR, "Failed to spawn inference thread");
      running = 0;
    } else {
      inference_thread_started = true;
    }
  }

  // frame bundling arrays
  ts_dev_ptr* dev_ptrs_set[MAX_CAMERAS];
  memset(dev_ptrs_set, 0, sizeof(dev_ptrs_set));

  // sleep timer for empty queue polling
  struct timespec sleep_ts = {
    .tv_sec = 0,
    .tv_nsec = EMPTY_QS_WAIT
  };

  // calibration cooldown state
  const uint32_t detection_cooldown = stream_conf.fps / 3;
  const uint32_t failure_cooldown = stream_conf.fps / 5;
  uint32_t cooldown = 0;
  uint32_t cooldown_counter = 0;
  bool calibration_complete = false;

  // create display windows for calibration modes
  if (mode == MODE_LENS_CALIBRATION) {
    cv::namedWindow("stream", cv::WINDOW_NORMAL);
    cv::resizeWindow("stream", 384, 512);
  } else if (mode == MODE_STEREO_CALIBRATION) {
    for (int i = 0; i < cam_count; i++) {
      cv::namedWindow(cam_confs[i].name, cv::WINDOW_NORMAL);
      cv::resizeWindow(cam_confs[i].name, 384, 512);
      cv::moveWindow(cam_confs[i].name, i * 390, 0);
    }
  }

  // frame bundling + dispatch loop
  while (running && !calibration_complete) {
    // dequeue a full set of timestamped frame buffers from each worker thread
    bool full_set = true;
    for (int i = 0; i < cam_count; i++) {
      if (dev_ptrs_set[i] != nullptr)
        continue;

      dev_ptrs_set[i] = static_cast<ts_dev_ptr*>(spsc_dequeue(&dev_ptr_cqs[i]));
      if (dev_ptrs_set[i] == nullptr)
        full_set = false;
    }

    if (!full_set) {
      nanosleep(&sleep_ts, nullptr);
      continue;
    }

    // find the max timestamp
    uint64_t max_timestamp = 0;
    for (int i = 0; i < cam_count; i++) {
      if (dev_ptrs_set[i]->timestamp > max_timestamp)
        max_timestamp = dev_ptrs_set[i]->timestamp;
    }

    // discard frames that don't match
    bool all_equal = true;
    for (int i = 0; i < cam_count; i++) {
      if (dev_ptrs_set[i]->timestamp != max_timestamp) {
        all_equal = false;
        dev_ptrs_used[i].fetch_sub(1, std::memory_order_relaxed);
        dev_ptrs_set[i] = nullptr;
      }
    }

    if (!all_equal)
      continue;

    log_write(BENCHMARK, "Received full frameset");

    // collect device pointers for this frameset
    void* frame_dev_ptrs[MAX_CAMERAS];
    for (int i = 0; i < cam_count; i++)
      frame_dev_ptrs[i] = dev_ptrs_set[i]->dev_ptr;

    // dispatch to mode handler
    if (mode == MODE_LENS_CALIBRATION) {
      // GPU -> host copy
      cudaError_t cudaErr = cudaMemcpy2D(
        host_frames,
        stream_conf.frame_width,
        frame_dev_ptrs[0],
        frame_pitch,
        stream_conf.frame_width,
        stream_conf.frame_height * 3 / 2,
        cudaMemcpyDeviceToHost
      );
      if (cudaErr != cudaSuccess) {
        snprintf(logstr, sizeof(logstr), "cudaMemcpy failed: %s", cudaGetErrorString(cudaErr));
        log_write(ERROR, logstr);
      }

      // release surfaces
      dev_ptrs_used[0].fetch_sub(1, std::memory_order_relaxed);
      memset(dev_ptrs_set, 0, sizeof(dev_ptrs_set));

      cv::Mat nv12_frame(
        stream_conf.frame_height * 3 / 2,
        stream_conf.frame_width,
        CV_8UC1,
        host_frames
      );

      cv::Mat unprocessed_bgr;
      cv::cvtColor(nv12_frame, unprocessed_bgr, cv::COLOR_YUV2BGR_NV12);
      cv::Mat bgr_frame = wide_to_3_4_ar(unprocessed_bgr);

      if (cooldown > 0) {
        cv::imshow("stream", bgr_frame);
        cv::waitKey(1);
        if (++cooldown_counter >= cooldown) {
          cooldown_counter = 0;
          cooldown = 0;
        }
        continue;
      }

      cv::Mat unprocessed_gray;
      cv::cvtColor(nv12_frame, unprocessed_gray, cv::COLOR_YUV2GRAY_NV12);
      cv::Mat gray_frame = wide_to_3_4_ar(unprocessed_gray);

      bool found_corners = lens_calibrator->try_frame(gray_frame);
      if (!found_corners) {
        cv::imshow("stream", bgr_frame);
        cv::waitKey(1);
        cooldown = failure_cooldown;
        continue;
      }

      cooldown = detection_cooldown;
      lens_calibrator->display_corners(bgr_frame);

      double err = lens_calibrator->calibrate();
      (void)err;
      calibration_complete = lens_calibrator->check_status();

    } else if (mode == MODE_STEREO_CALIBRATION) {
      // GPU -> host copy for all cameras
      for (int i = 0; i < cam_count; i++) {
        cudaError_t cudaErr = cudaMemcpy2D(
          host_frames + i * frame_size,
          stream_conf.frame_width,
          frame_dev_ptrs[i],
          frame_pitch,
          stream_conf.frame_width,
          stream_conf.frame_height * 3 / 2,
          cudaMemcpyDeviceToHost
        );
        if (cudaErr != cudaSuccess) {
          snprintf(logstr, sizeof(logstr), "cudaMemcpy failed: %s", cudaGetErrorString(cudaErr));
          log_write(ERROR, logstr);
        }
      }

      // release surfaces
      for (int i = 0; i < cam_count; i++)
        dev_ptrs_used[i].fetch_sub(1, std::memory_order_relaxed);
      memset(dev_ptrs_set, 0, sizeof(dev_ptrs_set));

      cv::Mat gray_frames[MAX_CAMERAS];
      cv::Mat bgr_frames[MAX_CAMERAS];
      for (int i = 0; i < cam_count; i++) {
        cv::Mat nv12_frame(
          stream_conf.frame_height * 3 / 2,
          stream_conf.frame_width,
          CV_8UC1,
          host_frames + i * frame_size
        );

        cv::Mat unprocessed_gray;
        cv::cvtColor(nv12_frame, unprocessed_gray, cv::COLOR_YUV2GRAY_NV12);
        gray_frames[i] = wide_to_3_4_ar(unprocessed_gray);

        cv::Mat unprocessed_bgr;
        cv::cvtColor(nv12_frame, unprocessed_bgr, cv::COLOR_YUV2BGR_NV12);
        bgr_frames[i] = wide_to_3_4_ar(unprocessed_bgr);
      }

      for (int i = 0; i < cam_count; i++)
        cv::imshow(cam_confs[i].name, bgr_frames[i]);
      cv::waitKey(16);
      usleep(1000);

      if (cooldown > 0) {
        if (++cooldown_counter >= cooldown)
          cooldown = 0;
        continue;
      }
      cooldown = detection_cooldown;
      cooldown_counter = 0;

      stereo_calibrator->try_frames(gray_frames);
      calibration_complete = stereo_calibrator->check_status();

    } else if (mode == MODE_MOCAP) {
      // package frameset and hand off to inference thread
      frameset* fs = &frameset_pool[frameset_pool_idx];
      frameset_pool_idx = (frameset_pool_idx + 1) % INFERENCE_Q_SLOTS;

      for (int i = 0; i < cam_count; i++) {
        fs->dev_ptrs[i] = frame_dev_ptrs[i];
        fs->surface_counters[i] = &dev_ptrs_used[i];
      }
      fs->cam_count = cam_count;

      // if inference queue is full, drop this frameset (inference can't keep up)
      if (spsc_enqueue(&frameset_pq, static_cast<void*>(fs)) != 0) {
        for (int i = 0; i < cam_count; i++)
          dev_ptrs_used[i].fetch_sub(1, std::memory_order_relaxed);
      }
      memset(dev_ptrs_set, 0, sizeof(dev_ptrs_set));
    }
  }

  // save calibration results
  if (mode == MODE_LENS_CALIBRATION && calibration_complete) {
    std::string filename = std::string(cam_confs[0].name) + "_calibration.yaml";
    lens_calibrator->save_params(filename);
  } else if (mode == MODE_STEREO_CALIBRATION && calibration_complete) {
    stereo_calibrator->calibrate();
    stereo_calibrator->save_params(cam_confs);
  }

  // stop the camera devices
  const char* stop_msg = "STOP";
  broadcast_msg(cam_confs, cam_count, stop_msg, strlen(stop_msg));

  // shutdown worker threads
  running = 0;
  for (int i = 0; i < thread_count; i++)
    pthread_kill(threads[i], SIGUSR2);
  for (int i = 0; i < thread_count; i++)
    pthread_join(threads[i], nullptr);

  // wait for inference thread
  if (inference_thread_started)
    pthread_join(inference_thread, nullptr);

  // cleanup
  delete lens_calibrator;
  delete stereo_calibrator;
  delete predictor;
  delete triangulator;
  if (host_frames)
    free(host_frames);
  if (mocap_host_frames)
    free(mocap_host_frames);
  if (kp_fd >= 0)
    close(kp_fd);
  if (hw_device_ctx)
    av_buffer_unref(&hw_device_ctx);
  cleanup_logging();

  return 0;
}
