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

#include "video/img_processing.hpp"
#include "calibration/lens_calibration.hpp"
#include "core/logging.h"
#include "net/network.hpp"
#include "core/parse_conf.h"
#include "model/pose_predictor.hpp"
#include "calibration/stereo_calibration.hpp"
#include "net/stream_mgr.hpp"
#include "triangulation/triangulator.hpp"
#include "render/renderer.hpp"
#include "cuda/gpu_triangulate.hpp"
#include "cuda/gpu_kalman_3d.hpp"
#include "core/pipeline_config.h"
#include "ik/cometh_pipeline.hpp"

constexpr const char* LOG_PATH = "/var/log/mocap-toolkit/toolkit.log";
constexpr const char* CAM_CONF_PATH = "/etc/mocap-toolkit/cams.yaml";
constexpr const char* CALIBRATION_PARAMS_PATH = "/etc/mocap-toolkit/";
constexpr const char* MODEL_PATH = "/home/alec/mocap-toolkit/models/rtmw-x.plan";

constexpr uint32_t CORES_PER_CCD = 8;
constexpr uint32_t TIMESTAMP_DELAY = 1; // seconds
constexpr uint32_t EMPTY_QS_WAIT = 10000; // 0.01 ms
constexpr uint32_t DEV_PTRS_PER_THREAD = 16;
constexpr uint32_t DEV_PTR_ACQUIRE_RETRY_LIMIT = 10;


constexpr uint32_t BOARD_WIDTH = 9;
constexpr uint32_t BOARD_HEIGHT = 6;
constexpr float SQUARE_SIZE = 25.0f; // mm

enum mode {
  MODE_LENS_CALIBRATION,
  MODE_STEREO_CALIBRATION,
  MODE_MOCAP
};

// COCO WholeBody 133 skeleton edges as flat pairs for the renderer
// Body: 0=nose,1=l_eye,2=r_eye,3=l_ear,4=r_ear,5=l_shoulder,6=r_shoulder,
//       7=l_elbow,8=r_elbow,9=l_wrist,10=r_wrist,11=l_hip,12=r_hip,
//       13=l_knee,14=r_knee,15=l_ankle,16=r_ankle
// Feet: 17=l_big_toe,18=l_small_toe,19=l_heel,20=r_big_toe,21=r_small_toe,22=r_heel
// Face: 23-90 (68 keypoints)
// L Hand: 91-111 (21 keypoints, 91=wrist)
// R Hand: 112-132 (21 keypoints, 112=wrist)
static const int RENDER_SKELETON_EDGES[] = {
  // body
  15,13, 13,11, 16,14, 14,12, 11,12, 5,11, 6,12, 5,6, 5,7, 6,8, 7,9, 8,10,
  // head
  1,2, 0,1, 0,2, 1,3, 2,4, 3,5, 4,6,
  // feet
  15,17, 15,18, 15,19, 16,20, 16,21, 16,22,
  // left hand (wrist=91, thumb=92-95, index=96-99, middle=100-103, ring=104-107, pinky=108-111)
  9,91, 91,92, 92,93, 93,94, 94,95,
  91,96, 96,97, 97,98, 98,99,
  91,100, 100,101, 101,102, 102,103,
  91,104, 104,105, 105,106, 106,107,
  91,108, 108,109, 109,110, 110,111,
  // right hand (wrist=112, thumb=113-116, index=117-120, middle=121-124, ring=125-128, pinky=129-132)
  10,112, 112,113, 113,114, 114,115, 115,116,
  112,117, 117,118, 118,119, 119,120,
  112,121, 121,122, 122,123, 123,124,
  112,125, 125,126, 126,127, 127,128,
  112,129, 129,130, 130,131, 131,132,
};
static constexpr int NUM_RENDER_EDGES = sizeof(RENDER_SKELETON_EDGES) / (2 * sizeof(int));

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
  uint8_t* host_frames; // for display (calibration modes)
  float* latest_keypoints_3d;           // shared buffer for renderer [NUM_KEYPOINTS * 3] (IK output)
  float* latest_keypoints_3d_raw;      // shared buffer for renderer [NUM_KEYPOINTS * 3] (raw DLT+Kalman)
  frame_metrics* latest_metrics;        // shared buffer for renderer
  std::atomic<bool>* keypoints_ready;   // signal main thread new data available
  volatile sig_atomic_t* running;
  // GPU triangulation state (device)
  double* dev_proj_matrices;
  double* dev_cam_matrices;
  double* dev_dist_coeffs;
  float* dev_keypoints_3d;
  float* dev_reproj_error;
  float* dev_pairwise_spread;
  triangulate_params* dev_tri_params;
  // 3D Kalman filter state (device)
  float* dev_kf3d_state;
  float* dev_kf3d_covariance;
  float* dev_kf3d_out;
  kalman_params* dev_kf3d_params;
  pipeline_config* pipe_config;
  ComethPipeline* cometh;
  // shared 2D keypoint buffer for overlay on camera feeds [cam_count * NUM_KEYPOINTS * 2]
  float* latest_keypoints_2d;
  float* latest_confidence;
  std::atomic<bool>* kp2d_ready;
};

static void* inference_thread_fn(void* ptr) {
  inference_ctx* ctx = static_cast<inference_ctx*>(ptr);
  char logstr[128];

  uint32_t frame_pitch = ((ctx->strm_conf->frame_width + 511) / 512) * 512;

  int total_kf = ctx->cam_count * NUM_KEYPOINTS;

  std::vector<std::vector<std::pair<float, float>>> keypoints_2d(
    ctx->cam_count, std::vector<std::pair<float, float>>(NUM_KEYPOINTS)
  );
  std::vector<std::vector<float>> confidence_scores(
    ctx->cam_count, std::vector<float>(NUM_KEYPOINTS)
  );
  std::vector<cv::Point3f> keypoints_3d(NUM_KEYPOINTS);

  // host buffers for D→H copy of filtered results
  std::vector<float> host_kp_buf(total_kf * 2);
  std::vector<float> host_conf_buf(total_kf);

  struct timespec sleep_ts = { .tv_sec = 0, .tv_nsec = 100000 }; // 0.1ms

  // run-level accumulators
  double run_total_reproj = 0.0;
  int64_t run_reproj_samples = 0;
  int64_t run_total_valid = 0;
  int64_t run_frames = 0;
  uint32_t last_stats_gen = ctx->pipe_config->stats_generation.load(std::memory_order_relaxed);

  cudaStream_t stream = ctx->predictor->stream();

  while (*ctx->running) {
    frameset* fs = static_cast<frameset*>(spsc_dequeue(ctx->frameset_cq));
    if (fs == nullptr) {
      nanosleep(&sleep_ts, nullptr);
      continue;
    }

    // TensorRT inference + GPU SimCC postprocess (stays on device)
    float* dev_kp_raw;
    float* dev_conf_raw;
    ctx->predictor->predict_gpu_device(
      fs->dev_ptrs,
      ctx->strm_conf->frame_width,
      ctx->strm_conf->frame_height,
      frame_pitch,
      &dev_kp_raw,
      &dev_conf_raw
    );

    // release decoder surfaces now
    for (int i = 0; i < ctx->cam_count; i++)
      fs->surface_counters[i]->fetch_sub(1, std::memory_order_relaxed);

    cudaStreamSynchronize(stream);

    // N-view DLT triangulation (all cameras required, with undistortion)
    gpu_triangulate(
      dev_kp_raw,
      dev_conf_raw,
      ctx->dev_keypoints_3d,
      ctx->dev_reproj_error,
      ctx->dev_pairwise_spread,
      ctx->dev_proj_matrices,
      ctx->dev_cam_matrices,
      ctx->dev_dist_coeffs,
      ctx->dev_tri_params,
      stream
    );

    // 3D Kalman filter (temporal smoothing + gap filling)
    gpu_kalman_3d_update(
      ctx->dev_kf3d_state, ctx->dev_kf3d_covariance,
      ctx->dev_keypoints_3d, ctx->dev_kf3d_out,
      ctx->dev_kf3d_params,
      NUM_KEYPOINTS, stream
    );
    cudaMemcpyAsync(ctx->dev_keypoints_3d, ctx->dev_kf3d_out,
      NUM_KEYPOINTS * 3 * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    cudaStreamSynchronize(stream);

    // Copy results D→H
    float host_kp3d[NUM_KEYPOINTS * 3];
    float host_reproj[NUM_KEYPOINTS];
    float host_spread[NUM_KEYPOINTS];
    cudaMemcpy(host_kp3d, ctx->dev_keypoints_3d,
      NUM_KEYPOINTS * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_reproj, ctx->dev_reproj_error,
      NUM_KEYPOINTS * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_spread, ctx->dev_pairwise_spread,
      NUM_KEYPOINTS * sizeof(float), cudaMemcpyDeviceToHost);

    // Also copy confidence and 2D keypoints D→H
    cudaMemcpy(host_conf_buf.data(), dev_conf_raw,
      total_kf * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_kp_buf.data(), dev_kp_raw,
      total_kf * 2 * sizeof(float), cudaMemcpyDeviceToHost);

    // ========== PIPELINE INSTRUMENTATION ==========
    // Stage 1: 2D detection — per-camera confidence stats for body keypoints (5-16)
    {
      int body_detected[MAX_CAMERAS] = {};  // keypoints with conf >= 3.0 per camera
      float cam_max_conf[MAX_CAMERAS] = {};
      float cam_min_conf[MAX_CAMERAS];
      for (int c = 0; c < ctx->cam_count; c++) cam_min_conf[c] = 999.0f;

      for (int c = 0; c < ctx->cam_count; c++) {
        for (int k = 5; k <= 16; k++) {
          float conf = host_conf_buf[c * NUM_KEYPOINTS + k];
          if (conf >= 3.0f) body_detected[c]++;
          if (conf > cam_max_conf[c]) cam_max_conf[c] = conf;
          if (conf < cam_min_conf[c]) cam_min_conf[c] = conf;
        }
      }
      // How many body keypoints have ALL cameras above threshold?
      int all_cams_body = 0;
      for (int k = 5; k <= 16; k++) {
        bool all_pass = true;
        for (int c = 0; c < ctx->cam_count; c++) {
          if (host_conf_buf[c * NUM_KEYPOINTS + k] < 3.0f) {
            all_pass = false;
            break;
          }
        }
        if (all_pass) all_cams_body++;
      }

      snprintf(logstr, sizeof(logstr),
        "2D: cam_body=[%d,%d,%d] conf=[%.1f-%.1f, %.1f-%.1f, %.1f-%.1f] all_cams=%d/12",
        body_detected[0], body_detected[1], ctx->cam_count > 2 ? body_detected[2] : 0,
        cam_min_conf[0], cam_max_conf[0],
        cam_min_conf[1], cam_max_conf[1],
        ctx->cam_count > 2 ? cam_min_conf[2] : 0.0f,
        ctx->cam_count > 2 ? cam_max_conf[2] : 0.0f,
        all_cams_body);
      log_write(INFO, logstr);
    }

    // Stage 2: DLT triangulation — how many keypoints survived?
    int dlt_body_valid = 0;
    int dlt_total_valid = 0;
    for (int k = 0; k < NUM_KEYPOINTS; k++) {
      if (!std::isnan(host_kp3d[k * 3])) {
        dlt_total_valid++;
        if (k >= 5 && k <= 16) dlt_body_valid++;
      }
    }

    // Stage 3: Kalman — same buffer post-kalman, count again
    // (kalman fills NaN gaps via prediction, so count may increase)
    // Note: host_kp3d already has kalman output since we copied dev_kf3d_out -> dev_keypoints_3d -> host

    snprintf(logstr, sizeof(logstr), "DLT: body=%d/12 total=%d/133",
      dlt_body_valid, dlt_total_valid);
    log_write(INFO, logstr);

    // Build frame_metrics from GPU results
    frame_metrics metrics;
    for (int k = 0; k < NUM_KEYPOINTS; k++) {
      metrics.reproj_error[k] = host_reproj[k];
      metrics.pairwise_spread[k] = host_spread[k];
      metrics.num_views[k] = 0;
    }
    for (int k = 0; k < NUM_KEYPOINTS; k++) {
      int views = 0;
      for (int c = 0; c < ctx->cam_count; c++) {
        if (host_conf_buf[c * NUM_KEYPOINTS + k] >= 3.0f)
          views++;
      }
      metrics.num_views[k] = views;
    }
    for (int b = 0; b < NUM_BONES; b++) {
      int a = BONE_EDGES[b].a;
      int bi = BONE_EDGES[b].b;
      float ax = host_kp3d[a * 3], ay = host_kp3d[a * 3 + 1], az = host_kp3d[a * 3 + 2];
      float bx = host_kp3d[bi * 3], by = host_kp3d[bi * 3 + 1], bz = host_kp3d[bi * 3 + 2];
      if (!std::isnan(ax) && !std::isnan(bx)) {
        float dx = ax - bx, dy = ay - by, dz = az - bz;
        metrics.bone_lengths[b] = std::sqrt(dx * dx + dy * dy + dz * dz);
      } else {
        metrics.bone_lengths[b] = std::numeric_limits<float>::quiet_NaN();
      }
    }

    // Stage 4: COMETH IK
    float ik_output[NUM_KEYPOINTS * 3];
    ctx->cometh->process(host_kp3d, ik_output);

    // Stage 5: Measure IK displacement on body joints
    {
      float total_dist = 0.0f, max_dist = 0.0f;
      int matched = 0;
      for (int k = 5; k <= 16; k++) {
        if (std::isnan(host_kp3d[k*3]) || std::isnan(ik_output[k*3])) continue;
        float dx = host_kp3d[k*3]-ik_output[k*3];
        float dy = host_kp3d[k*3+1]-ik_output[k*3+1];
        float dz = host_kp3d[k*3+2]-ik_output[k*3+2];
        float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
        total_dist += dist;
        if (dist > max_dist) max_dist = dist;
        matched++;
      }
      snprintf(logstr, sizeof(logstr), "IK: body_in=%d body_out=%d disp=%.1f/%.1fmm",
        dlt_body_valid, matched,
        matched > 0 ? total_dist / matched : 0.0f, max_dist);
      log_write(INFO, logstr);
    }
    // ========== END INSTRUMENTATION ==========

    // copy to shared buffers for renderer
    if (ctx->latest_keypoints_3d) {
      // IK output: only pass major body joints (0=nose, 5-16=body), NaN everything else
      for (int k = 0; k < NUM_KEYPOINTS; k++) {
        if (k == 0 || (k >= 5 && k <= 16)) {
          ctx->latest_keypoints_3d[k*3+0] = ik_output[k*3+0];
          ctx->latest_keypoints_3d[k*3+1] = ik_output[k*3+1];
          ctx->latest_keypoints_3d[k*3+2] = ik_output[k*3+2];
        } else {
          ctx->latest_keypoints_3d[k*3+0] = std::numeric_limits<float>::quiet_NaN();
          ctx->latest_keypoints_3d[k*3+1] = std::numeric_limits<float>::quiet_NaN();
          ctx->latest_keypoints_3d[k*3+2] = std::numeric_limits<float>::quiet_NaN();
        }
      }
      // Raw 3D: same filtering
      for (int k = 0; k < NUM_KEYPOINTS; k++) {
        if (k == 0 || (k >= 5 && k <= 16)) {
          ctx->latest_keypoints_3d_raw[k*3+0] = host_kp3d[k*3+0];
          ctx->latest_keypoints_3d_raw[k*3+1] = host_kp3d[k*3+1];
          ctx->latest_keypoints_3d_raw[k*3+2] = host_kp3d[k*3+2];
        } else {
          ctx->latest_keypoints_3d_raw[k*3+0] = std::numeric_limits<float>::quiet_NaN();
          ctx->latest_keypoints_3d_raw[k*3+1] = std::numeric_limits<float>::quiet_NaN();
          ctx->latest_keypoints_3d_raw[k*3+2] = std::numeric_limits<float>::quiet_NaN();
        }
      }
      if (ctx->latest_metrics)
        memcpy(ctx->latest_metrics, &metrics, sizeof(frame_metrics));
      ctx->keypoints_ready->store(true, std::memory_order_release);
    }

    // copy 2D keypoints + confidence to shared buffer for camera feed overlay
    if (ctx->latest_keypoints_2d) {
      memcpy(ctx->latest_keypoints_2d, host_kp_buf.data(), total_kf * 2 * sizeof(float));
      memcpy(ctx->latest_confidence, host_conf_buf.data(), total_kf * sizeof(float));
      ctx->kp2d_ready->store(true, std::memory_order_release);
    }

    // running stats for renderer overlay
    {
      int valid_count = 0;
      float total_reproj = 0.0f;
      int reproj_count = 0;
      for (int k = 0; k < NUM_KEYPOINTS; k++) {
        if (!std::isnan(host_kp3d[k * 3])) valid_count++;
        if (!std::isnan(host_reproj[k])) { total_reproj += host_reproj[k]; reproj_count++; }
      }
      uint32_t cur_gen = ctx->pipe_config->stats_generation.load(std::memory_order_relaxed);
      if (cur_gen != last_stats_gen) {
        run_total_reproj = 0.0; run_reproj_samples = 0; run_total_valid = 0; run_frames = 0;
        last_stats_gen = cur_gen;
      }
      run_total_reproj += total_reproj;
      run_reproj_samples += reproj_count;
      run_total_valid += valid_count;
      run_frames++;
      float running_avg = run_reproj_samples > 0 ? (float)(run_total_reproj / run_reproj_samples) : 0.0f;
      ctx->pipe_config->avg_reproj_running.store(running_avg, std::memory_order_relaxed);
      ctx->pipe_config->avg_reproj_frames.store((int32_t)run_frames, std::memory_order_relaxed);
    }
  }

  // log run summary
  if (run_frames > 0) {
    double run_avg_reproj = run_reproj_samples > 0 ? run_total_reproj / run_reproj_samples : 0.0;
    double run_avg_valid = (double)run_total_valid / run_frames;
    snprintf(logstr, sizeof(logstr),
      "RUN SUMMARY: %ld frames | avg reproj: %.2fpx | avg valid: %.1f/%d",
      (long)run_frames, run_avg_reproj, run_avg_valid, NUM_KEYPOINTS);
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

  // setup TCP control and stream listeners per camera
  int ctrl_listen_fds[MAX_CAMERAS];
  int ctrl_fds[MAX_CAMERAS];
  int stream_listen_fds[MAX_CAMERAS];
  int stream_fds[MAX_CAMERAS];
  memset(stream_fds, -1, sizeof(stream_fds));

  log_write(INFO, "Setting up control and stream listeners...");
  for (int i = 0; i < cam_count; i++) {
    ctrl_listen_fds[i] = setup_ctrl_listener(cam_confs[i].tcp_port);
    if (ctrl_listen_fds[i] < 0) {
      log_write(ERROR, "Failed to setup control listener");
      running = 0;
      break;
    }
    stream_listen_fds[i] = setup_stream_listener(cam_confs[i].udp_port);
    if (stream_listen_fds[i] < 0) {
      log_write(ERROR, "Failed to setup stream listener");
      running = 0;
      break;
    }
  }

  // wait for camera TCP connections
  log_write(INFO, "Waiting for camera control connections...");
  for (int i = 0; i < cam_count && running; i++) {
    ctrl_fds[i] = accept_ctrl_conn(ctrl_listen_fds[i]);
    if (ctrl_fds[i] < 0) {
      snprintf(logstr, sizeof(logstr), "Failed to accept control connection for cam %d", i);
      log_write(ERROR, logstr);
      running = 0;
      break;
    }

    // receive IDENTIFY message
    ctrl_msg_header hdr;
    uint8_t cam_id;
    int msg = recv_ctrl_msg(ctrl_fds[i], &hdr, &cam_id, sizeof(cam_id));
    if (msg == CTRL_IDENTIFY) {
      snprintf(logstr, sizeof(logstr), "Camera %d connected (id=%d)", i, cam_id);
      log_write(INFO, logstr);
    }
  }

  // send START to all cameras
  if (running) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    uint64_t timestamp = (ts.tv_sec + TIMESTAMP_DELAY) * 1000000000ULL + ts.tv_nsec;

    log_write(INFO, "Sending START to all cameras...");
    for (int i = 0; i < cam_count; i++) {
      ret = send_ctrl_msg(ctrl_fds[i], CTRL_START, &timestamp, sizeof(timestamp));
      if (ret < 0) {
        snprintf(logstr, sizeof(logstr), "Failed to send START to cam %d", i);
        log_write(ERROR, logstr);
        running = 0;
        break;
      }

      // wait for ACK
      ctrl_msg_header ack_hdr;
      int msg = recv_ctrl_msg(ctrl_fds[i], &ack_hdr, nullptr, 0);
      if (msg != CTRL_ACK) {
        snprintf(logstr, sizeof(logstr), "No ACK from cam %d", i);
        log_write(WARNING, logstr);
      }
    }
    log_write(INFO, "All cameras started");

    // accept stream TCP connections (Pi connects after receiving START)
    log_write(INFO, "Waiting for stream connections...");
    for (int i = 0; i < cam_count && running; i++) {
      snprintf(logstr, sizeof(logstr), "Accepting stream connection for cam %d on port %d...",
        i, cam_confs[i].udp_port);
      log_write(INFO, logstr);
      stream_fds[i] = accept_stream_conn(stream_listen_fds[i]);
      if (stream_fds[i] < 0) {
        snprintf(logstr, sizeof(logstr), "Failed to accept stream connection for cam %d: %s",
          i, strerror(errno));
        log_write(ERROR, logstr);
        running = 0;
        break;
      }

      // receive IDENTIFY from stream connection
      ctrl_msg_header stream_hdr;
      uint8_t stream_cam_id;
      int stream_msg = recv_ctrl_msg(stream_fds[i], &stream_hdr, &stream_cam_id, sizeof(stream_cam_id));
      if (stream_msg == CTRL_IDENTIFY) {
        snprintf(logstr, sizeof(logstr), "Stream connection from cam %d (id=%d)", i, stream_cam_id);
        log_write(INFO, logstr);
      }
    }
    log_write(INFO, "All stream connections established");
  }

  // spawn worker threads with stream fds
  thread_ctx ctxs[MAX_CAMERAS];
  pthread_t threads[MAX_CAMERAS];
  int thread_count = 0;

  for (int i = 0; i < cam_count && running; i++) {
    ctxs[i].conf = &cam_confs[i];
    ctxs[i].strm_conf = &stream_conf;
    ctxs[i].dev_ptr_queue = &dev_ptr_pqs[i];
    ctxs[i].dev_ptrs_used = &dev_ptrs_used[i];
    ctxs[i].dev_ptrs_total = DEV_PTRS_PER_THREAD;
    ctxs[i].dev_ptrs = &ts_dev_ptrs[i * DEV_PTRS_PER_THREAD];
    ctxs[i].core = i % CORES_PER_CCD;
    ctxs[i].main_running = &running;
    ctxs[i].hw_device_ctx = hw_device_ctx;
    ctxs[i].ctrl_fd = ctrl_fds[i];
    ctxs[i].stream_fd = stream_fds[i];

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
        calib_params, cam_count, cam_confs,
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

  // shared buffers for renderer (written by inference thread, read by main thread)
  float shared_keypoints_3d[NUM_KEYPOINTS * 3];
  float shared_keypoints_3d_raw[NUM_KEYPOINTS * 3];
  frame_metrics shared_metrics{};
  std::atomic<bool> keypoints_ready{false};
  pipeline_config pipe_config;
  for (int i = 0; i < NUM_KEYPOINTS * 3; i++) {
    shared_keypoints_3d[i] = std::numeric_limits<float>::quiet_NaN();
    shared_keypoints_3d_raw[i] = std::numeric_limits<float>::quiet_NaN();
  }

  // shared buffers for 2D keypoint overlay on camera feeds
  float shared_keypoints_2d[MAX_CAMERAS * NUM_KEYPOINTS * 2];
  float shared_confidence[MAX_CAMERAS * NUM_KEYPOINTS];
  std::atomic<bool> kp2d_ready{false};
  memset(shared_keypoints_2d, 0, sizeof(shared_keypoints_2d));
  memset(shared_confidence, 0, sizeof(shared_confidence));

  // inference thread setup for mocap mode
  float* dev_kf3d_state = nullptr;
  float* dev_kf3d_covariance = nullptr;
  float* dev_kf3d_out = nullptr;
  kalman_params* dev_kf3d_params = nullptr;
  double* dev_proj_matrices = nullptr;
  double* dev_cam_matrices = nullptr;
  double* dev_dist_coeffs = nullptr;
  float* dev_keypoints_3d = nullptr;
  float* dev_reproj_error = nullptr;
  float* dev_pairwise_spread = nullptr;
  triangulate_params* dev_tri_params = nullptr;
  uint8_t* mocap_host_frames = nullptr;
  producer_q frameset_pq;
  consumer_q frameset_cq;
  void* frameset_q_buf[INFERENCE_Q_SLOTS];
  frameset frameset_pool[INFERENCE_Q_SLOTS];
  uint32_t frameset_pool_idx = 0;
  pthread_t inference_thread;
  bool inference_thread_started = false;
  inference_ctx inf_ctx{};  // zero-initialize so cometh/pointers are null in non-mocap modes

  if (mode == MODE_MOCAP && running) {
    mocap_host_frames = static_cast<uint8_t*>(malloc(frame_size * cam_count));
    if (!mocap_host_frames) {
      log_write(ERROR, "Failed to allocate mocap display buffer");
      running = 0;
    }

    spsc_queue_init(&frameset_pq, &frameset_cq, frameset_q_buf, INFERENCE_Q_SLOTS);

    cudaStream_t kf_stream = predictor->stream();

    // 3D Kalman filter buffers
    cudaMalloc(&dev_kf3d_state, NUM_KEYPOINTS * 6 * sizeof(float));
    cudaMalloc(&dev_kf3d_covariance, NUM_KEYPOINTS * 36 * sizeof(float));
    cudaMalloc(&dev_kf3d_out, NUM_KEYPOINTS * 3 * sizeof(float));
    cudaMalloc(&dev_kf3d_params, sizeof(kalman_params));

    gpu_kalman_3d_init(dev_kf3d_state, dev_kf3d_covariance, NUM_KEYPOINTS, kf_stream);

    // 3D Kalman uses different tuning (mm-scale positions, not pixels)
    kalman_params kf3d_params;
    kf3d_params.process_noise = 50.0f;
    kf3d_params.measurement_noise = 100.0f;
    kf3d_params.gate_threshold = 10000.0f;
    kf3d_params.confidence_decay = 0.85f;
    kf3d_params.min_confidence = 0.5f;
    kf3d_params.dt = 1.0f / 30.0f;
    kf3d_params.confidence_threshold = 3.0f;
    cudaMemcpy(dev_kf3d_params, &kf3d_params, sizeof(kalman_params), cudaMemcpyHostToDevice);
    cudaStreamSynchronize(kf_stream);

    // GPU triangulation buffers
    cudaMalloc(&dev_proj_matrices, cam_count * 12 * sizeof(double));
    cudaMalloc(&dev_cam_matrices, cam_count * 9 * sizeof(double));
    cudaMalloc(&dev_dist_coeffs, cam_count * 5 * sizeof(double));
    cudaMalloc(&dev_keypoints_3d, NUM_KEYPOINTS * 3 * sizeof(float));
    cudaMalloc(&dev_reproj_error, NUM_KEYPOINTS * sizeof(float));
    cudaMalloc(&dev_pairwise_spread, NUM_KEYPOINTS * sizeof(float));
    cudaMalloc(&dev_tri_params, sizeof(triangulate_params));

    // extract projection and camera matrices from Triangulator and upload
    {
      std::vector<double> host_proj(cam_count * 12);
      std::vector<double> host_cam(cam_count * 9);
      std::vector<double> host_dist(cam_count * 5, 0.0);
      for (int i = 0; i < cam_count; i++) {
        const cv::Mat& P = triangulator->proj_matrices()[i];
        const cv::Mat& K = triangulator->cam_matrices()[i];
        const cv::Mat& D = triangulator->dist_coeffs()[i];
        for (int r = 0; r < 3; r++)
          for (int c = 0; c < 4; c++)
            host_proj[i * 12 + r * 4 + c] = P.at<double>(r, c);
        for (int r = 0; r < 3; r++)
          for (int c = 0; c < 3; c++)
            host_cam[i * 9 + r * 3 + c] = K.at<double>(r, c);
        for (int j = 0; j < std::min(5, D.rows * D.cols); j++)
          host_dist[i * 5 + j] = D.at<double>(j);
      }
      gpu_triangulate_init(
        dev_proj_matrices, dev_cam_matrices, dev_dist_coeffs,
        host_proj.data(), host_cam.data(), host_dist.data(),
        cam_count, kf_stream
      );
    }

    // upload triangulation params
    triangulate_params tri_params;
    tri_params.confidence_threshold = 3.0f;
    tri_params.num_cameras = cam_count;
    tri_params.num_keypoints = NUM_KEYPOINTS;
    cudaMemcpy(dev_tri_params, &tri_params, sizeof(triangulate_params), cudaMemcpyHostToDevice);
    cudaStreamSynchronize(kf_stream);

    inf_ctx.frameset_cq = &frameset_cq;
    inf_ctx.predictor = predictor;
    inf_ctx.triangulator = triangulator;
    inf_ctx.strm_conf = &stream_conf;
    inf_ctx.cam_confs = cam_confs;
    inf_ctx.cam_count = cam_count;
    inf_ctx.host_frames = mocap_host_frames;
    inf_ctx.running = &running;
    inf_ctx.latest_keypoints_3d = shared_keypoints_3d;
    inf_ctx.latest_keypoints_3d_raw = shared_keypoints_3d_raw;
    inf_ctx.latest_metrics = &shared_metrics;
    inf_ctx.keypoints_ready = &keypoints_ready;
    inf_ctx.latest_keypoints_2d = shared_keypoints_2d;
    inf_ctx.latest_confidence = shared_confidence;
    inf_ctx.kp2d_ready = &kp2d_ready;
    inf_ctx.dev_kf3d_state = dev_kf3d_state;
    inf_ctx.dev_kf3d_covariance = dev_kf3d_covariance;
    inf_ctx.dev_kf3d_out = dev_kf3d_out;
    inf_ctx.dev_kf3d_params = dev_kf3d_params;
    inf_ctx.dev_proj_matrices = dev_proj_matrices;
    inf_ctx.dev_cam_matrices = dev_cam_matrices;
    inf_ctx.dev_dist_coeffs = dev_dist_coeffs;
    inf_ctx.dev_keypoints_3d = dev_keypoints_3d;
    inf_ctx.dev_reproj_error = dev_reproj_error;
    inf_ctx.dev_pairwise_spread = dev_pairwise_spread;
    inf_ctx.dev_tri_params = dev_tri_params;
    inf_ctx.pipe_config = &pipe_config;
    inf_ctx.cometh = new ComethPipeline();

    ret = pthread_create(&inference_thread, nullptr, inference_thread_fn, &inf_ctx);
    if (ret) {
      log_write(ERROR, "Failed to spawn inference thread");
      running = 0;
    } else {
      inference_thread_started = true;
    }
  }

  // camera feed display windows (mocap mode) — landscape, half resolution
  if (mode == MODE_MOCAP && running) {
    for (int i = 0; i < cam_count; i++) {
      cv::namedWindow(cam_confs[i].name, cv::WINDOW_AUTOSIZE);
      cv::moveWindow(cam_confs[i].name, i * 650, 0);
    }
  }

  // Vulkan renderer (mocap mode, must be on main thread for GLFW)
  Renderer* renderer = nullptr;
  if (mode == MODE_MOCAP && running) {
    renderer_config rcfg;
    rcfg.num_keypoints = NUM_KEYPOINTS;
    rcfg.skeleton_edges = RENDER_SKELETON_EDGES;
    rcfg.num_edges = NUM_RENDER_EDGES;
    renderer = new Renderer(rcfg, &pipe_config);
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
        // label with camera index and name
        char label[64];
        snprintf(label, sizeof(label), "cam %d: %s", i, cam_confs[i].name);
        cv::putText(bgr_frames[i], label, cv::Point(10, 30),
          cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
      }

      if (cooldown > 0) {
        for (int i = 0; i < cam_count; i++)
          cv::imshow(cam_confs[i].name, bgr_frames[i]);
        cv::waitKey(16);
        if (++cooldown_counter >= cooldown)
          cooldown = 0;
        continue;
      }
      cooldown = detection_cooldown;
      cooldown_counter = 0;

      stereo_calibrator->try_frames(gray_frames);
      calibration_complete = stereo_calibrator->check_status();

      // draw detected corners on display frames
      cv::Size board_size(BOARD_WIDTH, BOARD_HEIGHT);
      for (int i = 0; i < cam_count; i++) {
        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(gray_frames[i], board_size, corners,
          cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);
        if (found)
          cv::drawChessboardCorners(bgr_frames[i], board_size, corners, true);
        cv::imshow(cam_confs[i].name, bgr_frames[i]);
      }
      cv::waitKey(16);

    } else if (mode == MODE_MOCAP) {
      // copy NV12 frames D→H for live camera feed display
      for (int i = 0; i < cam_count; i++) {
        cudaMemcpy2D(
          mocap_host_frames + i * frame_size,
          stream_conf.frame_width,
          frame_dev_ptrs[i],
          frame_pitch,
          stream_conf.frame_width,
          stream_conf.frame_height * 3 / 2,
          cudaMemcpyDeviceToHost
        );
      }

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

      // display live camera feeds with 2D keypoint overlay (no rotation — show raw orientation)
      for (int i = 0; i < cam_count; i++) {
        cv::Mat nv12(
          stream_conf.frame_height * 3 / 2,
          stream_conf.frame_width,
          CV_8UC1,
          mocap_host_frames + i * frame_size
        );
        cv::Mat bgr;
        cv::cvtColor(nv12, bgr, cv::COLOR_YUV2BGR_NV12);

        // overlay 2D keypoints if available
        // model space (288x384) is rotated+scaled+cropped from original (1280x720)
        // preprocess: rotate 90 CCW -> scale(288/720) -> center-crop height to 384
        // 90 CCW: rotated(rx,ry) came from source(src_width-1-ry, rx)
        // inverse: source(sx,sy) -> rotated(sy, src_width-1-sx)
        // so model(mx,my) -> rotated(mx, my+crop_offset) -> source(src_width-1-(my+crop_offset)/scale, mx/scale)
        if (kp2d_ready.load(std::memory_order_acquire)) {
          float scale = static_cast<float>(INPUT_WIDTH) / stream_conf.frame_height; // 288/720
          int scaled_h = static_cast<int>(stream_conf.frame_width * scale); // 1280 * 288/720 = 512
          int crop_offset = (scaled_h - INPUT_HEIGHT) / 2; // (512 - 384) / 2 = 64
          int kp_offset = i * NUM_KEYPOINTS;
          for (int k = 0; k < NUM_KEYPOINTS; k++) {
            float conf = shared_confidence[kp_offset + k];
            if (conf < 3.0f) continue;
            float mx = shared_keypoints_2d[(kp_offset + k) * 2 + 0];
            float my = shared_keypoints_2d[(kp_offset + k) * 2 + 1];
            // un-crop, un-scale, un-rotate (90 CCW inverse)
            int orig_x = static_cast<int>((stream_conf.frame_width - 1) - (my + crop_offset) / scale);
            int orig_y = static_cast<int>(mx / scale);
            if (orig_x >= 0 && orig_x < bgr.cols && orig_y >= 0 && orig_y < bgr.rows) {
              cv::Scalar color;
              if (k >= 91) color = cv::Scalar(255, 0, 0);
              else if (k >= 17 && k <= 22) color = cv::Scalar(0, 255, 255);
              else if (k >= 5 && k <= 16) color = cv::Scalar(0, 255, 0);
              else color = cv::Scalar(0, 0, 255);
              cv::circle(bgr, cv::Point(orig_x, orig_y), 4, color, -1);
            }
          }
        }

        // label with camera index and name
        {
          char label[64];
          snprintf(label, sizeof(label), "cam %d: %s", i, cam_confs[i].name);
          cv::putText(bgr, label, cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        }

        // resize to fit window (half size for display)
        cv::Mat display;
        cv::resize(bgr, display, cv::Size(bgr.cols / 2, bgr.rows / 2));
        cv::imshow(cam_confs[i].name, display);
      }
      if (kp2d_ready.load(std::memory_order_relaxed))
        kp2d_ready.store(false, std::memory_order_relaxed);
      cv::waitKey(1);

      // render latest 3D keypoints (main thread, required by GLFW)
      if (renderer && keypoints_ready.load(std::memory_order_acquire)) {
        keypoints_ready.store(false, std::memory_order_relaxed);
        if (!renderer->render_frame(shared_keypoints_3d, shared_keypoints_3d_raw, NUM_KEYPOINTS, &shared_metrics))
          running = 0; // window closed
      } else if (renderer) {
        // still poll GLFW events even without new data
        if (!renderer->render_frame(shared_keypoints_3d, shared_keypoints_3d_raw, NUM_KEYPOINTS, &shared_metrics))
          running = 0;
      }
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

  // stop cameras via TCP control channel
  log_write(INFO, "Sending STOP to all cameras...");
  for (int i = 0; i < cam_count; i++) {
    if (ctrl_fds[i] >= 0) {
      ret = send_ctrl_msg(ctrl_fds[i], CTRL_STOP);
      snprintf(logstr, sizeof(logstr), "Sent STOP to cam %d (ret=%d)", i, ret);
      log_write(INFO, logstr);
    }
  }

  // shutdown worker threads
  running = 0;
  for (int i = 0; i < thread_count; i++)
    pthread_kill(threads[i], SIGUSR2);
  for (int i = 0; i < thread_count; i++)
    pthread_join(threads[i], nullptr);

  // close network fds
  for (int i = 0; i < cam_count; i++) {
    if (ctrl_fds[i] >= 0) close(ctrl_fds[i]);
    if (ctrl_listen_fds[i] >= 0) close(ctrl_listen_fds[i]);
    if (stream_fds[i] >= 0) close(stream_fds[i]);
    if (stream_listen_fds[i] >= 0) close(stream_listen_fds[i]);
  }

  // wait for inference thread
  if (inference_thread_started)
    pthread_join(inference_thread, nullptr);

  // cleanup
  delete renderer;
  delete lens_calibrator;
  delete stereo_calibrator;
  delete predictor;
  delete triangulator;
  cudaFree(dev_kf3d_state);
  cudaFree(dev_kf3d_covariance);
  cudaFree(dev_kf3d_out);
  cudaFree(dev_kf3d_params);
  cudaFree(dev_proj_matrices);
  cudaFree(dev_cam_matrices);
  cudaFree(dev_dist_coeffs);
  cudaFree(dev_keypoints_3d);
  cudaFree(dev_reproj_error);
  cudaFree(dev_pairwise_spread);
  cudaFree(dev_tri_params);
  if (host_frames)
    free(host_frames);
  if (mocap_host_frames)
    free(mocap_host_frames);
  delete inf_ctx.cometh;
  if (hw_device_ctx)
    av_buffer_unref(&hw_device_ctx);
  cleanup_logging();

  return 0;
}
