#include <cuda_runtime.h>
#include <fstream>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>
#include <string>
#include <vector>
#include <iostream>
#include <cmath>

#include "model/pose_predictor.hpp"
#include "cuda/gpu_preprocess.hpp"
#include "cuda/gpu_postprocess.hpp"

class Logger : public nvinfer1::ILogger {
public:
  void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
    if (severity <= nvinfer1::ILogger::Severity::kWARNING)
      std::cout << "[TRT] " << msg << "\n";
  }
} gLogger;

PosePredictor::PosePredictor(
  const std::string& model_path,
  const int batch_size
) :
  m_batch_size(batch_size) {

  std::ifstream file(model_path, std::ios::binary);
  std::vector<char> model{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};

  m_runtime = nvinfer1::createInferRuntime(gLogger);
  m_engine = m_runtime->deserializeCudaEngine(model.data(), model.size());
  m_context = m_engine->createExecutionContext();
  m_context->setInputShape("input", nvinfer1::Dims4{batch_size, CHANNELS, INPUT_HEIGHT, INPUT_WIDTH});
  cudaStreamCreate(&m_stream);

  m_in_size = batch_size * CHANNELS * INPUT_HEIGHT * INPUT_WIDTH * sizeof(float);
  m_simcc_x_size = batch_size * NUM_KEYPOINTS * SIMCC_X_SIZE * sizeof(float);
  m_simcc_y_size = batch_size * NUM_KEYPOINTS * SIMCC_Y_SIZE * sizeof(float);

  cudaMalloc(&m_dev_in, m_in_size);
  cudaMalloc(&m_dev_simcc_x, m_simcc_x_size);
  cudaMalloc(&m_dev_simcc_y, m_simcc_y_size);

  m_host_simcc_x = static_cast<float*>(malloc(m_simcc_x_size));
  m_host_simcc_y = static_cast<float*>(malloc(m_simcc_y_size));

  // upload mean/std to device for GPU preprocessing
  cudaMalloc(&m_dev_mean, 3 * sizeof(float));
  cudaMalloc(&m_dev_std, 3 * sizeof(float));
  cudaMemcpy(m_dev_mean, MEAN, 3 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(m_dev_std, STD, 3 * sizeof(float), cudaMemcpyHostToDevice);

  // device-side postprocess output buffers
  cudaMalloc(&m_dev_keypoints, batch_size * NUM_KEYPOINTS * 2 * sizeof(float));
  cudaMalloc(&m_dev_confidence, batch_size * NUM_KEYPOINTS * sizeof(float));
}

PosePredictor::~PosePredictor() {
  free(m_host_simcc_x);
  free(m_host_simcc_y);
  cudaFree(m_dev_in);
  cudaFree(m_dev_simcc_x);
  cudaFree(m_dev_simcc_y);
  cudaFree(m_dev_mean);
  cudaFree(m_dev_std);
  cudaFree(m_dev_keypoints);
  cudaFree(m_dev_confidence);
  cudaStreamDestroy(m_stream);
  delete m_context;
  delete m_engine;
  delete m_runtime;
}

void PosePredictor::predict_gpu(
  void** dev_ptrs,
  uint32_t frame_width,
  uint32_t frame_height,
  uint32_t frame_pitch,
  std::vector<std::vector<std::pair<float, float>>>& coords,
  std::vector<std::vector<float>>& conf_scores
) {
  // GPU preprocessing: NV12 -> rotate -> resize -> crop -> normalize -> NCHW
  int chw = CHANNELS * INPUT_HEIGHT * INPUT_WIDTH;
  for (int b = 0; b < m_batch_size; b++) {
    gpu_preprocess_nv12(
      static_cast<const uint8_t*>(dev_ptrs[b]),
      frame_width,
      frame_height,
      frame_pitch,
      m_dev_in + b * chw,
      INPUT_WIDTH,
      INPUT_HEIGHT,
      m_dev_mean,
      m_dev_std,
      m_stream
    );
  }
  cudaStreamSynchronize(m_stream);

  // TensorRT inference
  m_context->setTensorAddress("input", m_dev_in);
  m_context->setTensorAddress("simcc_x", m_dev_simcc_x);
  m_context->setTensorAddress("simcc_y", m_dev_simcc_y);

  m_context->enqueueV3(m_stream);
  cudaStreamSynchronize(m_stream);

  // Copy SimCC outputs to host for postprocessing
  cudaMemcpy(m_host_simcc_x, m_dev_simcc_x, m_simcc_x_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(m_host_simcc_y, m_dev_simcc_y, m_simcc_y_size, cudaMemcpyDeviceToHost);

  // SimCC postprocess: argmax on 1D distributions -> (x, y) coordinates
  for (int b = 0; b < m_batch_size; b++) {
    for (int k = 0; k < NUM_KEYPOINTS; k++) {
      // argmax on simcc_x [SIMCC_X_SIZE]
      int x_offset = b * NUM_KEYPOINTS * SIMCC_X_SIZE + k * SIMCC_X_SIZE;
      int max_x_idx = 0;
      float max_x_val = m_host_simcc_x[x_offset];
      for (int i = 1; i < SIMCC_X_SIZE; i++) {
        float val = m_host_simcc_x[x_offset + i];
        if (val > max_x_val) {
          max_x_val = val;
          max_x_idx = i;
        }
      }

      // argmax on simcc_y [SIMCC_Y_SIZE]
      int y_offset = b * NUM_KEYPOINTS * SIMCC_Y_SIZE + k * SIMCC_Y_SIZE;
      int max_y_idx = 0;
      float max_y_val = m_host_simcc_y[y_offset];
      for (int i = 1; i < SIMCC_Y_SIZE; i++) {
        float val = m_host_simcc_y[y_offset + i];
        if (val > max_y_val) {
          max_y_val = val;
          max_y_idx = i;
        }
      }

      // convert to pixel coordinates in input space
      float px = static_cast<float>(max_x_idx) / SIMCC_SPLIT_RATIO;
      float py = static_cast<float>(max_y_idx) / SIMCC_SPLIT_RATIO;

      coords[b][k].first = px;
      coords[b][k].second = py;
      // confidence: use minimum of x/y max logits
      // strong detection = both x and y have sharp peaks (high max)
      // no detection = flat distribution (low max)
      conf_scores[b][k] = std::min(max_x_val, max_y_val);
    }
  }
}

void PosePredictor::predict_gpu_device(
  void** dev_ptrs,
  uint32_t frame_width,
  uint32_t frame_height,
  uint32_t frame_pitch,
  float** out_dev_keypoints,
  float** out_dev_confidence
) {
  // GPU preprocessing: NV12 -> rotate -> resize -> crop -> normalize -> NCHW
  int chw = CHANNELS * INPUT_HEIGHT * INPUT_WIDTH;
  for (int b = 0; b < m_batch_size; b++) {
    gpu_preprocess_nv12(
      static_cast<const uint8_t*>(dev_ptrs[b]),
      frame_width, frame_height, frame_pitch,
      m_dev_in + b * chw,
      INPUT_WIDTH, INPUT_HEIGHT,
      m_dev_mean, m_dev_std,
      m_stream
    );
  }
  cudaStreamSynchronize(m_stream);

  // TensorRT inference
  m_context->setTensorAddress("input", m_dev_in);
  m_context->setTensorAddress("simcc_x", m_dev_simcc_x);
  m_context->setTensorAddress("simcc_y", m_dev_simcc_y);
  m_context->enqueueV3(m_stream);

  // GPU SimCC postprocess (stays on device)
  gpu_simcc_postprocess(
    m_dev_simcc_x, m_dev_simcc_y,
    m_dev_keypoints, m_dev_confidence,
    m_batch_size, NUM_KEYPOINTS,
    SIMCC_X_SIZE, SIMCC_Y_SIZE,
    SIMCC_SPLIT_RATIO,
    m_stream
  );

  cudaStreamSynchronize(m_stream);

  *out_dev_keypoints = m_dev_keypoints;
  *out_dev_confidence = m_dev_confidence;
}
