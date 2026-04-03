#include <cuda_runtime.h>
#include <fstream>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "pose_predictor.hpp"
#include "gpu_preprocess.hpp"

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
  m_out_size = batch_size * NUM_KEYPOINTS * OUTPUT_HEIGHT * OUTPUT_WIDTH * sizeof(float);

  cudaMalloc(&m_dev_in, m_in_size);
  cudaMalloc(&m_dev_out, m_out_size);
  m_host_out = static_cast<float*>(malloc(m_out_size));

  // upload mean/std to device for GPU preprocessing
  cudaMalloc(&m_dev_mean, 3 * sizeof(float));
  cudaMalloc(&m_dev_std, 3 * sizeof(float));
  cudaMemcpy(m_dev_mean, MEAN, 3 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(m_dev_std, STD, 3 * sizeof(float), cudaMemcpyHostToDevice);
}

PosePredictor::~PosePredictor() {
  free(m_host_out);
  cudaFree(m_dev_in);
  cudaFree(m_dev_out);
  cudaFree(m_dev_mean);
  cudaFree(m_dev_std);
  cudaStreamDestroy(m_stream);
  delete m_context;
  delete m_engine;
  delete m_runtime;
}

void PosePredictor::postprocess(
  const float* heatmaps,
  float scale,
  int pad_x,
  int pad_y,
  std::vector<std::vector<std::pair<float, float>>>& coords,
  std::vector<std::vector<float>>& conf_scores
) {
  for (int b = 0; b < m_batch_size; b++) {
    for (int k = 0; k < NUM_KEYPOINTS; k++) {
      int max_idx = 0;
      float max_val = -1e9;
      for (int i = 0; i < OUTPUT_HEIGHT * OUTPUT_WIDTH; i++) {
        int idx = b * NUM_KEYPOINTS * OUTPUT_HEIGHT * OUTPUT_WIDTH +
                  k * OUTPUT_HEIGHT * OUTPUT_WIDTH +
                  i;
        float val = heatmaps[idx];
        if (val > max_val) {
          max_val = val;
          max_idx = i;
        }
      }

      int x = max_idx % OUTPUT_WIDTH;
      int y = max_idx / OUTPUT_WIDTH;

      float model_x = x * 4.0f;
      float model_y = y * 4.0f;

      float unpadded_x = (model_x - pad_x) / scale;
      float unpadded_y = (model_y - pad_y) / scale;

      coords[b][k].first = unpadded_x;
      coords[b][k].second = unpadded_y;
      conf_scores[b][k] = max_val;
    }
  }
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
  // The gpu_preprocess kernel outputs exactly INPUT_WIDTH x INPUT_HEIGHT (768x1024)
  // which matches the model input, so scale=1.0, pad=0
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

  m_context->setTensorAddress("input", m_dev_in);
  m_context->setTensorAddress("output", m_dev_out);

  m_context->enqueueV3(m_stream);
  cudaStreamSynchronize(m_stream);

  cudaMemcpy(
    m_host_out,
    m_dev_out,
    m_out_size,
    cudaMemcpyDeviceToHost
  );

  // scale=1.0, pad=0 since GPU preprocessing outputs exact model input size
  postprocess(
    m_host_out,
    1.0f,
    0,
    0,
    coords,
    conf_scores
  );
}
