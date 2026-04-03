#ifndef POSE_PREDICTOR_HPP
#define POSE_PREDICTOR_HPP

#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>
#include <string>
#include <vector>

constexpr int CHANNELS = 3;
constexpr int INPUT_HEIGHT = 384;
constexpr int INPUT_WIDTH = 288;

constexpr int NUM_KEYPOINTS = 133;

// SimCC output dimensions (input_dim * split_ratio)
constexpr int SIMCC_X_SIZE = 576;  // 288 * 2
constexpr int SIMCC_Y_SIZE = 768;  // 384 * 2
constexpr float SIMCC_SPLIT_RATIO = 2.0f;

constexpr float MEAN[3] = {123.675f/255.0f, 116.28f/255.0f, 103.53f/255.0f};
constexpr float STD[3] = {58.395f/255.0f, 57.12f/255.0f, 57.375f/255.0f};

class PosePredictor {
private:
  nvinfer1::IRuntime* m_runtime;
  nvinfer1::ICudaEngine* m_engine;
  nvinfer1::IExecutionContext* m_context;
  cudaStream_t m_stream;

  // input buffer
  int m_in_size;
  float* m_dev_in;

  // SimCC output buffers
  int m_simcc_x_size;
  int m_simcc_y_size;
  float* m_dev_simcc_x;  // [batch, 133, 576]
  float* m_dev_simcc_y;  // [batch, 133, 768]
  float* m_host_simcc_x;
  float* m_host_simcc_y;

  int m_batch_size;

  // device-side mean/std for GPU preprocessing
  float* m_dev_mean;
  float* m_dev_std;

  // device-side postprocess output buffers
  float* m_dev_keypoints;    // [batch * NUM_KEYPOINTS * 2]
  float* m_dev_confidence;   // [batch * NUM_KEYPOINTS]

public:
  PosePredictor(
    const std::string& model_path,
    const int batch_size
  );
  ~PosePredictor();

  // GPU-native path: takes NV12 device pointers directly from CUVID decoder
  void predict_gpu(
    void** dev_ptrs,
    uint32_t frame_width,
    uint32_t frame_height,
    uint32_t frame_pitch,
    std::vector<std::vector<std::pair<float, float>>>& coords,
    std::vector<std::vector<float>>& conf_scores
  );

  // GPU-native path with device output (no D→H copy, for Kalman filter pipeline)
  void predict_gpu_device(
    void** dev_ptrs,
    uint32_t frame_width,
    uint32_t frame_height,
    uint32_t frame_pitch,
    float** out_dev_keypoints,
    float** out_dev_confidence
  );

  cudaStream_t stream() const { return m_stream; }
};

#endif // POSE_PREDICTOR_HPP
