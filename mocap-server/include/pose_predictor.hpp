#ifndef POSE_PREDICTOR_HPP
#define POSE_PREDICTOR_HPP

#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

constexpr int CHANNELS = 3;
constexpr int INPUT_HEIGHT = 1024;
constexpr int INPUT_WIDTH = 768;

constexpr int NUM_KEYPOINTS = 308;
constexpr int OUTPUT_HEIGHT = 256;
constexpr int OUTPUT_WIDTH = 192;

constexpr float MEAN[3] = {123.675f/255.0f, 116.28f/255.0f, 103.53f/255.0f};
constexpr float STD[3] = {58.395f/255.0f, 57.12f/255.0f, 57.375f/255.0f};

class PosePredictor {
private:
  nvinfer1::IRuntime* m_runtime;
  nvinfer1::ICudaEngine* m_engine;
  nvinfer1::IExecutionContext* m_context;
  cudaStream_t m_stream;
  int m_in_size;
  int m_out_size;
  float* m_dev_in;
  float* m_dev_out;
  float* m_host_out;
  int m_batch_size;

  // device-side mean/std for GPU preprocessing
  float* m_dev_mean;
  float* m_dev_std;

  void postprocess(
    const float* heatmaps,
    float scale,
    int pad_x,
    int pad_y,
    std::vector<std::vector<std::pair<float, float>>>& coords,
    std::vector<std::vector<float>>& conf_scores
  );

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
};

#endif // POSE_PREDICTOR_HPP
