#ifndef POSE_PREDICTOR_HPP
#define POSE_PREDICTOR_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <torch/script.h>
#include <vector>

constexpr int NUM_KEYPOINTS = 133;
constexpr int INPUT_HEIGHT = 1024;
constexpr int INPUT_WIDTH = 768;
constexpr float MEAN[3] = {123.675f/255.0f, 116.28f/255.0f, 103.53f/255.0f};
constexpr float STD[3] = {58.395f/255.0f, 57.12f/255.0f, 57.375f/255.0f};

class PosePredictor {
private:
  torch::jit::script::Module model;

  void preprocess(
    const std::vector<cv::Mat>& bgr_frames,
    torch::Tensor& rgb_tensors
  );

  torch::Tensor infer(const torch::Tensor& rgb_tensors);

  void postprocess(
    const torch::Tensor& predicted_keypoints,
    std::vector<std::pair<std::vector<float>, std::vector<float>>>& keypoints,
    std::vector<std::vector<float>>& confidence_scores
  );

public:
  PosePredictor(const std::string& model_path);
  void predict(
    const std::vector<cv::Mat>& bgr_frames,
    std::vector<std::pair<std::vector<float>, std::vector<float>>>& keypoints,
    std::vector<std::vector<float>>& confidence_scores
  );
};

#endif // POSE_PREDICTOR_HPP
