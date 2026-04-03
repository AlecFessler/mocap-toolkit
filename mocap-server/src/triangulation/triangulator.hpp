#ifndef TRIANGULATOR_HPP
#define TRIANGULATOR_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <utility>
#include <vector>

#include "calibration/lens_calibration.hpp"
#include "core/parse_conf.h"
#include "model/pose_predictor.hpp"

constexpr int NUM_BONES = 12;

struct bone_def {
  int a, b;
};

// COCO WholeBody major skeleton bones
constexpr bone_def BONE_EDGES[NUM_BONES] = {
  {5, 7},   // left shoulder -> left elbow
  {7, 9},   // left elbow -> left wrist
  {6, 8},   // right shoulder -> right elbow
  {8, 10},  // right elbow -> right wrist
  {5, 11},  // left shoulder -> left hip
  {6, 12},  // right shoulder -> right hip
  {11, 13}, // left hip -> left knee
  {13, 15}, // left knee -> left ankle
  {12, 14}, // right hip -> right knee
  {14, 16}, // right knee -> right ankle
  {5, 6},   // shoulder width
  {11, 12}, // hip width
};

struct frame_metrics {
  float reproj_error[NUM_KEYPOINTS];
  float pairwise_spread[NUM_KEYPOINTS];
  int num_views[NUM_KEYPOINTS];
  float bone_lengths[NUM_BONES];
};

class Triangulator {
private:
  int m_cam_count;
  std::vector<cv::Mat> m_proj_matrices;
  std::vector<cv::Mat> m_cam_matrices;
  std::vector<cv::Mat> m_dist_coeffs;

  cv::Point3f triangulate_pair(
    int cam_a, int cam_b,
    const std::vector<std::vector<std::pair<float, float>>>& keypoints_2d,
    int kp_idx
  );

  float compute_reproj_error(
    cv::Point3f pt3d,
    int cam,
    std::pair<float, float> pt2d
  );

public:
  Triangulator(
    calibration_params* cam_params,
    int cam_count,
    const std::string& stereo_params_dir,
    cam_conf* confs
  );

  int cam_count() const { return m_cam_count; }
  const std::vector<cv::Mat>& proj_matrices() const { return m_proj_matrices; }
  const std::vector<cv::Mat>& cam_matrices() const { return m_cam_matrices; }
  const std::vector<cv::Mat>& dist_coeffs() const { return m_dist_coeffs; }

  void triangulate(
    const std::vector<std::vector<std::pair<float, float>>>& keypoints_2d,
    const std::vector<std::vector<float>>& confidence,
    std::vector<cv::Point3f>& keypoints_3d,
    frame_metrics* metrics = nullptr,
    float confidence_threshold = 3.0f
  );
};

#endif // TRIANGULATOR_HPP
