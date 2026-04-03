#ifndef TRIANGULATOR_HPP
#define TRIANGULATOR_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "lens_calibration.hpp"
#include "parse_conf.h"
#include "pose_predictor.hpp"

class Triangulator {
private:
  int m_cam_count;
  // projection_matrices[i] is the 3x4 projection matrix for camera i
  // camera 0 is the world reference frame
  std::vector<cv::Mat> m_proj_matrices;
  std::vector<cv::Mat> m_cam_matrices;
  std::vector<cv::Mat> m_dist_coeffs;

public:
  Triangulator(
    calibration_params* cam_params,
    int cam_count,
    const std::string& stereo_params_dir,
    cam_conf* confs
  );

  // Takes 2D keypoints from all cameras, returns 3D keypoints
  // keypoints_2d: [cam_count][NUM_KEYPOINTS] pairs of (x,y)
  // confidence: [cam_count][NUM_KEYPOINTS] confidence scores
  // keypoints_3d: output [NUM_KEYPOINTS] 3D points (NaN if not triangulatable)
  void triangulate(
    const std::vector<std::vector<std::pair<float, float>>>& keypoints_2d,
    const std::vector<std::vector<float>>& confidence,
    std::vector<cv::Point3f>& keypoints_3d,
    float confidence_threshold = 0.5f
  );
};

#endif // TRIANGULATOR_HPP
