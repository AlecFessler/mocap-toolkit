#include <cmath>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "triangulator.hpp"
#include "logging.h"

Triangulator::Triangulator(
  calibration_params* cam_params,
  int cam_count,
  const std::string& stereo_params_dir,
  cam_conf* confs
) :
  m_cam_count(cam_count) {

  m_proj_matrices.resize(cam_count);
  m_cam_matrices.resize(cam_count);
  m_dist_coeffs.resize(cam_count);

  for (int i = 0; i < cam_count; i++) {
    m_cam_matrices[i] = cam_params[i].cam_matrix;
    m_dist_coeffs[i] = cam_params[i].dist_coeffs;
  }

  // camera 0 is the world reference: P0 = [I | 0]
  // We use normalized coordinates (undistorted, no K) so projection matrices don't include K
  cv::Mat identity = cv::Mat::eye(3, 3, CV_64F);
  cv::Mat zero_t = cv::Mat::zeros(3, 1, CV_64F);
  cv::hconcat(identity, zero_t, m_proj_matrices[0]); // 3x4

  // for each other camera, load stereo params relative to camera 0
  char logstr[128];
  for (int i = 1; i < cam_count; i++) {
    std::string filename =
      stereo_params_dir +
      std::string(confs[0].name) + "_" +
      std::string(confs[i].name) + "_calibration.yaml";

    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
      snprintf(
        logstr,
        sizeof(logstr),
        "Failed to open stereo calibration: %s",
        filename.c_str()
      );
      log_write(ERROR, logstr);
      // set identity as fallback
      cv::hconcat(identity, zero_t, m_proj_matrices[i]);
      continue;
    }

    cv::Mat R, T;
    fs["rotation_matrix"] >> R;
    fs["translation_matrix"] >> T;
    fs.release();

    cv::hconcat(R, T, m_proj_matrices[i]); // 3x4, no K
  }
}

void Triangulator::triangulate(
  const std::vector<std::vector<std::pair<float, float>>>& keypoints_2d,
  const std::vector<std::vector<float>>& confidence,
  std::vector<cv::Point3f>& keypoints_3d,
  float confidence_threshold
) {
  keypoints_3d.resize(NUM_KEYPOINTS);

  for (int k = 0; k < NUM_KEYPOINTS; k++) {
    // find the two cameras with highest confidence for this keypoint
    int best_cam1 = -1, best_cam2 = -1;
    float best_conf1 = -1.0f, best_conf2 = -1.0f;

    for (int c = 0; c < m_cam_count; c++) {
      if (confidence[c][k] < confidence_threshold)
        continue;

      if (confidence[c][k] > best_conf1) {
        best_conf2 = best_conf1;
        best_cam2 = best_cam1;
        best_conf1 = confidence[c][k];
        best_cam1 = c;
      } else if (confidence[c][k] > best_conf2) {
        best_conf2 = confidence[c][k];
        best_cam2 = c;
      }
    }

    if (best_cam1 < 0 || best_cam2 < 0) {
      keypoints_3d[k] = cv::Point3f(
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN()
      );
      continue;
    }

    // undistort to normalized coordinates (no K in output)
    std::vector<cv::Point2f> pts1_distorted = {
      cv::Point2f(keypoints_2d[best_cam1][k].first, keypoints_2d[best_cam1][k].second)
    };
    std::vector<cv::Point2f> pts2_distorted = {
      cv::Point2f(keypoints_2d[best_cam2][k].first, keypoints_2d[best_cam2][k].second)
    };

    std::vector<cv::Point2f> pts1_undistorted, pts2_undistorted;
    cv::undistortPoints(pts1_distorted, pts1_undistorted, m_cam_matrices[best_cam1], m_dist_coeffs[best_cam1]);
    cv::undistortPoints(pts2_distorted, pts2_undistorted, m_cam_matrices[best_cam2], m_dist_coeffs[best_cam2]);

    // triangulate
    cv::Mat pts4d;
    cv::triangulatePoints(
      m_proj_matrices[best_cam1],
      m_proj_matrices[best_cam2],
      pts1_undistorted,
      pts2_undistorted,
      pts4d
    );

    // convert from homogeneous
    float w = pts4d.at<float>(3, 0);
    if (std::abs(w) < 1e-6f) {
      keypoints_3d[k] = cv::Point3f(
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN()
      );
      continue;
    }

    keypoints_3d[k] = cv::Point3f(
      pts4d.at<float>(0, 0) / w,
      pts4d.at<float>(1, 0) / w,
      pts4d.at<float>(2, 0) / w
    );
  }
}
