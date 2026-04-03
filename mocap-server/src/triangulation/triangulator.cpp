#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "triangulation/triangulator.hpp"
#include "core/logging.h"

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

  // scale K matrices from calibration space to model output space
  // calibration was done at PROCESSED_WIDTH x PROCESSED_HEIGHT (768x1024)
  // model outputs at INPUT_WIDTH x INPUT_HEIGHT (288x384)
  double sx = static_cast<double>(INPUT_WIDTH) / static_cast<double>(cam_params[0].image_width);
  double sy = static_cast<double>(INPUT_HEIGHT) / static_cast<double>(cam_params[0].image_height);

  for (int i = 0; i < cam_count; i++) {
    m_cam_matrices[i] = cam_params[i].cam_matrix.clone();
    m_cam_matrices[i].at<double>(0, 0) *= sx; // fx
    m_cam_matrices[i].at<double>(0, 2) *= sx; // cx
    m_cam_matrices[i].at<double>(1, 1) *= sy; // fy
    m_cam_matrices[i].at<double>(1, 2) *= sy; // cy
    m_dist_coeffs[i] = cam_params[i].dist_coeffs;
  }

  // camera 0 is the world reference: P0 = [I | 0]
  cv::Mat identity = cv::Mat::eye(3, 3, CV_64F);
  cv::Mat zero_t = cv::Mat::zeros(3, 1, CV_64F);
  cv::hconcat(identity, zero_t, m_proj_matrices[0]);

  char logstr[128];
  for (int i = 1; i < cam_count; i++) {
    std::string filename =
      stereo_params_dir +
      std::string(confs[0].name) + "_" +
      std::string(confs[i].name) + "_calibration.yaml";

    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
      snprintf(logstr, sizeof(logstr), "Failed to open stereo calibration: %s", filename.c_str());
      log_write(ERROR, logstr);
      cv::hconcat(identity, zero_t, m_proj_matrices[i]);
      continue;
    }

    cv::Mat R, T;
    fs["rotation_matrix"] >> R;
    fs["translation_matrix"] >> T;
    fs.release();

    cv::hconcat(R, T, m_proj_matrices[i]);
  }
}

cv::Point3f Triangulator::triangulate_pair(
  int cam_a, int cam_b,
  const std::vector<std::vector<std::pair<float, float>>>& keypoints_2d,
  int kp_idx
) {
  const float nan = std::numeric_limits<float>::quiet_NaN();

  std::vector<cv::Point2f> pts_a_dist = {
    cv::Point2f(keypoints_2d[cam_a][kp_idx].first, keypoints_2d[cam_a][kp_idx].second)
  };
  std::vector<cv::Point2f> pts_b_dist = {
    cv::Point2f(keypoints_2d[cam_b][kp_idx].first, keypoints_2d[cam_b][kp_idx].second)
  };

  std::vector<cv::Point2f> pts_a_undist, pts_b_undist;
  cv::undistortPoints(pts_a_dist, pts_a_undist, m_cam_matrices[cam_a], m_dist_coeffs[cam_a]);
  cv::undistortPoints(pts_b_dist, pts_b_undist, m_cam_matrices[cam_b], m_dist_coeffs[cam_b]);

  cv::Mat pts4d;
  cv::triangulatePoints(
    m_proj_matrices[cam_a],
    m_proj_matrices[cam_b],
    pts_a_undist,
    pts_b_undist,
    pts4d
  );

  float w = pts4d.at<float>(3, 0);
  if (std::abs(w) < 1e-6f)
    return cv::Point3f(nan, nan, nan);

  return cv::Point3f(
    pts4d.at<float>(0, 0) / w,
    pts4d.at<float>(1, 0) / w,
    pts4d.at<float>(2, 0) / w
  );
}

float Triangulator::compute_reproj_error(
  cv::Point3f pt3d,
  int cam,
  std::pair<float, float> pt2d
) {
  // project 3D to normalized coords via projection matrix
  cv::Mat pt_h = (cv::Mat_<double>(4, 1) << pt3d.x, pt3d.y, pt3d.z, 1.0);
  cv::Mat proj = m_proj_matrices[cam] * pt_h; // 3x1 normalized
  double pz = proj.at<double>(2);
  if (std::abs(pz) < 1e-10)
    return std::numeric_limits<float>::quiet_NaN();

  double px = proj.at<double>(0) / pz;
  double py = proj.at<double>(1) / pz;

  // normalized -> pixel via K
  const cv::Mat& K = m_cam_matrices[cam];
  double px_pixel = K.at<double>(0, 0) * px + K.at<double>(0, 2);
  double py_pixel = K.at<double>(1, 1) * py + K.at<double>(1, 2);

  double dx = px_pixel - pt2d.first;
  double dy = py_pixel - pt2d.second;
  return static_cast<float>(std::sqrt(dx * dx + dy * dy));
}

static inline float median3(float a, float b, float c) {
  if (a > b) std::swap(a, b);
  if (b > c) std::swap(b, c);
  if (a > b) std::swap(a, b);
  return b;
}

static inline float dist3d(cv::Point3f a, cv::Point3f b) {
  float dx = a.x - b.x;
  float dy = a.y - b.y;
  float dz = a.z - b.z;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

void Triangulator::triangulate(
  const std::vector<std::vector<std::pair<float, float>>>& keypoints_2d,
  const std::vector<std::vector<float>>& confidence,
  std::vector<cv::Point3f>& keypoints_3d,
  frame_metrics* metrics,
  float confidence_threshold
) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  keypoints_3d.resize(NUM_KEYPOINTS);

  // initialize metrics if provided
  if (metrics) {
    for (int k = 0; k < NUM_KEYPOINTS; k++) {
      metrics->reproj_error[k] = nan;
      metrics->pairwise_spread[k] = nan;
      metrics->num_views[k] = 0;
    }
    for (int b = 0; b < NUM_BONES; b++)
      metrics->bone_lengths[b] = nan;
  }

  for (int k = 0; k < NUM_KEYPOINTS; k++) {
    // collect visible cameras
    int visible[MAX_CAMERAS];
    int num_visible = 0;
    for (int c = 0; c < m_cam_count; c++) {
      if (confidence[c][k] >= confidence_threshold)
        visible[num_visible++] = c;
    }

    if (metrics)
      metrics->num_views[k] = num_visible;

    if (num_visible < 2) {
      keypoints_3d[k] = cv::Point3f(nan, nan, nan);
      continue;
    }

    if (num_visible == 2) {
      // single pair triangulation
      cv::Point3f pt = triangulate_pair(visible[0], visible[1], keypoints_2d, k);
      keypoints_3d[k] = pt;

      if (metrics && !std::isnan(pt.x)) {
        float e0 = compute_reproj_error(pt, visible[0], keypoints_2d[visible[0]][k]);
        float e1 = compute_reproj_error(pt, visible[1], keypoints_2d[visible[1]][k]);
        metrics->reproj_error[k] = (e0 + e1) * 0.5f;
      }

    } else {
      // 3 cameras visible: pairwise consensus
      cv::Point3f p01 = triangulate_pair(visible[0], visible[1], keypoints_2d, k);
      cv::Point3f p02 = triangulate_pair(visible[0], visible[2], keypoints_2d, k);
      cv::Point3f p12 = triangulate_pair(visible[1], visible[2], keypoints_2d, k);

      // check for NaN in any pair
      bool valid01 = !std::isnan(p01.x);
      bool valid02 = !std::isnan(p02.x);
      bool valid12 = !std::isnan(p12.x);
      int valid_count = valid01 + valid02 + valid12;

      if (valid_count == 0) {
        keypoints_3d[k] = cv::Point3f(nan, nan, nan);
        continue;
      }

      cv::Point3f final_pt;
      if (valid_count == 3) {
        // component-wise median of 3 estimates
        final_pt.x = median3(p01.x, p02.x, p12.x);
        final_pt.y = median3(p01.y, p02.y, p12.y);
        final_pt.z = median3(p01.z, p02.z, p12.z);

        if (metrics) {
          // pairwise spread = max distance between any two estimates
          float d01_02 = dist3d(p01, p02);
          float d01_12 = dist3d(p01, p12);
          float d02_12 = dist3d(p02, p12);
          metrics->pairwise_spread[k] = std::max({d01_02, d01_12, d02_12});
        }
      } else if (valid_count == 2) {
        // average the two valid ones
        cv::Point3f pts[2];
        int idx = 0;
        if (valid01) pts[idx++] = p01;
        if (valid02) pts[idx++] = p02;
        if (valid12) pts[idx++] = p12;
        final_pt.x = (pts[0].x + pts[1].x) * 0.5f;
        final_pt.y = (pts[0].y + pts[1].y) * 0.5f;
        final_pt.z = (pts[0].z + pts[1].z) * 0.5f;
      } else {
        // only one valid pair
        if (valid01) final_pt = p01;
        else if (valid02) final_pt = p02;
        else final_pt = p12;
      }

      keypoints_3d[k] = final_pt;

      // reprojection error against all visible cameras
      if (metrics && !std::isnan(final_pt.x)) {
        float total_err = 0.0f;
        for (int v = 0; v < num_visible; v++)
          total_err += compute_reproj_error(final_pt, visible[v], keypoints_2d[visible[v]][k]);
        metrics->reproj_error[k] = total_err / num_visible;
      }
    }
  }

  // compute bone lengths
  if (metrics) {
    for (int b = 0; b < NUM_BONES; b++) {
      int a = BONE_EDGES[b].a;
      int bi = BONE_EDGES[b].b;
      if (!std::isnan(keypoints_3d[a].x) && !std::isnan(keypoints_3d[bi].x))
        metrics->bone_lengths[b] = dist3d(keypoints_3d[a], keypoints_3d[bi]);
    }
  }
}
