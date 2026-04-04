#include <cmath>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <vector>

#include "calibration/lens_calibration.hpp"
#include "calibration/stereo_calibration.hpp"
#include "core/logging.h"

static int pair_idx(int i, int j, int cam_count) {
  return (i * (2 * cam_count - i - 1)) / 2 + (j - i - 1);
}

StereoCalibration::StereoCalibration(
  struct calibration_params* calib_params,
  int cam_count,
  cam_conf* confs,
  int frame_width,
  int frame_height,
  int board_width,
  int board_height,
  float square_size
) :
  cam_confs(confs),
  cam_count(cam_count),
  frame_width(frame_width),
  frame_height(frame_height),
  board_width(board_width),
  board_height(board_height),
  square_size(square_size) {

  objp.reserve(board_width * board_height);
  for (int i = 0; i < board_height; i++) {
    for (int j = 0; j < board_width; j++)
      objp.push_back(cv::Point3f(j*square_size, i*square_size, 0));
  }

  size_t pairs_count = cam_count * (cam_count - 1) / 2;
  stereo_pairs.resize(pairs_count);

  for (int i = 0; i < cam_count - 1; i++) {
    for (int j = i + 1; j < cam_count; j++) {
      struct stereo_pair pair;
      pair.img_pts1.reserve(MIN_FRAMES);
      pair.img_pts2.reserve(MIN_FRAMES);

      pair.cam1_matrix = calib_params[i].cam_matrix;
      pair.cam2_matrix = calib_params[j].cam_matrix;
      pair.cam1_dist_coeffs = calib_params[i].dist_coeffs;
      pair.cam2_dist_coeffs = calib_params[j].dist_coeffs;

      stereo_pairs[pair_idx(i, j, cam_count)] = pair;
    }
  }
}

void StereoCalibration::try_frames(cv::Mat* frames) {
  cv::Size board_size(board_width, board_height);
  bool found_patterns[cam_count];
  std::vector<cv::Point2f> corners[cam_count];
  char logstr[256];

  auto find_chessboard_flags = cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK;
  cv::TermCriteria criteria(
    cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
    30,
    0.001
  );

  for (int i = 0; i < cam_count; i++) {
    found_patterns[i] = cv::findChessboardCorners(
      frames[i],
      board_size,
      corners[i],
      find_chessboard_flags
    );
    if (!found_patterns[i]) continue;

    cv::cornerSubPix(
      frames[i],
      corners[i],
      cv::Size(11,11),
      cv::Size(-1,-1),
      criteria
    );
  }

  // log which cameras found corners
  snprintf(logstr, sizeof(logstr), "Corners found:");
  for (int i = 0; i < cam_count; i++) {
    char buf[64];
    snprintf(buf, sizeof(buf), " %s=%s", cam_confs[i].name, found_patterns[i] ? "YES" : "no");
    strncat(logstr, buf, sizeof(logstr) - strlen(logstr) - 1);
  }
  log_write(INFO, logstr);

  for (int i = 0; i < cam_count - 1; i++) {
    for (int j = i + 1; j < cam_count; j++) {
      if (!(found_patterns[i] && found_patterns[j]))
        continue;

      int idx = pair_idx(i, j, cam_count);
      stereo_pairs[idx].img_pts1.push_back(corners[i]);
      stereo_pairs[idx].img_pts2.push_back(corners[j]);

      size_t n = stereo_pairs[idx].img_pts1.size();

      // on first accepted frame, log corner coordinate ranges as sanity check
      if (n == 1) {
        float min_x1 = 1e9f, max_x1 = -1e9f, min_y1 = 1e9f, max_y1 = -1e9f;
        float min_x2 = 1e9f, max_x2 = -1e9f, min_y2 = 1e9f, max_y2 = -1e9f;
        for (const auto& pt : corners[i]) {
          min_x1 = std::min(min_x1, pt.x); max_x1 = std::max(max_x1, pt.x);
          min_y1 = std::min(min_y1, pt.y); max_y1 = std::max(max_y1, pt.y);
        }
        for (const auto& pt : corners[j]) {
          min_x2 = std::min(min_x2, pt.x); max_x2 = std::max(max_x2, pt.x);
          min_y2 = std::min(min_y2, pt.y); max_y2 = std::max(max_y2, pt.y);
        }
        snprintf(logstr, sizeof(logstr),
          "Pair %s_%s first frame corners: cam%d=[%.0f-%.0f, %.0f-%.0f] cam%d=[%.0f-%.0f, %.0f-%.0f]",
          cam_confs[i].name, cam_confs[j].name,
          i, min_x1, max_x1, min_y1, max_y1,
          j, min_x2, max_x2, min_y2, max_y2);
        log_write(INFO, logstr);
      }

      snprintf(logstr, sizeof(logstr), "Pair %s_%s: %zu frames collected",
        cam_confs[i].name, cam_confs[j].name, n);
      log_write(INFO, logstr);
    }
  }
}

bool StereoCalibration::check_status() {
  size_t pairs_count = cam_count * (cam_count - 1) / 2;
  bool all_zero = true;
  bool calibration_complete = true;
  for (size_t i = 0; i < pairs_count; i++) {
    size_t frames = stereo_pairs[i].img_pts1.size();

    if (frames > 0)
      all_zero = false;
    else
      continue;

    if (frames <= MIN_FRAMES)
      calibration_complete = false;
  }

  return calibration_complete && !all_zero;
}

void StereoCalibration::calibrate() {
  char logstr[256];
  size_t pairs_count = cam_count * (cam_count - 1) / 2;

  int ci = 0, cj = 1; // track which pair we're on for logging
  for (size_t p = 0; p < pairs_count; p++) {
    // recover (i, j) from linear index
    // iterate to find the right pair
    ci = 0; cj = 0;
    int idx = 0;
    for (int i = 0; i < cam_count - 1 && cj == 0; i++) {
      for (int j = i + 1; j < cam_count; j++) {
        if (idx == (int)p) { ci = i; cj = j; break; }
        idx++;
      }
    }

    if (stereo_pairs[p].img_pts1.size() == 0)
      continue;

    double rms = cv::stereoCalibrate(
      std::vector<std::vector<cv::Point3f>>(
        stereo_pairs[p].img_pts1.size(),
        objp
      ),
      stereo_pairs[p].img_pts1,
      stereo_pairs[p].img_pts2,
      stereo_pairs[p].cam1_matrix,
      stereo_pairs[p].cam1_dist_coeffs,
      stereo_pairs[p].cam2_matrix,
      stereo_pairs[p].cam2_dist_coeffs,
      cv::Size(frame_width, frame_height),
      stereo_pairs[p].rotation_mat,
      stereo_pairs[p].translation_mat,
      cv::noArray(), // essential matrix
      cv::noArray(), // fundamental matrix
      cv::CALIB_FIX_INTRINSIC
    );

    // log RMS and translation vector
    cv::Mat& T = stereo_pairs[p].translation_mat;
    double tx = T.at<double>(0);
    double ty = T.at<double>(1);
    double tz = T.at<double>(2);
    double t_mag = std::sqrt(tx*tx + ty*ty + tz*tz);

    snprintf(logstr, sizeof(logstr),
      "Pair %s_%s: RMS=%.4f | T=[%.1f, %.1f, %.1f]mm |T|=%.1fmm | %zu frames",
      cam_confs[ci].name, cam_confs[cj].name,
      rms, tx, ty, tz, t_mag,
      stereo_pairs[p].img_pts1.size());
    log_write(INFO, logstr);

    // log rotation as Rodrigues angles (degrees)
    cv::Mat rvec;
    cv::Rodrigues(stereo_pairs[p].rotation_mat, rvec);
    double rx = rvec.at<double>(0) * 180.0 / M_PI;
    double ry = rvec.at<double>(1) * 180.0 / M_PI;
    double rz = rvec.at<double>(2) * 180.0 / M_PI;
    snprintf(logstr, sizeof(logstr),
      "Pair %s_%s: rotation=[%.1f, %.1f, %.1f] deg",
      cam_confs[ci].name, cam_confs[cj].name, rx, ry, rz);
    log_write(INFO, logstr);

    stereo_pairs[p].rms_error = rms;
  }
}

void StereoCalibration::save_params(struct cam_conf* confs) {
  for (int i = 0; i < cam_count - 1; i++) {
    for (int j = i + 1; j < cam_count; j++) {
      int idx = pair_idx(i, j, cam_count);
      if (stereo_pairs[idx].img_pts1.size() == 0)
        continue;

      std::string filename = std::string(confs[i].name) + "_" + std::string(confs[j].name) + "_calibration.yaml";
      cv::FileStorage fs(filename, cv::FileStorage::WRITE);

      fs << "rotation_matrix" << stereo_pairs[idx].rotation_mat;
      fs << "translation_matrix" << stereo_pairs[idx].translation_mat;

      cv::Mat& T = stereo_pairs[idx].translation_mat;
      double tx = T.at<double>(0), ty = T.at<double>(1), tz = T.at<double>(2);
      double t_mag = std::sqrt(tx*tx + ty*ty + tz*tz);
      fs << "stereo_reproj_err" << stereo_pairs[idx].rms_error;
      fs << "frames_used" << (int)stereo_pairs[idx].img_pts1.size();
      fs << "translation_magnitude_mm" << t_mag;

      fs.release();
    }
  }
}
