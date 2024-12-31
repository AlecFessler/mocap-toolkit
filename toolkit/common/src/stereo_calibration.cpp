#include <opencv2/opencv.hpp>
#include <vector>

#include "lens_calibration.hpp"
#include "stereo_calibration.hpp"

static int pair_idx(int i, int j, int cam_count) {
  return (i * (2 * cam_count - i - 1)) / 2 + (j - i - 1);
}

StereoCalibration::StereoCalibration(
  struct calibration_params* calib_params,
  int cam_count,
  int frame_width,
  int frame_height,
  int board_width,
  int board_height,
  float square_size
) :
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

  for (int i = 0; i < cam_count - 1; i++) {
    for (int j = i + 1; j < cam_count; j++) {
      if (!(found_patterns[i] && found_patterns[j]))
        continue;

      int idx = pair_idx(i, j, cam_count);
      stereo_pairs[idx].img_pts1.push_back(corners[i]);
      stereo_pairs[idx].img_pts2.push_back(corners[j]);
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
  size_t pairs_count = cam_count * (cam_count - 1) / 2;
  for (size_t i = 0; i < pairs_count; i++) {
    if (stereo_pairs[i].img_pts1.size() == 0)
      continue;

    cv::stereoCalibrate(
      std::vector<std::vector<cv::Point3f>>(
        stereo_pairs[i].img_pts1.size(),
        objp
      ),
      stereo_pairs[i].img_pts1,
      stereo_pairs[i].img_pts2,
      stereo_pairs[i].cam1_matrix,
      stereo_pairs[i].cam2_matrix,
      stereo_pairs[i].cam1_dist_coeffs,
      stereo_pairs[i].cam2_dist_coeffs,
      cv::Size(frame_width, frame_height),
      stereo_pairs[i].rotation_mat,
      stereo_pairs[i].translation_mat,
      cv::noArray(), // essential matrix
      cv::noArray(), // fundamental matrix
      cv::CALIB_FIX_INTRINSIC
    );
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

      fs.release();
    }
  }
}
