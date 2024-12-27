#include <cstdint>
#include <opencv2/opencv.hpp>
#include <vector>

#include "lens_calibration.hpp"

LensCalibration::LensCalibration(
  int32_t frame_width,
  int32_t frame_height,
  int32_t board_width,
  int32_t board_height,
  float square_size
) :
  frame_width(frame_width),
  frame_height(frame_height),
  board_width(board_width),
  board_height(board_height),
  square_size(square_size),
  frame_count(0),
  reprojection_err(-1.0) {

  objp.reserve(board_width * board_height);
  for (int i = 0; i < board_height; i++) {
    for (int j = 0; j < board_width; j++)
      objp.push_back(cv::Point3f(j*square_size, i*square_size, 0));
  }

  corners.reserve(board_width * board_height);
  img_pts.reserve(EXPECTED_FRAMES);
  obj_pts.reserve(EXPECTED_FRAMES);
  rvecs.reserve(EXPECTED_FRAMES);
  tvecs.reserve(EXPECTED_FRAMES);
  projected_pts.reserve(EXPECTED_FRAMES);

  for (int i = 0; i < EXPECTED_FRAMES; i++) {
    std::vector<cv::Point2f> inner_vec;
    inner_vec.reserve(board_width * board_height);
    img_pts.push_back(inner_vec);
    projected_pts.push_back(inner_vec);

    std::vector<cv::Point3f> inner_obj_vec;
    inner_obj_vec.reserve(board_width * board_height);
    obj_pts.push_back(inner_obj_vec);
  }
}

bool LensCalibration::try_frame(cv::Mat& gray_frame) {
  cv::Size board_size(board_width, board_height);
  corners.clear();

  bool found = cv::findChessboardCorners(
    gray_frame,
    board_size,
    corners,
    cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE
  );

  if (!found) return false;

  cv::TermCriteria criteria(
    cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
    30,
    0.001
  );
  cv::cornerSubPix(
    gray_frame,
    corners,
    cv::Size(11,11),
    cv::Size(-1,-1),
    criteria
  );

  img_pts[frame_count] = corners;
  obj_pts[frame_count] = objp;
  frame_count++;

  return true;
}

void LensCalibration::display_corners(cv::Mat& bgr_frame) {
  cv::drawChessboardCorners(
    bgr_frame,
    cv::Size(board_width, board_height),
    img_pts[frame_count - 1],
    true
  );
  cv::imshow("Corners", bgr_frame);
  cv::waitKey(1);
}

void LensCalibration::calibrate() {
  cv::Size img_size(frame_width, frame_height);
  projected_pts.clear();
  rvecs.clear();
  tvecs.clear();
  cam_matrix = cv::Mat::eye(3, 3, CV_64F);
  dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);

  double rms = cv::calibrateCamera(
    obj_pts,
    img_pts,
    img_size,
    cam_matrix,
    dist_coeffs,
    rvecs,
    tvecs
  );

  double total_err = 0;
  uint32_t total_pts = 0;

  for (uint32_t i = 0; i < obj_pts.size(); i++) {
    cv::projectPoints(
      obj_pts[i],
      rvecs[i],
      tvecs[i],
      cam_matrix,
      dist_coeffs,
      projected_pts[i]
    );

    double err = cv::norm(
      cv::Mat(img_pts[i]),
      cv::Mat(projected_pts[i]),
      cv::NORM_L2
    );

    uint32_t n = obj_pts[i].size();
    total_err += err*err;
    total_pts += n;
  }

  reprojection_err = total_err / total_pts;
}

bool LensCalibration::check_status() {
  if (frame_count < MIN_FRAMES) return false;
  return reprojection_err < 1.0;
}
