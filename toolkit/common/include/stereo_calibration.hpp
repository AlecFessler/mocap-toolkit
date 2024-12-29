#ifndef STEREO_CALIBRATION_H
#define STEREO_CALIBRATION_H

#include <array>
#include <opencv2/opencv.hpp>
#include <vector>

#include "parse_conf.h"
#include "lens_calibration.hpp"

struct stereo_pair {
  std::vector<std::vector<cv::Point2f>> img_pts1;
  std::vector<std::vector<cv::Point2f>> img_pts2;
  cv::Mat cam1_matrix;
  cv::Mat cam1_dist_coeffs;
  cv::Mat cam2_matrix;
  cv::Mat cam2_dist_coeffs;
  cv::Mat rotation_mat;
  cv::Mat translation_mat;
};

class StereoCalibration {
private:
  std::vector<cv::Point3f> objp;

  std::vector<stereo_pair> stereo_pairs;
  int cam_count;
  int frame_width;
  int frame_height;
  int board_width;
  int board_height;
  float square_size;

public:
  StereoCalibration(
    struct calibration_params* calib_params,
    int cam_count,
    int frame_width,
    int frame_height,
    int board_width,
    int board_height,
    float square_size
  );

  void try_frames(cv::Mat* frames);
  bool check_status();
  void calibrate();
  void save_params(struct cam_conf* confs);
};

#endif // STEREO_CALIBRATION_H
