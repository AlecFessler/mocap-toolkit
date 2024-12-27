#ifndef LENS_CALIBRATION_HPP
#define LENS_CALIBRATION_HPP

#include <cstdint>
#include <string>
#include <opencv2/opencv.hpp>
#include <vector>

constexpr uint32_t EXPECTED_FRAMES = 30;
constexpr uint32_t MIN_FRAMES = 10;
constexpr double MIN_ERR = 1.0;

class LensCalibration {
private:
  int32_t frame_width;
  int32_t frame_height;
  int32_t board_width;
  int32_t board_height;
  float square_size;

  uint32_t frame_count;

  std::vector<cv::Point3f> objp;
  std::vector<cv::Point2f> corners;
  std::vector<std::vector<cv::Point2f>> img_pts;
  std::vector<std::vector<cv::Point3f>> obj_pts;

  cv::Mat cam_matrix;
  cv::Mat dist_coeffs;
  std::vector<cv::Mat> rvecs;
  std::vector<cv::Mat> tvecs;
  std::vector<std::vector<cv::Point2f>> projected_pts;

  double reprojection_err;

public:
  LensCalibration(
    int32_t frame_width,
    int32_t frame_height,
    int32_t board_width,
    int32_t board_height,
    float square_size
  );
  bool try_frame(cv::Mat& gray_frame);
  void display_corners(cv::Mat& bgr_frame);
  void calibrate();
  bool check_status();
  //std::string final_params();
};

#endif // LENS_CALIBRATION_HPP
