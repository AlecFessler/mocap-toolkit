#ifndef LENS_CALIBRATION_HPP
#define LENS_CALIBRATION_HPP

#include <string>
#include <opencv2/opencv.hpp>
#include <vector>

constexpr int MIN_FRAMES = 10;
constexpr double MIN_ERR = 1.0;

class LensCalibration {
private:
  int frame_width;
  int frame_height;
  int board_width;
  int board_height;
  float square_size;

  int frame_count;

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
    int frame_width,
    int frame_height,
    int board_width,
    int board_height,
    float square_size
  );
  bool try_frame(cv::Mat& gray_frame);
  void display_corners(cv::Mat& bgr_frame);
  double calibrate();
  bool check_status();
  void save_params(const std::string& filename);
};

struct calibration_params {
  cv::Mat cam_matrix;
  cv::Mat dist_coeffs;
  int image_width;
  int image_height;
};

bool load_calibration_params(
  const std::string& filename,
  struct calibration_params& params
);

#endif // LENS_CALIBRATION_HPP
