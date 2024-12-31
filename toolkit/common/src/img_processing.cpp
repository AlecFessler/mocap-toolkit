#include <opencv2/opencv.hpp>

#include <iostream>

#include "img_processing.hpp"

cv::Mat wide_to_3_4_ar(const cv::Mat& input) {
  cv::Size input_size = input.size();

  cv::Mat rotated;
  cv::rotate(input, rotated, cv::ROTATE_90_CLOCKWISE);

  float scale = static_cast<float>(PROCESSED_WIDTH) / static_cast<float>(input_size.height);

  cv::Mat scaled;
  cv::resize(rotated, scaled, cv::Size(), scale, scale, cv::INTER_LINEAR);

  int height_diff = scaled.rows - PROCESSED_HEIGHT;
  int crop_start = height_diff / 2;

  cv::Rect roi(0, crop_start, PROCESSED_WIDTH, PROCESSED_HEIGHT);
  cv::Mat result = scaled(roi).clone();

  return result;
}
