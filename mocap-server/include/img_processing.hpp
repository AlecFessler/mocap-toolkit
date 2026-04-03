#ifndef IMG_PROCESSING_HPP
#define IMG_PROCESSING_HPP

#include <opencv2/opencv.hpp>

constexpr int PROCESSED_WIDTH = 768;
constexpr int PROCESSED_HEIGHT = 1024;

cv::Mat wide_to_3_4_ar(const cv::Mat& input);

#endif // IMG_PROCESSING_HPP
