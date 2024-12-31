#include <opencv2/opencv.hpp>
#include <string>
#include <torch/script.h>
#include <torch/torch.h>
#include <torch/nn/functional/padding.h>
#include <vector>

#include "pose_predictor.hpp"

PosePredictor::PosePredictor(const std::string& model_path) {
  try {
    model = torch::jit::load(model_path);
    model.to(torch::kCUDA);
    model.eval();
  } catch (const c10::Error& e) {
    throw std::runtime_error("Failed to load model: " + std::string(e.msg()));
  } catch (...) {
    throw std::runtime_error("Unknown error initializing pose predictor");
  }
}

void PosePredictor::preprocess(
  const std::vector<cv::Mat>& bgr_frames,
  torch::Tensor& rgb_tensors
) {
  cv::Size img_size = bgr_frames[0].size();

  float original_width = static_cast<float>(img_size.width);
  float original_height = static_cast<float>(img_size.height);
  float target_width = static_cast<float>(INPUT_WIDTH);
  float target_height = static_cast<float>(INPUT_HEIGHT);

  float scale_width = target_width / original_width;
  float scale_height = target_height / original_height;
  float scale = std::min(scale_width, scale_height);

  int resized_width = static_cast<int>(original_width * scale);
  int resized_height = static_cast<int>(original_height * scale);
  int pad_x = static_cast<int>((target_width - resized_width) / 2);
  int pad_y = static_cast<int>((target_height - resized_height) / 2);

  rgb_tensors = torch::zeros({static_cast<long>(bgr_frames.size()), 3, INPUT_HEIGHT, INPUT_WIDTH});

  for (int i = 0; i < bgr_frames.size(); i++) {
    cv::Mat padded(
      INPUT_HEIGHT,
      INPUT_WIDTH,
      CV_8UC3,
      cv::Scalar(0,0,0)
    );

    cv::Mat resized;
    cv::resize(
      bgr_frames[i],
      resized,
      cv::Size(resized_width, resized_height)
    );

    resized.copyTo(
      padded(cv::Rect(
        pad_x,
        pad_y,
        resized_width,
        resized_height
      ))
    );

    cv::Mat rgb_frame;
    cv::cvtColor(
      padded,
      rgb_frame,
      cv::COLOR_BGR2RGB
    );

    torch::Tensor frame_tensor = torch::from_blob(
      rgb_frame.data,
      {INPUT_HEIGHT, INPUT_WIDTH, 3},
      torch::kUInt8
    );

    frame_tensor = frame_tensor.permute({2, 0, 1}).contiguous().to(torch::kFloat32).div(255.0);

    for (int c = 0; c < 3; c++) {
      frame_tensor[c].sub_(MEAN[c]).div_(STD[c]);
    }

    rgb_tensors[i] = frame_tensor;
  }

  rgb_tensors = rgb_tensors.to(torch::kCUDA);
}

torch::Tensor PosePredictor::infer(const torch::Tensor& rgb_tensors) {
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(rgb_tensors);

  torch::NoGradGuard no_grad;
  return model.forward(inputs).toTensor();
}

static std::pair<torch::Tensor, torch::Tensor> get_heatmap_max(const torch::Tensor& heatmaps) {
  torch::IntArrayRef dims = heatmaps.sizes();
  bool has_batch = dims.size() == 4;

  torch::Tensor flattened = has_batch ?
    heatmaps.reshape({dims[0] * dims[1], -1}) :
    heatmaps.reshape({dims[0], -1});

  std::tuple<torch::Tensor, torch::Tensor> max_result = flattened.max(1);
  torch::Tensor vals = std::get<0>(max_result);
  torch::Tensor flat_indices = std::get<1>(max_result);

  int64_t width = dims[dims.size()-1];

  torch::Tensor y_coords = flat_indices.div(width, "floor");
  torch::Tensor x_coords = flat_indices.fmod(width);

  torch::Tensor coords = torch::stack({x_coords, y_coords}, -1).to(torch::kFloat32);
  coords.masked_fill_(vals.unsqueeze(-1) <= 0, -1);

  if (has_batch) {
    coords = coords.reshape({dims[0], dims[1], 2});
    vals = vals.reshape({dims[0], dims[1]});
  }

  return {coords, vals};
}

static torch::Tensor gaussian_blur(
  const torch::Tensor& heatmaps,
  int kernel_size,
  double sigma
) {
  const int64_t B = heatmaps.size(0);
  const int64_t K = heatmaps.size(1);
  const int64_t H = heatmaps.size(2);
  const int64_t W = heatmaps.size(3);

  torch::Tensor flat_input = heatmaps.reshape({B * K, 1, H, W});

  torch::Tensor orig_max = flat_input.amax({2, 3}, true);

  int half_size = kernel_size / 2;
  torch::Tensor grid = torch::arange(
    -half_size,
    half_size + 1,
    torch::TensorOptions().dtype(torch::kFloat32).device(heatmaps.device())
  );
  torch::Tensor gauss = torch::exp(-0.5 * torch::pow(grid / sigma, 2));
  gauss /= gauss.sum();

  torch::Tensor kernel_2d = gauss.unsqueeze(1).mm(gauss.unsqueeze(0));
  kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0);

  torch::Tensor blurred = torch::conv2d(
    flat_input,
    kernel_2d,
    {},
    1,
    half_size,
    1,
    1
  );

  torch::Tensor blurred_max = blurred.amax({2, 3}, true);

  blurred_max = blurred_max.clamp_min(1e-12);
  torch::Tensor scale_factor = orig_max / blurred_max;
  blurred = blurred * scale_factor;

  blurred = blurred.squeeze(1);
  blurred = blurred.reshape({B, K, H, W});

  return blurred;
}

static torch::Tensor refine_keypoints(
  const torch::Tensor& keypoints,
  const torch::Tensor& heatmaps
) {
  torch::IntArrayRef dims = keypoints.sizes();
  int64_t batch_size = dims[0];
  int64_t num_keypoints = dims[1];

  torch::IntArrayRef hdims = heatmaps.sizes();
  int64_t height = hdims[hdims.size()-2];
  int64_t width = hdims[hdims.size()-1];

  torch::Tensor blurred = gaussian_blur(heatmaps, 11, 2.0);
  blurred = torch::clamp(blurred, 1e-3, 50.0);
  blurred = torch::log(blurred);

  torch::Tensor padded = torch::nn::functional::pad(
    blurred,
    torch::nn::functional::PadFuncOptions({1,1,1,1}).mode(torch::kReplicate)
  ).flatten();

  torch::Tensor refined_keypoints = keypoints.clone();

  for (int64_t n = 0; n < batch_size; n++) {
    torch::Tensor index = refined_keypoints[n].select(1,0) + 1 + (refined_keypoints[n].select(1,1) + 1) * (width + 2);
    index += (width + 2) * (height + 2) * torch::arange(0, num_keypoints, keypoints.device());
    index = index.reshape({-1, 1}).to(torch::kLong);

    torch::Tensor i_ = padded.index_select(0, index.squeeze());
    torch::Tensor ix1 = padded.index_select(0, index.squeeze() + 1);
    torch::Tensor iy1 = padded.index_select(0, index.squeeze() + width + 2);
    torch::Tensor ix1y1 = padded.index_select(0, index.squeeze() + width + 3);
    torch::Tensor ix1_y1_ = padded.index_select(0, index.squeeze() - width - 3);
    torch::Tensor iy1_ = padded.index_select(0, index.squeeze() - width - 2);
    torch::Tensor ix1_ = padded.index_select(0, index.squeeze() - 1);

    torch::Tensor dx = 0.5 * (ix1 - ix1_);
    torch::Tensor dy = 0.5 * (iy1 - iy1_);
    torch::Tensor derivative = torch::stack({dx, dy}, 1).reshape({num_keypoints, 2, 1});

    torch::Tensor dxx = ix1 - 2 * i_ + ix1_;
    torch::Tensor dyy = iy1 - 2 * i_ + iy1_;
    torch::Tensor dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_);

    torch::Tensor hessian = torch::stack({dxx, dxy, dxy, dyy}, 1).reshape({num_keypoints, 2, 2});
    torch::Tensor eps_tensor = torch::eye(2, keypoints.device()).mul(std::numeric_limits<float>::epsilon());
    hessian += eps_tensor.unsqueeze(0).expand({num_keypoints, -1, -1});

    torch::Tensor hessian_inv = torch::linalg::inv(hessian);
    torch::Tensor adjustment = torch::matmul(hessian_inv, derivative).squeeze(-1);

    refined_keypoints[n].sub_(adjustment);
  }

  return refined_keypoints;
}

void PosePredictor::postprocess(
  const torch::Tensor& predicted_keypoints,
  std::vector<std::pair<std::vector<float>, std::vector<float>>>& keypoints
) {
  auto [coords, vals] = get_heatmap_max(predicted_keypoints);
  torch::Tensor refined_coords = refine_keypoints(coords, predicted_keypoints);
  refined_coords = refined_coords.div(torch::tensor({191.0, 255.0}, refined_coords.device())); // heatmap size, img size / 4

  torch::Tensor coords_cpu = refined_coords.to(torch::kCPU);
  int64_t batch_size = coords_cpu.size(0);
  keypoints.resize(batch_size);

  for (int64_t n = 0; n < batch_size; n++) {
    torch::Tensor batch_tensor = coords_cpu[n];
    auto accessor = batch_tensor.accessor<float,2>();

    std::vector<float> x_vec(NUM_KEYPOINTS);
    std::vector<float> y_vec(NUM_KEYPOINTS);

    for (int64_t i = 0; i < NUM_KEYPOINTS; i++) {
      x_vec[i] = accessor[i][0];
      y_vec[i] = accessor[i][1];
    }

    keypoints[n] = std::make_pair(x_vec, y_vec);
  }
}

void PosePredictor::predict(
  const std::vector<cv::Mat>& bgr_frames,
  std::vector<std::pair<std::vector<float>, std::vector<float>>>& keypoints
) {
  torch::Tensor rgb_tensors;
  preprocess(bgr_frames, rgb_tensors);
  torch::Tensor predicted_keypoints = infer(rgb_tensors);
  postprocess(predicted_keypoints, keypoints);
}
