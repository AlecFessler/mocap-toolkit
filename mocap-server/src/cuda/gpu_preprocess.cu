#include <cuda_runtime.h>
#include <cstdint>

#include "cuda/gpu_preprocess.hpp"

// Fused kernel: NV12 -> RGB -> rotate 90 CW -> resize -> center crop -> normalize -> NCHW float32
//
// After rotation, the source image becomes (src_height x src_width) i.e. 720x1280
// We scale so width matches dst_width: scale = dst_width / src_height
// Then center-crop the height to dst_height
//
// Each thread computes one output pixel (dst_x, dst_y) for all 3 channels

__global__ void nv12_preprocess_kernel(
  const uint8_t* __restrict__ nv12,
  uint32_t src_width,
  uint32_t src_height,
  uint32_t src_pitch,
  float* __restrict__ dst,
  uint32_t dst_width,
  uint32_t dst_height,
  const float* __restrict__ mean,
  const float* __restrict__ std_dev
) {
  uint32_t dst_x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t dst_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (dst_x >= dst_width || dst_y >= dst_height)
    return;

  // After 90 CW rotation, the rotated image is (src_height x src_width) = (720 x 1280)
  // rotated_width = src_height, rotated_height = src_width
  uint32_t rotated_width = src_height;
  uint32_t rotated_height = src_width;

  // Scale so rotated_width -> dst_width
  float scale = static_cast<float>(dst_width) / static_cast<float>(rotated_width);
  uint32_t scaled_height = static_cast<uint32_t>(rotated_height * scale);

  // Center crop: offset in the scaled image to start the dst_height crop
  int crop_offset_y = (static_cast<int>(scaled_height) - static_cast<int>(dst_height)) / 2;

  // Map output pixel to position in the scaled image
  // scaled_x = dst_x, scaled_y = dst_y + crop_offset_y
  float scaled_x = static_cast<float>(dst_x);
  float scaled_y = static_cast<float>(dst_y) + static_cast<float>(crop_offset_y);

  // Map back to rotated image coordinates
  float rot_x = scaled_x / scale;
  float rot_y = scaled_y / scale;

  // Rotation 90 CW maps rotated(rx, ry) -> source(ry, rotated_width - 1 - rx)
  // source_x = rot_y, source_y = rotated_width - 1 - rot_x = src_height - 1 - rot_x
  float src_x_f = rot_y;
  float src_y_f = static_cast<float>(src_height - 1) - rot_x;

  // Clamp to valid source range
  src_x_f = fminf(fmaxf(src_x_f, 0.0f), static_cast<float>(src_width - 1));
  src_y_f = fminf(fmaxf(src_y_f, 0.0f), static_cast<float>(src_height - 1));

  // Nearest neighbor sampling for Y
  uint32_t src_x = static_cast<uint32_t>(src_x_f + 0.5f);
  uint32_t src_y = static_cast<uint32_t>(src_y_f + 0.5f);

  // NV12 layout: Y plane at offset 0 (src_height rows of src_pitch bytes)
  //              UV plane at offset src_pitch * src_height (src_height/2 rows of src_pitch bytes, interleaved U V)
  uint8_t Y = nv12[src_y * src_pitch + src_x];

  uint32_t uv_row = src_y / 2;
  uint32_t uv_col = (src_x / 2) * 2; // UV pairs are interleaved
  uint32_t uv_offset = src_pitch * src_height + uv_row * src_pitch + uv_col;
  uint8_t U = nv12[uv_offset];
  uint8_t V = nv12[uv_offset + 1];

  // YUV NV12 to RGB (BT.601)
  float Yf = static_cast<float>(Y);
  float Uf = static_cast<float>(U) - 128.0f;
  float Vf = static_cast<float>(V) - 128.0f;

  float R = Yf + 1.402f * Vf;
  float G = Yf - 0.344136f * Uf - 0.714136f * Vf;
  float B = Yf + 1.772f * Uf;

  R = fminf(fmaxf(R, 0.0f), 255.0f);
  G = fminf(fmaxf(G, 0.0f), 255.0f);
  B = fminf(fmaxf(B, 0.0f), 255.0f);

  // Normalize: (pixel / 255.0 - mean) / std
  float r_norm = (R / 255.0f - mean[0]) / std_dev[0];
  float g_norm = (G / 255.0f - mean[1]) / std_dev[1];
  float b_norm = (B / 255.0f - mean[2]) / std_dev[2];

  // Write in NCHW layout: dst[c * H * W + y * W + x]
  uint32_t hw = dst_height * dst_width;
  uint32_t pixel_idx = dst_y * dst_width + dst_x;
  dst[0 * hw + pixel_idx] = r_norm;
  dst[1 * hw + pixel_idx] = g_norm;
  dst[2 * hw + pixel_idx] = b_norm;
}

void gpu_preprocess_nv12(
  const uint8_t* nv12_dev_ptr,
  uint32_t src_width,
  uint32_t src_height,
  uint32_t src_pitch,
  float* dst,
  uint32_t dst_width,
  uint32_t dst_height,
  const float* mean,
  const float* std_dev,
  cudaStream_t stream
) {
  dim3 block(16, 16);
  dim3 grid(
    (dst_width + block.x - 1) / block.x,
    (dst_height + block.y - 1) / block.y
  );

  nv12_preprocess_kernel<<<grid, block, 0, stream>>>(
    nv12_dev_ptr,
    src_width,
    src_height,
    src_pitch,
    dst,
    dst_width,
    dst_height,
    mean,
    std_dev
  );
}
