#ifndef GPU_PREPROCESS_HPP
#define GPU_PREPROCESS_HPP

#include <cuda_runtime.h>
#include <cstdint>

// Fused NV12 -> RGB -> rotate 90 CW -> resize -> center crop -> normalize -> NCHW float32
// Writes directly into the TensorRT input buffer for one batch element
void gpu_preprocess_nv12(
  const uint8_t* nv12_dev_ptr,  // CUVID decoded NV12 frame (device memory)
  uint32_t src_width,            // source frame width (e.g. 1280)
  uint32_t src_height,           // source frame height (e.g. 720)
  uint32_t src_pitch,            // CUVID pitch (512-byte aligned)
  float* dst,                    // output NCHW float buffer (device memory)
  uint32_t dst_width,            // target width after processing (e.g. 768)
  uint32_t dst_height,           // target height after processing (e.g. 1024)
  const float* mean,             // RGB channel means (device memory, 3 floats)
  const float* std_dev,          // RGB channel stds (device memory, 3 floats)
  cudaStream_t stream
);

#endif // GPU_PREPROCESS_HPP
