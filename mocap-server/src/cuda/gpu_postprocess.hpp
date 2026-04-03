#ifndef GPU_POSTPROCESS_HPP
#define GPU_POSTPROCESS_HPP

#include <cuda_runtime.h>

// GPU SimCC argmax: extracts 2D keypoint coordinates and confidence from SimCC 1D distributions
void gpu_simcc_postprocess(
    const float* dev_simcc_x,    // [batch * num_kp * simcc_x_size]
    const float* dev_simcc_y,    // [batch * num_kp * simcc_y_size]
    float* dev_keypoints,        // output: [batch * num_kp * 2] (x, y in pixel coords)
    float* dev_confidence,       // output: [batch * num_kp]
    int batch_size,
    int num_keypoints,
    int simcc_x_size,
    int simcc_y_size,
    float split_ratio,
    cudaStream_t stream
);

#endif // GPU_POSTPROCESS_HPP
