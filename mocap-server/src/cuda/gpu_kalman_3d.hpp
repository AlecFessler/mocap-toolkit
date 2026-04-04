#ifndef GPU_KALMAN_3D_HPP
#define GPU_KALMAN_3D_HPP

#include <cuda_runtime.h>

struct kalman_params {
    float process_noise;         // Q diagonal value
    float measurement_noise;     // R diagonal value
    float gate_threshold;        // Mahalanobis distance squared gate
    float confidence_decay;      // per-frame decay when coasting
    float min_confidence;        // below this, reset filter state
    float dt;                    // time step in seconds (1.0/fps)
    float confidence_threshold;  // minimum raw confidence to accept measurement
};

// Initialize 3D Kalman filter state and covariance
void gpu_kalman_3d_init(
    float* dev_state,          // [num_keypoints * 6] (x,y,z,vx,vy,vz)
    float* dev_covariance,     // [num_keypoints * 36] (6x6 row-major)
    int num_keypoints,
    cudaStream_t stream
);

// Run one predict+update cycle on triangulated 3D keypoints
// NaN inputs are treated as missing measurements (coast/predict only)
void gpu_kalman_3d_update(
    float* dev_state,                    // [num_keypoints * 6], in-place
    float* dev_covariance,               // [num_keypoints * 36], in-place
    const float* dev_keypoints_3d_in,    // raw triangulated [num_keypoints * 3]
    float* dev_keypoints_3d_out,         // filtered output [num_keypoints * 3]
    const kalman_params* dev_params,
    int num_keypoints,
    cudaStream_t stream
);

#endif // GPU_KALMAN_3D_HPP
