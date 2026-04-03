#ifndef GPU_KALMAN_HPP
#define GPU_KALMAN_HPP

#include <cuda_runtime.h>

struct kalman_params {
    float process_noise;         // Q diagonal value (pixels^2/frame^2, typically 1.0-10.0)
    float measurement_noise;     // R diagonal value (pixels^2, typically 5.0-20.0)
    float gate_threshold;        // Mahalanobis distance squared gate (typically 9.0 = 3 sigma)
    float confidence_decay;      // per-frame decay when coasting (typically 0.9)
    float min_confidence;        // below this, reset filter state (typically 0.1)
    float dt;                    // time step in seconds (1.0/fps)
    float confidence_threshold;  // minimum raw confidence to accept measurement (3.0)
};

// Initialize Kalman filter state and covariance buffers
void gpu_kalman_init(
    float* dev_state,          // [num_cameras * num_keypoints * 4], zero-initialized
    float* dev_covariance,     // [num_cameras * num_keypoints * 16], large diagonal init
    int num_cameras,
    int num_keypoints,
    cudaStream_t stream
);

// Run one predict+update cycle for all keypoints across all cameras
void gpu_kalman_update(
    float* dev_state,                    // [cameras * keypoints * 4], in-place
    float* dev_covariance,               // [cameras * keypoints * 16], in-place
    const float* dev_keypoints_in,       // raw detections [cameras * keypoints * 2]
    const float* dev_confidence_in,      // raw confidence [cameras * keypoints]
    float* dev_keypoints_out,            // filtered [cameras * keypoints * 2]
    float* dev_confidence_out,           // filtered [cameras * keypoints]
    const kalman_params* dev_params,     // device-side params struct
    int num_cameras,
    int num_keypoints,
    cudaStream_t stream
);

#endif // GPU_KALMAN_HPP
