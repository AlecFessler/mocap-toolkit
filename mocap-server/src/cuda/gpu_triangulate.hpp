#ifndef GPU_TRIANGULATE_HPP
#define GPU_TRIANGULATE_HPP

#include <cuda_runtime.h>

struct triangulate_params {
    float confidence_threshold;
    int num_cameras;
    int num_keypoints;
};

// Upload camera calibration data to device (call once at init)
void gpu_triangulate_init(
    double* dev_proj_matrices,    // device buffer [num_cameras * 12]
    double* dev_cam_matrices,     // device buffer [num_cameras * 9]
    double* dev_dist_coeffs,      // device buffer [num_cameras * 5]
    const double* host_proj,      // host data [num_cameras * 12]
    const double* host_cam,       // host data [num_cameras * 9]
    const double* host_dist,      // host data [num_cameras * 5]
    int num_cameras,
    cudaStream_t stream
);

// N-view DLT triangulation with lens undistortion, requires all cameras visible
void gpu_triangulate(
    const float* dev_keypoints_2d,  // [cameras * keypoints * 2]
    const float* dev_confidence,    // [cameras * keypoints]
    float* dev_keypoints_3d,        // output: [keypoints * 3]
    float* dev_reproj_error,        // output: [keypoints] avg reproj error (px)
    float* dev_pairwise_spread,     // output: [keypoints] max spread (mm), NaN if <3 views
    const double* dev_proj_matrices,
    const double* dev_cam_matrices,
    const double* dev_dist_coeffs,  // [cameras * 5] (k1, k2, p1, p2, k3)
    const triangulate_params* dev_params,
    cudaStream_t stream
);

#endif // GPU_TRIANGULATE_HPP
