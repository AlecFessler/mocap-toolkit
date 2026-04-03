#include <cuda_runtime.h>
#include <cfloat>

#include "cuda/gpu_postprocess.hpp"

// One thread per (batch_element, keypoint)
// Each thread scans simcc_x and simcc_y 1D distributions for argmax
__global__ void simcc_argmax_kernel(
    const float* __restrict__ simcc_x,
    const float* __restrict__ simcc_y,
    float* __restrict__ keypoints,
    float* __restrict__ confidence,
    int num_keypoints,
    int simcc_x_size,
    int simcc_y_size,
    float split_ratio
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = idx / num_keypoints;
    int k = idx % num_keypoints;

    // argmax on simcc_x
    int x_offset = b * num_keypoints * simcc_x_size + k * simcc_x_size;
    float max_x_val = -FLT_MAX;
    int max_x_idx = 0;
    for (int i = 0; i < simcc_x_size; i++) {
        float val = simcc_x[x_offset + i];
        if (val > max_x_val) {
            max_x_val = val;
            max_x_idx = i;
        }
    }

    // argmax on simcc_y
    int y_offset = b * num_keypoints * simcc_y_size + k * simcc_y_size;
    float max_y_val = -FLT_MAX;
    int max_y_idx = 0;
    for (int i = 0; i < simcc_y_size; i++) {
        float val = simcc_y[y_offset + i];
        if (val > max_y_val) {
            max_y_val = val;
            max_y_idx = i;
        }
    }

    // convert to pixel coordinates
    float px = static_cast<float>(max_x_idx) / split_ratio;
    float py = static_cast<float>(max_y_idx) / split_ratio;

    int kp_idx = b * num_keypoints + k;
    keypoints[kp_idx * 2 + 0] = px;
    keypoints[kp_idx * 2 + 1] = py;
    // confidence: min of x/y max logits
    confidence[kp_idx] = fminf(max_x_val, max_y_val);
}

void gpu_simcc_postprocess(
    const float* dev_simcc_x,
    const float* dev_simcc_y,
    float* dev_keypoints,
    float* dev_confidence,
    int batch_size,
    int num_keypoints,
    int simcc_x_size,
    int simcc_y_size,
    float split_ratio,
    cudaStream_t stream
) {
    int total = batch_size * num_keypoints;
    int block = 256;
    int grid = (total + block - 1) / block;

    simcc_argmax_kernel<<<grid, block, 0, stream>>>(
        dev_simcc_x, dev_simcc_y,
        dev_keypoints, dev_confidence,
        num_keypoints,
        simcc_x_size, simcc_y_size,
        split_ratio
    );
}
