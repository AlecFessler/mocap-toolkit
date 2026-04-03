#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

#include "cuda/gpu_kalman.hpp"

// State: [x, y, vx, vy] (4 floats)
// Covariance: 4x4 row-major (16 floats)
// Measurement: [x, y] (2 floats)
//
// F = [1 0 dt 0]    H = [1 0 0 0]
//     [0 1 0 dt]        [0 1 0 0]
//     [0 0 1  0]
//     [0 0 0  1]

__global__ void kalman_init_kernel(
    float* __restrict__ state,
    float* __restrict__ covariance,
    int total  // num_cameras * num_keypoints
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;

    // zero state
    state[idx * 4 + 0] = 0.0f;
    state[idx * 4 + 1] = 0.0f;
    state[idx * 4 + 2] = 0.0f;
    state[idx * 4 + 3] = 0.0f;

    // large diagonal covariance (high initial uncertainty)
    float* P = &covariance[idx * 16];
    for (int i = 0; i < 16; i++)
        P[i] = 0.0f;
    P[0]  = 1000.0f; // x variance
    P[5]  = 1000.0f; // y variance
    P[10] = 1000.0f; // vx variance
    P[15] = 1000.0f; // vy variance
}

__global__ void kalman_update_kernel(
    float* __restrict__ state,
    float* __restrict__ covariance,
    const float* __restrict__ kp_in,
    const float* __restrict__ conf_in,
    float* __restrict__ kp_out,
    float* __restrict__ conf_out,
    const kalman_params* __restrict__ params,
    int total  // num_cameras * num_keypoints
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;

    float dt = params->dt;
    float q = params->process_noise;
    float r = params->measurement_noise;
    float gate = params->gate_threshold;
    float decay = params->confidence_decay;
    float min_conf = params->min_confidence;

    // load state
    float x  = state[idx * 4 + 0];
    float y  = state[idx * 4 + 1];
    float vx = state[idx * 4 + 2];
    float vy = state[idx * 4 + 3];

    // load covariance (4x4 row-major)
    float P[16];
    for (int i = 0; i < 16; i++)
        P[i] = covariance[idx * 16 + i];

    // load measurement
    float zx = kp_in[idx * 2 + 0];
    float zy = kp_in[idx * 2 + 1];
    float conf = conf_in[idx];

    // ========== PREDICT ==========
    // x' = F * x
    float px = x + vx * dt;
    float py = y + vy * dt;
    float pvx = vx;
    float pvy = vy;

    // P' = F * P * F^T + Q
    // F multiplies: P'[i][j] = P[i][j] + dt*P[i+2][j] + dt*P[i][j+2] + dt^2*P[i+2][j+2]
    // for i,j in {0,1} (position rows/cols)
    // Plus Q on diagonal
    float PP[16];

    // Row 0 (x)
    PP[0]  = P[0]  + dt * P[8]  + dt * P[2]  + dt * dt * P[10] + q * dt * dt; // P'(0,0)
    PP[1]  = P[1]  + dt * P[9]  + dt * P[3]  + dt * dt * P[11]; // P'(0,1)
    PP[2]  = P[2]  + dt * P[10];  // P'(0,2)
    PP[3]  = P[3]  + dt * P[11];  // P'(0,3)

    // Row 1 (y)
    PP[4]  = P[4]  + dt * P[12] + dt * P[6]  + dt * dt * P[14]; // P'(1,0)
    PP[5]  = P[5]  + dt * P[13] + dt * P[7]  + dt * dt * P[15] + q * dt * dt; // P'(1,1)
    PP[6]  = P[6]  + dt * P[14];  // P'(1,2)
    PP[7]  = P[7]  + dt * P[15];  // P'(1,3)

    // Row 2 (vx)
    PP[8]  = P[8]  + dt * P[10]; // P'(2,0)
    PP[9]  = P[9]  + dt * P[11]; // P'(2,1)
    PP[10] = P[10] + q;           // P'(2,2)
    PP[11] = P[11];               // P'(2,3)

    // Row 3 (vy)
    PP[12] = P[12] + dt * P[14]; // P'(3,0)
    PP[13] = P[13] + dt * P[15]; // P'(3,1)
    PP[14] = P[14];               // P'(3,2)
    PP[15] = P[15] + q;           // P'(3,3)

    // ========== GATE CHECK ==========
    // Innovation: y = z - H * x'  (H extracts first 2 components)
    float yx = zx - px;
    float yy = zy - py;

    // Innovation covariance: S = H * P' * H^T + R  (2x2)
    float s00 = PP[0] + r;
    float s01 = PP[1];
    float s10 = PP[4];
    float s11 = PP[5] + r;

    // S^-1 (2x2 inverse)
    float det_s = s00 * s11 - s01 * s10;
    bool valid_det = fabsf(det_s) > 1e-10f;

    // Mahalanobis distance: d = y^T * S^-1 * y
    float mahal = 0.0f;
    if (valid_det) {
        float si00 = s11 / det_s;
        float si01 = -s01 / det_s;
        float si10 = -s10 / det_s;
        float si11 = s00 / det_s;
        mahal = yx * (si00 * yx + si01 * yy) + yy * (si10 * yx + si11 * yy);
    }

    float conf_thresh = params->confidence_threshold;
    bool accept = valid_det && (conf >= conf_thresh) && (mahal < gate);

    // check if filter is uninitialized (high covariance) — accept first detection
    bool uninitialized = (PP[0] > 500.0f);
    if (uninitialized && conf >= conf_thresh)
        accept = true;

    float out_conf;

    if (accept) {
        // ========== UPDATE ==========
        // Kalman gain: K = P' * H^T * S^-1  (4x2)
        float si00 = s11 / det_s;
        float si01 = -s01 / det_s;
        float si10 = -s10 / det_s;
        float si11 = s00 / det_s;

        // K = P' * H^T * S^-1
        // H^T = [1 0; 0 1; 0 0; 0 0], so P'*H^T = first 2 columns of P'
        float K[8]; // 4x2
        K[0] = PP[0] * si00 + PP[1] * si10;  K[1] = PP[0] * si01 + PP[1] * si11;
        K[2] = PP[4] * si00 + PP[5] * si10;  K[3] = PP[4] * si01 + PP[5] * si11;
        K[4] = PP[8] * si00 + PP[9] * si10;  K[5] = PP[8] * si01 + PP[9] * si11;
        K[6] = PP[12]* si00 + PP[13]* si10;  K[7] = PP[12]* si01 + PP[13]* si11;

        // state update: x = x' + K * y
        px  += K[0] * yx + K[1] * yy;
        py  += K[2] * yx + K[3] * yy;
        pvx += K[4] * yx + K[5] * yy;
        pvy += K[6] * yx + K[7] * yy;

        // covariance update: P = (I - K*H) * P'
        // I - K*H: 4x4, where K*H affects first 2 columns
        float IKH[16];
        IKH[0]  = 1.0f - K[0]; IKH[1]  = -K[1];      IKH[2]  = 0.0f; IKH[3]  = 0.0f;
        IKH[4]  = -K[2];       IKH[5]  = 1.0f - K[3]; IKH[6]  = 0.0f; IKH[7]  = 0.0f;
        IKH[8]  = -K[4];       IKH[9]  = -K[5];       IKH[10] = 1.0f; IKH[11] = 0.0f;
        IKH[12] = -K[6];       IKH[13] = -K[7];       IKH[14] = 0.0f; IKH[15] = 1.0f;

        // P = IKH * PP (4x4 multiply)
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                float sum = 0.0f;
                for (int m = 0; m < 4; m++)
                    sum += IKH[i * 4 + m] * PP[m * 4 + j];
                P[i * 4 + j] = sum;
            }
        }

        out_conf = conf;

        // if uninitialized, snap state directly to measurement
        if (uninitialized) {
            px = zx;
            py = zy;
            pvx = 0.0f;
            pvy = 0.0f;
        }
    } else {
        // ========== COAST (no update) ==========
        // keep predicted state, covariance grows
        for (int i = 0; i < 16; i++)
            P[i] = PP[i];

        out_conf = conf_out[idx] * decay; // decay previous output confidence
        if (out_conf < min_conf) {
            // reset filter — too long without measurement
            px = 0.0f; py = 0.0f; pvx = 0.0f; pvy = 0.0f;
            for (int i = 0; i < 16; i++) P[i] = 0.0f;
            P[0] = 1000.0f; P[5] = 1000.0f; P[10] = 1000.0f; P[15] = 1000.0f;
            out_conf = 0.0f;
        }
    }

    // store state
    state[idx * 4 + 0] = px;
    state[idx * 4 + 1] = py;
    state[idx * 4 + 2] = pvx;
    state[idx * 4 + 3] = pvy;

    // store covariance
    for (int i = 0; i < 16; i++)
        covariance[idx * 16 + i] = P[i];

    // write output
    kp_out[idx * 2 + 0] = px;
    kp_out[idx * 2 + 1] = py;
    conf_out[idx] = out_conf;
}

void gpu_kalman_init(
    float* dev_state,
    float* dev_covariance,
    int num_cameras,
    int num_keypoints,
    cudaStream_t stream
) {
    int total = num_cameras * num_keypoints;
    int block = 256;
    int grid = (total + block - 1) / block;
    kalman_init_kernel<<<grid, block, 0, stream>>>(dev_state, dev_covariance, total);
}

void gpu_kalman_update(
    float* dev_state,
    float* dev_covariance,
    const float* dev_keypoints_in,
    const float* dev_confidence_in,
    float* dev_keypoints_out,
    float* dev_confidence_out,
    const kalman_params* dev_params,
    int num_cameras,
    int num_keypoints,
    cudaStream_t stream
) {
    int total = num_cameras * num_keypoints;
    int block = 256;
    int grid = (total + block - 1) / block;
    kalman_update_kernel<<<grid, block, 0, stream>>>(
        dev_state, dev_covariance,
        dev_keypoints_in, dev_confidence_in,
        dev_keypoints_out, dev_confidence_out,
        dev_params,
        total
    );
}
