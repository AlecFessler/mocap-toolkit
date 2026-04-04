#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

#include "cuda/gpu_kalman_3d.hpp"

// State: [x, y, z, vx, vy, vz] (6 floats)
// Covariance: 6x6 row-major (36 floats)
// Measurement: [x, y, z] (3 floats)
//
// F = [I3  dt*I3]    H = [I3  03]
//     [03  I3   ]

__global__ void kalman_3d_init_kernel(
    float* __restrict__ state,
    float* __restrict__ covariance,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;

    for (int i = 0; i < 6; i++)
        state[idx * 6 + i] = 0.0f;

    float* P = &covariance[idx * 36];
    for (int i = 0; i < 36; i++)
        P[i] = 0.0f;
    // large diagonal
    P[0]  = 1000.0f; // x
    P[7]  = 1000.0f; // y
    P[14] = 1000.0f; // z
    P[21] = 1000.0f; // vx
    P[28] = 1000.0f; // vy
    P[35] = 1000.0f; // vz
}

__global__ void kalman_3d_update_kernel(
    float* __restrict__ state,
    float* __restrict__ covariance,
    const float* __restrict__ kp3d_in,
    float* __restrict__ kp3d_out,
    const kalman_params* __restrict__ params,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;

    float dt = params->dt;
    float q = params->process_noise;
    float r = params->measurement_noise;
    float gate = params->gate_threshold;
    (void)params->confidence_decay;

    // load state [x, y, z, vx, vy, vz]
    float s[6];
    for (int i = 0; i < 6; i++)
        s[i] = state[idx * 6 + i];

    // load covariance 6x6
    float P[36];
    for (int i = 0; i < 36; i++)
        P[i] = covariance[idx * 36 + i];

    // load measurement
    float zx = kp3d_in[idx * 3 + 0];
    float zy = kp3d_in[idx * 3 + 1];
    float zz = kp3d_in[idx * 3 + 2];
    bool has_measurement = !isnan(zx) && !isnan(zy) && !isnan(zz);

    // ========== PREDICT ==========
    // x' = F * x: position += velocity * dt
    float ps[6];
    ps[0] = s[0] + s[3] * dt;
    ps[1] = s[1] + s[4] * dt;
    ps[2] = s[2] + s[5] * dt;
    ps[3] = s[3];
    ps[4] = s[4];
    ps[5] = s[5];

    // P' = F * P * F^T + Q
    // F = [I3  dt*I3; 03  I3]
    // For i,j in [0..2] (position block):
    //   PP[i][j] = P[i][j] + dt*P[i+3][j] + dt*P[i][j+3] + dt^2*P[i+3][j+3]
    // For i in [0..2], j in [3..5]:
    //   PP[i][j] = P[i][j] + dt*P[i+3][j]
    // For i in [3..5], j in [0..2]:
    //   PP[i][j] = P[i][j] + dt*P[i][j+3]
    // For i,j in [3..5]:
    //   PP[i][j] = P[i][j]
    float PP[36];

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            PP[i * 6 + j] = P[i * 6 + j]
                + dt * P[(i + 3) * 6 + j]
                + dt * P[i * 6 + (j + 3)]
                + dt * dt * P[(i + 3) * 6 + (j + 3)];
        }
        for (int j = 3; j < 6; j++) {
            PP[i * 6 + j] = P[i * 6 + j] + dt * P[(i + 3) * 6 + j];
        }
    }
    for (int i = 3; i < 6; i++) {
        for (int j = 0; j < 3; j++) {
            PP[i * 6 + j] = P[i * 6 + j] + dt * P[i * 6 + (j + 3)];
        }
        for (int j = 3; j < 6; j++) {
            PP[i * 6 + j] = P[i * 6 + j];
        }
    }

    // add process noise Q to diagonal
    float q_pos = q * dt * dt;
    PP[0]  += q_pos; PP[7]  += q_pos; PP[14] += q_pos;
    PP[21] += q;     PP[28] += q;     PP[35] += q;

    bool uninitialized = (PP[0] > 500.0f);

    if (has_measurement) {
        // ========== GATE CHECK ==========
        // Innovation: y = z - H * x' (H extracts first 3)
        float y[3] = {zx - ps[0], zy - ps[1], zz - ps[2]};

        // Innovation covariance: S = H * P' * H^T + R (3x3, top-left of PP + R*I)
        float S[9];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                S[i * 3 + j] = PP[i * 6 + j];
        S[0] += r; S[4] += r; S[8] += r;

        // S^-1 via cofactor expansion
        float det = S[0] * (S[4] * S[8] - S[5] * S[7])
                  - S[1] * (S[3] * S[8] - S[5] * S[6])
                  + S[2] * (S[3] * S[7] - S[4] * S[6]);

        bool valid_det = fabsf(det) > 1e-10f;
        float mahal = 0.0f;

        float Si[9]; // S inverse
        if (valid_det) {
            float inv_det = 1.0f / det;
            Si[0] = (S[4] * S[8] - S[5] * S[7]) * inv_det;
            Si[1] = (S[2] * S[7] - S[1] * S[8]) * inv_det;
            Si[2] = (S[1] * S[5] - S[2] * S[4]) * inv_det;
            Si[3] = (S[5] * S[6] - S[3] * S[8]) * inv_det;
            Si[4] = (S[0] * S[8] - S[2] * S[6]) * inv_det;
            Si[5] = (S[2] * S[3] - S[0] * S[5]) * inv_det;
            Si[6] = (S[3] * S[7] - S[4] * S[6]) * inv_det;
            Si[7] = (S[1] * S[6] - S[0] * S[7]) * inv_det;
            Si[8] = (S[0] * S[4] - S[1] * S[3]) * inv_det;

            // Mahalanobis: y^T * S^-1 * y
            for (int i = 0; i < 3; i++) {
                float row_sum = 0.0f;
                for (int j = 0; j < 3; j++)
                    row_sum += Si[i * 3 + j] * y[j];
                mahal += y[i] * row_sum;
            }
        }

        bool accept = valid_det && (mahal < gate);
        if (uninitialized)
            accept = true;

        if (accept) {
            // ========== UPDATE ==========
            // K = P' * H^T * S^-1 (6x3)
            // P' * H^T = first 3 columns of P' (since H = [I3 03])
            float K[18]; // 6x3
            for (int i = 0; i < 6; i++) {
                for (int j = 0; j < 3; j++) {
                    float sum = 0.0f;
                    for (int m = 0; m < 3; m++)
                        sum += PP[i * 6 + m] * Si[m * 3 + j];
                    K[i * 3 + j] = sum;
                }
            }

            // state update: x = x' + K * y
            for (int i = 0; i < 6; i++) {
                for (int j = 0; j < 3; j++)
                    ps[i] += K[i * 3 + j] * y[j];
            }

            // covariance: P = (I - K*H) * P'
            // IKH = I - K*H (6x6), K*H affects first 3 columns
            float IKH[36];
            for (int i = 0; i < 6; i++) {
                for (int j = 0; j < 6; j++) {
                    float kh = 0.0f;
                    if (j < 3) kh = K[i * 3 + j];
                    IKH[i * 6 + j] = (i == j ? 1.0f : 0.0f) - kh;
                }
            }

            // P = IKH * PP
            for (int i = 0; i < 6; i++) {
                for (int j = 0; j < 6; j++) {
                    float sum = 0.0f;
                    for (int m = 0; m < 6; m++)
                        sum += IKH[i * 6 + m] * PP[m * 6 + j];
                    P[i * 6 + j] = sum;
                }
            }

            if (uninitialized) {
                ps[0] = zx; ps[1] = zy; ps[2] = zz;
                ps[3] = 0.0f; ps[4] = 0.0f; ps[5] = 0.0f;
            }
        } else {
            // gate rejected — coast
            for (int i = 0; i < 36; i++)
                P[i] = PP[i];
        }
    } else {
        // ========== COAST (no measurement) ==========
        for (int i = 0; i < 36; i++)
            P[i] = PP[i];

        // cap covariance growth during coast — keep predicting but don't let uncertainty explode
        // preserve position/velocity (don't reset to zero), just clamp diagonal
        float max_cov = 1e6f;
        for (int i = 0; i < 6; i++) {
            if (P[i * 6 + i] > max_cov)
                P[i * 6 + i] = max_cov;
        }
    }

    // store state and covariance
    for (int i = 0; i < 6; i++)
        state[idx * 6 + i] = ps[i];
    for (int i = 0; i < 36; i++)
        covariance[idx * 36 + i] = P[i];

    // output: use predicted/updated position
    // NaN only if filter has never received a valid measurement (covariance still at init value)
    bool never_initialized = (P[0] >= 999.0f && !has_measurement);
    if (never_initialized) {
        kp3d_out[idx * 3 + 0] = nanf("");
        kp3d_out[idx * 3 + 1] = nanf("");
        kp3d_out[idx * 3 + 2] = nanf("");
    } else {
        kp3d_out[idx * 3 + 0] = ps[0];
        kp3d_out[idx * 3 + 1] = ps[1];
        kp3d_out[idx * 3 + 2] = ps[2];
    }
}

void gpu_kalman_3d_init(
    float* dev_state,
    float* dev_covariance,
    int num_keypoints,
    cudaStream_t stream
) {
    int block = 256;
    int grid = (num_keypoints + block - 1) / block;
    kalman_3d_init_kernel<<<grid, block, 0, stream>>>(dev_state, dev_covariance, num_keypoints);
}

void gpu_kalman_3d_update(
    float* dev_state,
    float* dev_covariance,
    const float* dev_keypoints_3d_in,
    float* dev_keypoints_3d_out,
    const kalman_params* dev_params,
    int num_keypoints,
    cudaStream_t stream
) {
    int block = 256;
    int grid = (num_keypoints + block - 1) / block;
    kalman_3d_update_kernel<<<grid, block, 0, stream>>>(
        dev_state, dev_covariance,
        dev_keypoints_3d_in, dev_keypoints_3d_out,
        dev_params, num_keypoints
    );
}
