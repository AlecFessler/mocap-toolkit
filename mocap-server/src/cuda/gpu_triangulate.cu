#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

#include "cuda/gpu_triangulate.hpp"

#define MAX_CAMS 8

// ========== Device helpers ==========

// 3x3 determinant
__device__ static double det3(
    double a00, double a01, double a02,
    double a10, double a11, double a12,
    double a20, double a21, double a22
) {
    return a00 * (a11 * a22 - a12 * a21)
         - a01 * (a10 * a22 - a12 * a20)
         + a02 * (a10 * a21 - a11 * a20);
}

// Solve 3x3 system Ax = b via Cramer's rule
__device__ static bool solve3x3(
    const double A[9],
    const double b[3],
    double x[3]
) {
    double d = det3(
        A[0], A[1], A[2],
        A[3], A[4], A[5],
        A[6], A[7], A[8]
    );
    if (fabs(d) < 1e-12)
        return false;

    double inv_d = 1.0 / d;

    x[0] = det3(b[0], A[1], A[2],
                b[1], A[4], A[5],
                b[2], A[7], A[8]) * inv_d;

    x[1] = det3(A[0], b[0], A[2],
                A[3], b[1], A[5],
                A[6], b[2], A[8]) * inv_d;

    x[2] = det3(A[0], A[1], b[0],
                A[3], A[4], b[1],
                A[6], A[7], b[2]) * inv_d;

    return true;
}

// Undistort pixel coords to normalized coords using K^-1 + iterative undistortion
// dist: [k1, k2, p1, p2, k3] (OpenCV 5-coefficient model)
__device__ static void pixel_to_normalized(
    double px, double py,
    const double* K,    // 3x3 row-major
    const double* dist, // 5 coefficients
    double* nx, double* ny
) {
    double fx = K[0], cx = K[2];
    double fy = K[4], cy = K[5];

    double xd = (px - cx) / fx;
    double yd = (py - cy) / fy;

    double k1 = dist[0], k2 = dist[1], p1 = dist[2], p2 = dist[3], k3 = dist[4];

    // iterative undistortion (Newton's method, ~5 iterations converges)
    double x = xd, y = yd;
    for (int i = 0; i < 5; i++) {
        double r2 = x * x + y * y;
        double r4 = r2 * r2;
        double r6 = r4 * r2;
        double radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
        double dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
        double dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
        x = (xd - dx) / radial;
        y = (yd - dy) / radial;
    }

    *nx = x;
    *ny = y;
}

// Project 3D point to normalized coords through P
__device__ static void project_to_normalized(
    float x3d, float y3d, float z3d,
    const double* P,
    double* nx, double* ny
) {
    double h0 = P[0] * x3d + P[1] * y3d + P[2] * z3d + P[3];
    double h1 = P[4] * x3d + P[5] * y3d + P[6] * z3d + P[7];
    double h2 = P[8] * x3d + P[9] * y3d + P[10] * z3d + P[11];

    if (fabs(h2) < 1e-10) {
        *nx = nan(""); *ny = nan("");
        return;
    }

    *nx = h0 / h2;
    *ny = h1 / h2;
}

__device__ static float dist3f(float x0, float y0, float z0, float x1, float y1, float z1) {
    float dx = x0 - x1, dy = y0 - y1, dz = z0 - z1;
    return sqrtf(dx * dx + dy * dy + dz * dz);
}

// Pairwise DLT for spread diagnostic
__device__ static void triangulate_dlt_pair(
    const double* P1, const double* P2,
    double u1, double v1, double u2, double v2,
    float* out_x, float* out_y, float* out_z
) {
    double AtA[9] = {0};
    double Atb[3] = {0};

    const double* Ps[2] = {P1, P2};
    double us[2] = {u1, u2};
    double vs[2] = {v1, v2};

    for (int ci = 0; ci < 2; ci++) {
        const double* P = Ps[ci];
        double u = us[ci], v = vs[ci];
        double a0[3], a1[3], b0, b1;
        for (int j = 0; j < 3; j++) {
            a0[j] = u * P[2 * 4 + j] - P[0 * 4 + j];
            a1[j] = v * P[2 * 4 + j] - P[1 * 4 + j];
        }
        b0 = -(u * P[2 * 4 + 3] - P[0 * 4 + 3]);
        b1 = -(v * P[2 * 4 + 3] - P[1 * 4 + 3]);
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++)
                AtA[r * 3 + c] += a0[r] * a0[c] + a1[r] * a1[c];
            Atb[r] += a0[r] * b0 + a1[r] * b1;
        }
    }

    double xyz[3];
    if (!solve3x3(AtA, Atb, xyz)) {
        *out_x = nanf(""); *out_y = nanf(""); *out_z = nanf("");
        return;
    }
    *out_x = (float)xyz[0]; *out_y = (float)xyz[1]; *out_z = (float)xyz[2];
}

// ========== Main triangulation kernel ==========
// N-view DLT with undistortion, uses all cameras with sufficient confidence (min 2)

__global__ void triangulate_kernel(
    const float* __restrict__ keypoints_2d,
    const float* __restrict__ confidence,
    float* __restrict__ keypoints_3d,
    float* __restrict__ reproj_error,
    float* __restrict__ pairwise_spread,
    const double* __restrict__ proj_matrices,
    const double* __restrict__ cam_matrices,
    const double* __restrict__ dist_coeffs,
    int num_cameras,
    int num_keypoints,
    float conf_threshold
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= num_keypoints)
        return;

    const float nan_val = nanf("");

    // collect visible cameras and their confidence weights
    int visible[MAX_CAMS];
    float weights[MAX_CAMS];
    int num_visible = 0;
    for (int c = 0; c < num_cameras && c < MAX_CAMS; c++) {
        float conf = confidence[c * num_keypoints + k];
        if (conf >= conf_threshold) {
            visible[num_visible] = c;
            weights[num_visible] = conf;  // use raw confidence as weight
            num_visible++;
        }
    }

    if (num_visible < 2) {
        keypoints_3d[k * 3 + 0] = nan_val;
        keypoints_3d[k * 3 + 1] = nan_val;
        keypoints_3d[k * 3 + 2] = nan_val;
        reproj_error[k] = nan_val;
        pairwise_spread[k] = nan_val;
        return;
    }

    // undistort all 2D points to normalized coords
    double norm_pts[MAX_CAMS * 2];
    for (int i = 0; i < num_visible; i++) {
        int c = visible[i];
        float px = keypoints_2d[(c * num_keypoints + k) * 2 + 0];
        float py = keypoints_2d[(c * num_keypoints + k) * 2 + 1];
        pixel_to_normalized(px, py, &cam_matrices[c * 9], &dist_coeffs[c * 5],
                            &norm_pts[i * 2 + 0], &norm_pts[i * 2 + 1]);
    }

    // N-view DLT: stack 2 weighted rows per camera, solve via normal equations
    // Higher confidence cameras contribute more to the solution
    double AtA[9] = {0};
    double Atb[3] = {0};

    for (int i = 0; i < num_visible; i++) {
        const double* P = &proj_matrices[visible[i] * 12];
        double u = norm_pts[i * 2 + 0];
        double v = norm_pts[i * 2 + 1];
        double w = static_cast<double>(weights[i]);  // confidence weight

        double a0[3], a1[3], b0, b1;
        for (int j = 0; j < 3; j++) {
            a0[j] = u * P[2 * 4 + j] - P[0 * 4 + j];
            a1[j] = v * P[2 * 4 + j] - P[1 * 4 + j];
        }
        b0 = -(u * P[2 * 4 + 3] - P[0 * 4 + 3]);
        b1 = -(v * P[2 * 4 + 3] - P[1 * 4 + 3]);

        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++)
                AtA[r * 3 + c] += w * (a0[r] * a0[c] + a1[r] * a1[c]);
            Atb[r] += w * (a0[r] * b0 + a1[r] * b1);
        }
    }

    float pt_x, pt_y, pt_z;
    double xyz[3];
    if (!solve3x3(AtA, Atb, xyz)) {
        keypoints_3d[k * 3 + 0] = nan_val;
        keypoints_3d[k * 3 + 1] = nan_val;
        keypoints_3d[k * 3 + 2] = nan_val;
        reproj_error[k] = nan_val;
        pairwise_spread[k] = nan_val;
        return;
    }
    pt_x = (float)xyz[0]; pt_y = (float)xyz[1]; pt_z = (float)xyz[2];

    // pairwise spread diagnostic (triangulate each pair for comparison)
    if (num_visible >= 3) {
        float candidates[9]; // 3 pairs * xyz
        int nc = 0;
        for (int i = 0; i < num_visible && nc < 3; i++) {
            for (int j = i + 1; j < num_visible && nc < 3; j++) {
                triangulate_dlt_pair(
                    &proj_matrices[visible[i] * 12],
                    &proj_matrices[visible[j] * 12],
                    norm_pts[i * 2], norm_pts[i * 2 + 1],
                    norm_pts[j * 2], norm_pts[j * 2 + 1],
                    &candidates[nc * 3], &candidates[nc * 3 + 1], &candidates[nc * 3 + 2]);
                if (!isnan(candidates[nc * 3])) nc++;
            }
        }
        if (nc == 3) {
            float d01 = dist3f(candidates[0], candidates[1], candidates[2],
                               candidates[3], candidates[4], candidates[5]);
            float d02 = dist3f(candidates[0], candidates[1], candidates[2],
                               candidates[6], candidates[7], candidates[8]);
            float d12 = dist3f(candidates[3], candidates[4], candidates[5],
                               candidates[6], candidates[7], candidates[8]);
            pairwise_spread[k] = fmaxf(fmaxf(d01, d02), d12);
        } else {
            pairwise_spread[k] = nan_val;
        }
    } else {
        pairwise_spread[k] = nan_val;
    }

    // reprojection error (pixel space)
    float total_err = 0.0f;
    for (int i = 0; i < num_visible; i++) {
        int c = visible[i];
        const double* P = &proj_matrices[c * 12];
        const double* K = &cam_matrices[c * 9];

        double proj_nx, proj_ny;
        project_to_normalized(pt_x, pt_y, pt_z, P, &proj_nx, &proj_ny);
        if (isnan(proj_nx))
            continue;

        double fx = K[0], fy = K[4];
        double dx = (proj_nx - norm_pts[i * 2 + 0]) * fx;
        double dy = (proj_ny - norm_pts[i * 2 + 1]) * fy;
        total_err += sqrtf((float)(dx * dx + dy * dy));
    }
    reproj_error[k] = total_err / num_visible;

    keypoints_3d[k * 3 + 0] = pt_x;
    keypoints_3d[k * 3 + 1] = pt_y;
    keypoints_3d[k * 3 + 2] = pt_z;
}

// ========== Host functions ==========

void gpu_triangulate_init(
    double* dev_proj_matrices,
    double* dev_cam_matrices,
    double* dev_dist_coeffs,
    const double* host_proj,
    const double* host_cam,
    const double* host_dist,
    int num_cameras,
    cudaStream_t stream
) {
    cudaMemcpyAsync(dev_proj_matrices, host_proj, num_cameras * 12 * sizeof(double),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dev_cam_matrices, host_cam, num_cameras * 9 * sizeof(double),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dev_dist_coeffs, host_dist, num_cameras * 5 * sizeof(double),
                    cudaMemcpyHostToDevice, stream);
}

void gpu_triangulate(
    const float* dev_keypoints_2d,
    const float* dev_confidence,
    float* dev_keypoints_3d,
    float* dev_reproj_error,
    float* dev_pairwise_spread,
    const double* dev_proj_matrices,
    const double* dev_cam_matrices,
    const double* dev_dist_coeffs,
    const triangulate_params* dev_params,
    cudaStream_t stream
) {
    triangulate_params h_params;
    cudaMemcpy(&h_params, dev_params, sizeof(triangulate_params), cudaMemcpyDeviceToHost);

    int block = 256;
    int grid = (h_params.num_keypoints + block - 1) / block;

    triangulate_kernel<<<grid, block, 0, stream>>>(
        dev_keypoints_2d,
        dev_confidence,
        dev_keypoints_3d,
        dev_reproj_error,
        dev_pairwise_spread,
        dev_proj_matrices,
        dev_cam_matrices,
        dev_dist_coeffs,
        h_params.num_cameras,
        h_params.num_keypoints,
        h_params.confidence_threshold
    );
}
