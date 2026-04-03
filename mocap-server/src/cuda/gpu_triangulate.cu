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
// Returns false if singular
__device__ static bool solve3x3(
    const double A[9], // row-major
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

// Triangulate one point from two cameras using DLT (cross-product formulation)
// P1, P2 are 3x4 projection matrices (row-major, 12 doubles each)
// Returns (x, y, z) or NaN if degenerate
__device__ static void triangulate_dlt(
    const double* P1, const double* P2,
    double u1, double v1,  // normalized 2D coords in cam1
    double u2, double v2,  // normalized 2D coords in cam2
    float* out_x, float* out_y, float* out_z
) {
    // Build A (4x4) from cross-product equations:
    // row0 = u1 * P1[2,:] - P1[0,:]
    // row1 = v1 * P1[2,:] - P1[1,:]
    // row2 = u2 * P2[2,:] - P2[0,:]
    // row3 = v2 * P2[2,:] - P2[1,:]
    // Then solve A[:,0:3] * xyz = -A[:,3]

    double A[12]; // 4x3 (left side)
    double b[4];  // right side (-A[:,3])

    for (int j = 0; j < 3; j++) {
        A[0 * 3 + j] = u1 * P1[2 * 4 + j] - P1[0 * 4 + j];
        A[1 * 3 + j] = v1 * P1[2 * 4 + j] - P1[1 * 4 + j];
        A[2 * 3 + j] = u2 * P2[2 * 4 + j] - P2[0 * 4 + j];
        A[3 * 3 + j] = v2 * P2[2 * 4 + j] - P2[1 * 4 + j];
    }
    b[0] = -(u1 * P1[2 * 4 + 3] - P1[0 * 4 + 3]);
    b[1] = -(v1 * P1[2 * 4 + 3] - P1[1 * 4 + 3]);
    b[2] = -(u2 * P2[2 * 4 + 3] - P2[0 * 4 + 3]);
    b[3] = -(v2 * P2[2 * 4 + 3] - P2[1 * 4 + 3]);

    // Solve via normal equations: (A^T A) x = A^T b
    double AtA[9] = {0}; // 3x3
    double Atb[3] = {0}; // 3x1

    for (int i = 0; i < 4; i++) {
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++)
                AtA[r * 3 + c] += A[i * 3 + r] * A[i * 3 + c];
            Atb[r] += A[i * 3 + r] * b[i];
        }
    }

    double xyz[3];
    if (!solve3x3(AtA, Atb, xyz)) {
        *out_x = nanf(""); *out_y = nanf(""); *out_z = nanf("");
        return;
    }

    *out_x = static_cast<float>(xyz[0]);
    *out_y = static_cast<float>(xyz[1]);
    *out_z = static_cast<float>(xyz[2]);
}

// Undistort pixel coords to normalized coords using K^-1 + iterative undistortion
// dist: [k1, k2, p1, p2, k3] (OpenCV 5-coefficient model)
__device__ static void pixel_to_normalized(
    double px, double py,
    const double* K, // 3x3 row-major
    const double* dist, // 5 coefficients (nullptr to skip undistortion)
    double* nx, double* ny
) {
    double fx = K[0], cx = K[2];
    double fy = K[4], cy = K[5];

    // remove K to get distorted normalized coords
    double xd = (px - cx) / fx;
    double yd = (py - cy) / fy;

    if (dist == nullptr) {
        *nx = xd;
        *ny = yd;
        return;
    }

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

// Project 3D point to normalized coords through P (no K, no distortion)
__device__ static void project_to_normalized(
    float x3d, float y3d, float z3d,
    const double* P,  // 3x4
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

__device__ static float median3f(float a, float b, float c) {
    if (a > b) { float t = a; a = b; b = t; }
    if (b > c) { float t = b; b = c; c = t; }
    if (a > b) { float t = a; a = b; b = t; }
    return b;
}

__device__ static float dist3f(float x0, float y0, float z0, float x1, float y1, float z1) {
    float dx = x0 - x1, dy = y0 - y1, dz = z0 - z1;
    return sqrtf(dx * dx + dy * dy + dz * dz);
}

// ========== Main triangulation + optimization kernel ==========

__global__ void triangulate_optimize_kernel(
    const float* __restrict__ keypoints_2d,   // [cameras * num_kp * 2]
    const float* __restrict__ confidence,      // [cameras * num_kp]
    float* __restrict__ keypoints_3d,          // [num_kp * 3]
    float* __restrict__ reproj_error,          // [num_kp]
    float* __restrict__ pairwise_spread,       // [num_kp]
    const double* __restrict__ proj_matrices,  // [cameras * 12]
    const double* __restrict__ cam_matrices,   // [cameras * 9]
    const double* __restrict__ dist_coeffs,    // [cameras * 5]
    int num_cameras,
    int num_keypoints,
    float conf_threshold,
    int optim_iters
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= num_keypoints)
        return;

    const float nan_val = nanf("");

    // find visible cameras
    int visible[MAX_CAMS];
    int num_visible = 0;
    for (int c = 0; c < num_cameras && c < MAX_CAMS; c++) {
        if (confidence[c * num_keypoints + k] >= conf_threshold)
            visible[num_visible++] = c;
    }

    if (num_visible < 2) {
        keypoints_3d[k * 3 + 0] = nan_val;
        keypoints_3d[k * 3 + 1] = nan_val;
        keypoints_3d[k * 3 + 2] = nan_val;
        reproj_error[k] = nan_val;
        pairwise_spread[k] = nan_val;
        return;
    }

    // undistort all visible 2D points to normalized coords
    double norm_pts[MAX_CAMS * 2];
    for (int i = 0; i < num_visible; i++) {
        int c = visible[i];
        float px = keypoints_2d[(c * num_keypoints + k) * 2 + 0];
        float py = keypoints_2d[(c * num_keypoints + k) * 2 + 1];
        pixel_to_normalized(px, py, &cam_matrices[c * 9], &dist_coeffs[c * 5], &norm_pts[i * 2 + 0], &norm_pts[i * 2 + 1]);
    }

    // triangulate all pairs
    float candidates[3 * 3]; // max 3 pairs for 3 cameras, xyz each
    int num_candidates = 0;

    for (int i = 0; i < num_visible && num_candidates < 3; i++) {
        for (int j = i + 1; j < num_visible && num_candidates < 3; j++) {
            float cx, cy, cz;
            triangulate_dlt(
                &proj_matrices[visible[i] * 12],
                &proj_matrices[visible[j] * 12],
                norm_pts[i * 2 + 0], norm_pts[i * 2 + 1],
                norm_pts[j * 2 + 0], norm_pts[j * 2 + 1],
                &cx, &cy, &cz
            );
            if (!isnan(cx)) {
                candidates[num_candidates * 3 + 0] = cx;
                candidates[num_candidates * 3 + 1] = cy;
                candidates[num_candidates * 3 + 2] = cz;
                num_candidates++;
            }
        }
    }

    if (num_candidates == 0) {
        keypoints_3d[k * 3 + 0] = nan_val;
        keypoints_3d[k * 3 + 1] = nan_val;
        keypoints_3d[k * 3 + 2] = nan_val;
        reproj_error[k] = nan_val;
        pairwise_spread[k] = nan_val;
        return;
    }

    // initial 3D estimate: median of candidates
    float pt_x, pt_y, pt_z;
    if (num_candidates == 3) {
        pt_x = median3f(candidates[0], candidates[3], candidates[6]);
        pt_y = median3f(candidates[1], candidates[4], candidates[7]);
        pt_z = median3f(candidates[2], candidates[5], candidates[8]);

        // compute pairwise spread
        float d01 = dist3f(candidates[0], candidates[1], candidates[2],
                           candidates[3], candidates[4], candidates[5]);
        float d02 = dist3f(candidates[0], candidates[1], candidates[2],
                           candidates[6], candidates[7], candidates[8]);
        float d12 = dist3f(candidates[3], candidates[4], candidates[5],
                           candidates[6], candidates[7], candidates[8]);
        pairwise_spread[k] = fmaxf(fmaxf(d01, d02), d12);
    } else if (num_candidates == 2) {
        pt_x = (candidates[0] + candidates[3]) * 0.5f;
        pt_y = (candidates[1] + candidates[4]) * 0.5f;
        pt_z = (candidates[2] + candidates[5]) * 0.5f;
        pairwise_spread[k] = nan_val;
    } else {
        pt_x = candidates[0];
        pt_y = candidates[1];
        pt_z = candidates[2];
        pairwise_spread[k] = nan_val;
    }

    // ========== GAUSS-NEWTON OPTIMIZATION (normalized undistorted space) ==========
    for (int iter = 0; iter < optim_iters; iter++) {
        double JtJ[9] = {0};
        double Jtr[3] = {0};

        for (int i = 0; i < num_visible; i++) {
            int c = visible[i];
            const double* P = &proj_matrices[c * 12];

            // project current 3D point to normalized coords
            double proj_nx, proj_ny;
            project_to_normalized(pt_x, pt_y, pt_z, P, &proj_nx, &proj_ny);
            if (isnan(proj_nx))
                continue;

            // residual in normalized space (vs undistorted observations)
            double rx = proj_nx - norm_pts[i * 2 + 0];
            double ry = proj_ny - norm_pts[i * 2 + 1];

            // Jacobian J (2x3): d(normalized)/d(point3d)
            double h0 = P[0] * pt_x + P[1] * pt_y + P[2] * pt_z + P[3];
            double h1 = P[4] * pt_x + P[5] * pt_y + P[6] * pt_z + P[7];
            double h2 = P[8] * pt_x + P[9] * pt_y + P[10] * pt_z + P[11];

            if (fabs(h2) < 1e-10)
                continue;

            double inv_h2 = 1.0 / h2;
            double inv_h2_sq = inv_h2 * inv_h2;

            double dnx[3], dny[3];
            for (int d = 0; d < 3; d++) {
                dnx[d] = (P[0 * 4 + d] * h2 - P[2 * 4 + d] * h0) * inv_h2_sq;
                dny[d] = (P[1 * 4 + d] * h2 - P[2 * 4 + d] * h1) * inv_h2_sq;
            }

            double J[6];
            J[0] = dnx[0]; J[1] = dnx[1]; J[2] = dnx[2];
            J[3] = dny[0]; J[4] = dny[1]; J[5] = dny[2];

            for (int r = 0; r < 3; r++) {
                for (int c2 = 0; c2 < 3; c2++)
                    JtJ[r * 3 + c2] += J[0 * 3 + r] * J[0 * 3 + c2] + J[1 * 3 + r] * J[1 * 3 + c2];
                Jtr[r] += J[0 * 3 + r] * rx + J[1 * 3 + r] * ry;
            }
        }

        double delta[3];
        if (!solve3x3(JtJ, Jtr, delta))
            break;

        pt_x -= static_cast<float>(delta[0]);
        pt_y -= static_cast<float>(delta[1]);
        pt_z -= static_cast<float>(delta[2]);
    }

    // ========== COMPUTE FINAL REPROJECTION ERROR (pixel space) ==========
    // Convert normalized error back to pixels for interpretable metric
    float total_err = 0.0f;
    for (int i = 0; i < num_visible; i++) {
        int c = visible[i];
        const double* P = &proj_matrices[c * 12];
        const double* K = &cam_matrices[c * 9];

        double proj_nx, proj_ny;
        project_to_normalized(pt_x, pt_y, pt_z, P, &proj_nx, &proj_ny);
        if (isnan(proj_nx))
            continue;

        // scale normalized error to pixel error using focal lengths
        double fx = K[0], fy = K[4];
        double dx = (proj_nx - norm_pts[i * 2 + 0]) * fx;
        double dy = (proj_ny - norm_pts[i * 2 + 1]) * fy;
        total_err += sqrtf(static_cast<float>(dx * dx + dy * dy));
    }
    reproj_error[k] = total_err / num_visible;

    // write final 3D point
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
    // copy params to get num_keypoints on host for grid calculation
    triangulate_params h_params;
    cudaMemcpy(&h_params, dev_params, sizeof(triangulate_params), cudaMemcpyDeviceToHost);

    int block = 256;
    int grid = (h_params.num_keypoints + block - 1) / block;

    triangulate_optimize_kernel<<<grid, block, 0, stream>>>(
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
        h_params.confidence_threshold,
        h_params.optim_iterations
    );
}
