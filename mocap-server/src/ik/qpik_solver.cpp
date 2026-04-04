#include "ik/qpik_solver.hpp"
#include <cmath>

QPIKSolver::QPIKSolver(Skeleton& skeleton, const QPIKParams& params)
    : m_skeleton(skeleton), m_params(params) {
  m_rom_lower = skeleton.rom_lower();
  m_rom_upper = skeleton.rom_upper();
  build_velocity_limits();
  build_site_weights();
}

void QPIKSolver::build_velocity_limits() {
  m_vel_limits = Eigen::VectorXf(NUM_DOFS);

  for (int j = 0; j < NUM_JOINTS; j++) {
    const JointDef& jd = m_skeleton.joints[j];
    float limit;

    if (j == JOINT_ROOT_TRANS)
      limit = m_params.root_vel_limit;
    else if (j >= JOINT_L_THUMB_CMC) // all finger joints
      limit = m_params.finger_vel_limit;
    else
      limit = m_params.body_vel_limit;

    for (int d = 0; d < jd.num_dofs; d++)
      m_vel_limits[jd.dof_offset + d] = limit;
  }
}

void QPIKSolver::build_site_weights() {
  int num_sites = m_skeleton.num_sites();
  m_site_weights = Eigen::VectorXf(3 * num_sites);

  for (int s = 0; s < num_sites; s++) {
    int coco = m_skeleton.sites[s].coco_index;
    float w;

    if (coco >= 91) // hand keypoints
      w = m_params.weight_hands;
    else if (coco >= 17 && coco <= 22) // feet
      w = m_params.weight_feet;
    else // body (5-16)
      w = m_params.weight_body;

    m_site_weights[3 * s + 0] = w;
    m_site_weights[3 * s + 1] = w;
    m_site_weights[3 * s + 2] = w;
  }
}

void QPIKSolver::solve_box_qp(const Eigen::MatrixXf& H, const Eigen::VectorXf& g,
                               const Eigen::VectorXf& lb, const Eigen::VectorXf& ub,
                               Eigen::VectorXf& x) const {
  int n = static_cast<int>(g.size());

  // Solve unconstrained: x = -H^{-1} * g via Cholesky
  Eigen::LLT<Eigen::MatrixXf> llt(H);
  x = llt.solve(-g);

  // Project onto box constraints
  for (int i = 0; i < n; i++)
    x[i] = std::clamp(x[i], lb[i], ub[i]);

  // Iterative refinement for active constraints
  // When some variables are clamped, the unconstrained solution for free variables changes
  for (int iter = 0; iter < m_params.max_qp_iters; iter++) {
    // Gradient at current point
    Eigen::VectorXf grad = H * x + g;

    // Check optimality: for free variables, gradient should be ~0
    // For clamped variables, gradient should point into the constraint
    bool optimal = true;
    for (int i = 0; i < n; i++) {
      bool at_lower = (x[i] <= lb[i] + 1e-8f);
      bool at_upper = (x[i] >= ub[i] - 1e-8f);

      if (at_lower && grad[i] > 0) continue;  // clamped at lower, gradient pushes up = ok
      if (at_upper && grad[i] < 0) continue;  // clamped at upper, gradient pushes down = ok
      if (std::abs(grad[i]) > 1e-4f) {
        optimal = false;
        break;
      }
    }
    if (optimal) break;

    // Projected gradient step
    // Step size: use 1/max_diag(H) as a conservative step (spectral step)
    float max_diag = H.diagonal().maxCoeff();
    float alpha = 1.0f / max_diag;

    Eigen::VectorXf x_new = x - alpha * grad;
    for (int i = 0; i < n; i++)
      x_new[i] = std::clamp(x_new[i], lb[i], ub[i]);

    if ((x_new - x).squaredNorm() < 1e-8f) break;
    x = x_new;
  }
}

float QPIKSolver::solve(const std::vector<Eigen::Vector3f>& target_positions,
                         const std::vector<bool>& target_valid,
                         const Eigen::VectorXf& /*q_prev_velocity*/) {
  int num_sites = m_skeleton.num_sites();
  Eigen::MatrixXf J(3 * num_sites, NUM_DOFS);
  Eigen::VectorXf q_dot(NUM_DOFS);
  Eigen::VectorXf q_dot_accumulated = Eigen::VectorXf::Zero(NUM_DOFS);

  float dt = m_params.dt;
  float weighted_error = 0.0f;
  float prev_error = std::numeric_limits<float>::max();

  for (int ik_iter = 0; ik_iter < m_params.max_ik_iters; ik_iter++) {
    // Compute FK and Jacobian at current q
    m_skeleton.compute_fk();
    m_skeleton.compute_jacobian(J);

    // Compute error vector: e = target - current
    Eigen::VectorXf error(3 * num_sites);
    error.setZero();

    // Build per-site weight vector (zero for invalid targets)
    Eigen::VectorXf w = m_site_weights;
    for (int s = 0; s < num_sites; s++) {
      if (!target_valid[s]) {
        w[3 * s + 0] = 0.0f;
        w[3 * s + 1] = 0.0f;
        w[3 * s + 2] = 0.0f;
        continue;
      }
      Eigen::Vector3f diff = target_positions[s] - m_skeleton.site_positions[s];
      error.segment<3>(3 * s) = diff;
    }

    Eigen::VectorXf w_error = w.asDiagonal() * error;
    weighted_error = w_error.norm();

    // Check convergence
    if (weighted_error < m_params.convergence_threshold)
      break;

    // Early termination: if error is increasing, the IK is diverging — stop
    if (ik_iter > 0 && weighted_error > prev_error * 1.01f)
      break;
    prev_error = weighted_error;

    // Form QP: min q_dot^T * (lambda*I + J^T*W*J) * q_dot - 2 * e^T*W*J * q_dot
    // lambda controls regularization: higher = more conservative, prevents wild swings
    Eigen::MatrixXf WJ = w.asDiagonal() * J;
    Eigen::MatrixXf H = m_params.lambda * Eigen::MatrixXf::Identity(NUM_DOFS, NUM_DOFS)
                       + J.transpose() * WJ;
    Eigen::VectorXf g = -J.transpose() * w_error;

    // Compute box bounds from ROM + velocity constraints
    const Eigen::VectorXf& q = m_skeleton.q;
    Eigen::VectorXf lb(NUM_DOFS), ub(NUM_DOFS);

    for (int d = 0; d < NUM_DOFS; d++) {
      // ROM approach constraint: -gamma*(q - q_lower) <= q_dot <= gamma*(q_upper - q)
      float rom_lb = -m_params.gamma * (q[d] - m_rom_lower[d]);
      float rom_ub = m_params.gamma * (m_rom_upper[d] - q[d]);

      // Velocity constraint
      float vel_lb = -m_vel_limits[d] * dt - q_dot_accumulated[d];
      float vel_ub = m_vel_limits[d] * dt - q_dot_accumulated[d];

      lb[d] = std::max(rom_lb, vel_lb);
      ub[d] = std::min(rom_ub, vel_ub);

      // Ensure lb <= ub
      if (lb[d] > ub[d]) {
        float mid = (lb[d] + ub[d]) / 2.0f;
        lb[d] = mid;
        ub[d] = mid;
      }
    }

    // Solve box-constrained QP
    solve_box_qp(H, g, lb, ub, q_dot);

    // Update joint angles
    m_skeleton.q += q_dot;
    q_dot_accumulated += q_dot;

    // Clamp to ROM (safety)
    for (int d = 0; d < NUM_DOFS; d++)
      m_skeleton.q[d] = std::clamp(m_skeleton.q[d], m_rom_lower[d], m_rom_upper[d]);
  }

  return weighted_error;
}

