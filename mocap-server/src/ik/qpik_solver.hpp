#ifndef IK_QPIK_SOLVER_HPP
#define IK_QPIK_SOLVER_HPP

#include "ik/skeleton.hpp"

struct QPIKParams {
  int max_ik_iters = 20;          // outer IK iterations (reduced from 100, early termination handles the rest)
  int max_qp_iters = 10;          // inner projected gradient descent iterations
  float convergence_threshold = 5.0f; // mm, weighted error norm to stop
  float gamma = 1.0f;             // ROM approach factor (how aggressively joints approach limits)
  float dt = 1.0f / 30.0f;       // physical timestep (seconds)
  float lambda = 50.0f;           // regularization: penalizes joint velocity to prevent wild swings
                                   // H = lambda*I + J^T*W*J. Higher = more conservative/stable

  // Per-DOF velocity limits (rad/s for rotational, mm/s for translational root)
  float body_vel_limit = 10.0f;   // rad/s for body joints
  float finger_vel_limit = 15.0f; // rad/s for finger joints
  float root_vel_limit = 3000.0f; // mm/s for root translation

  // IK target weights by region
  float weight_body = 1.0f;
  float weight_feet = 0.5f;
  float weight_hands = 0.3f;
};

class QPIKSolver {
public:
  QPIKSolver(Skeleton& skeleton, const QPIKParams& params = QPIKParams{});

  // Solve IK for one frame.
  // target_positions: desired 3D positions for each site (indexed by site index, not COCO index)
  // target_valid: which sites have valid targets (non-NaN and non-outlier)
  // q_prev_velocity: accumulated velocity from previous frame's observer (for velocity constraint)
  // Updates skeleton.q in-place.
  // Returns weighted error norm after solving.
  float solve(const std::vector<Eigen::Vector3f>& target_positions,
              const std::vector<bool>& target_valid,
              const Eigen::VectorXf& q_prev_velocity);

private:
  Skeleton& m_skeleton;
  QPIKParams m_params;

  // Precomputed ROM bounds
  Eigen::VectorXf m_rom_lower;
  Eigen::VectorXf m_rom_upper;

  // Per-DOF velocity limits
  Eigen::VectorXf m_vel_limits;

  // Weight per site (3 values per site, same for xyz)
  Eigen::VectorXf m_site_weights;

  void build_velocity_limits();
  void build_site_weights();

  // Solve box-constrained QP: min 0.5*x^T*H*x + g^T*x, s.t. lb <= x <= ub
  // Uses Cholesky solve + projection + iteration for active bounds
  void solve_box_qp(const Eigen::MatrixXf& H, const Eigen::VectorXf& g,
                    const Eigen::VectorXf& lb, const Eigen::VectorXf& ub,
                    Eigen::VectorXf& x) const;
};

#endif // IK_QPIK_SOLVER_HPP
