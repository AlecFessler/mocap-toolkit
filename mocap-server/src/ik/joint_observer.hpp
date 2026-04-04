#ifndef IK_JOINT_OBSERVER_HPP
#define IK_JOINT_OBSERVER_HPP

#include <Eigen/Dense>

// Per-DOF Kalman filter with constant-acceleration motion model.
// State: [position, velocity, acceleration]
// Measurement: position only (from QPIK output)
class JointObserver {
public:
  // num_dofs: number of DOFs to track (75 for our skeleton)
  // dt: time step in seconds
  // process_noise: Q diagonal value (higher = trusts measurements more)
  // measurement_noise: R value (higher = trusts prediction more)
  JointObserver(int num_dofs, float dt, float process_noise = 0.5f, float measurement_noise = 1.0f);

  // Predict step: advance state using motion model.
  // Call this once per frame before update.
  void predict();

  // Update step: incorporate QPIK measurement.
  // q_measured: joint angles from QPIK solve [num_dofs]
  // valid: if false for a DOF, coast (predict only, no correction)
  // rom_lower/rom_upper: clamp filtered output to ROM bounds
  void update(const Eigen::VectorXf& q_measured, bool valid,
              const Eigen::VectorXf& rom_lower, const Eigen::VectorXf& rom_upper);

  // Get filtered joint angle positions
  const Eigen::VectorXf& position() const { return m_pos; }

  // Get filtered velocities (useful for QPIK velocity constraints)
  const Eigen::VectorXf& velocity() const { return m_vel; }

  // Reset to a given position (e.g., on first frame or after long occlusion)
  void reset(const Eigen::VectorXf& q_init);

  // Set per-DOF process noise (allows different noise for body vs fingers)
  void set_process_noise(int dof, float noise);

  // Number of frames since last valid measurement
  int coast_count() const { return m_coast_count; }

private:
  int m_num_dofs;
  float m_dt;
  float m_default_process_noise;
  float m_measurement_noise;
  int m_coast_count;
  bool m_initialized;

  // State per DOF: position, velocity, acceleration
  Eigen::VectorXf m_pos;
  Eigen::VectorXf m_vel;
  Eigen::VectorXf m_acc;

  // Covariance: 3x3 per DOF, stored as flat arrays for efficiency
  // P[i] = 3x3 covariance matrix for DOF i
  std::vector<Eigen::Matrix3f> m_P;

  // Per-DOF process noise
  Eigen::VectorXf m_process_noise;
};

#endif // IK_JOINT_OBSERVER_HPP
