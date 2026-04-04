#include "ik/joint_observer.hpp"
#include <cmath>

JointObserver::JointObserver(int num_dofs, float dt, float process_noise, float measurement_noise)
    : m_num_dofs(num_dofs),
      m_dt(dt),
      m_default_process_noise(process_noise),
      m_measurement_noise(measurement_noise),
      m_coast_count(0),
      m_initialized(false) {
  m_pos = Eigen::VectorXf::Zero(num_dofs);
  m_vel = Eigen::VectorXf::Zero(num_dofs);
  m_acc = Eigen::VectorXf::Zero(num_dofs);

  m_process_noise = Eigen::VectorXf::Constant(num_dofs, process_noise);

  // Initialize covariance to large values (high uncertainty)
  m_P.resize(num_dofs);
  for (int i = 0; i < num_dofs; i++)
    m_P[i] = Eigen::Matrix3f::Identity() * 100.0f;
}

void JointObserver::reset(const Eigen::VectorXf& q_init) {
  m_pos = q_init;
  m_vel.setZero();
  m_acc.setZero();
  m_coast_count = 0;
  m_initialized = true;

  for (int i = 0; i < m_num_dofs; i++)
    m_P[i] = Eigen::Matrix3f::Identity() * 10.0f;
}

void JointObserver::set_process_noise(int dof, float noise) {
  m_process_noise[dof] = noise;
}

void JointObserver::predict() {
  float dt = m_dt;
  float dt2 = 0.5f * dt * dt;

  // State transition: constant acceleration model
  // pos += vel * dt + 0.5 * acc * dt^2
  // vel += acc * dt
  // acc = acc (unchanged)
  for (int i = 0; i < m_num_dofs; i++) {
    m_pos[i] += m_vel[i] * dt + m_acc[i] * dt2;
    m_vel[i] += m_acc[i] * dt;
  }

  // Covariance prediction: P = F * P * F^T + Q
  // F = [1   dt   0.5*dt^2]
  //     [0   1    dt       ]
  //     [0   0    1        ]
  Eigen::Matrix3f F;
  F << 1, dt, dt2,
       0, 1, dt,
       0, 0, 1;

  for (int i = 0; i < m_num_dofs; i++) {
    float q = m_process_noise[i];
    Eigen::Matrix3f Q = Eigen::Matrix3f::Identity() * q;
    m_P[i] = F * m_P[i] * F.transpose() + Q;
  }
}

void JointObserver::update(const Eigen::VectorXf& q_measured, bool valid,
                           const Eigen::VectorXf& rom_lower, const Eigen::VectorXf& rom_upper) {
  if (!m_initialized) {
    reset(q_measured);
    return;
  }

  if (!valid) {
    // Coast: no measurement, just use prediction
    m_coast_count++;
    // Clamp to ROM even during coast
    for (int i = 0; i < m_num_dofs; i++)
      m_pos[i] = std::clamp(m_pos[i], rom_lower[i], rom_upper[i]);
    return;
  }

  m_coast_count = 0;

  // Measurement model: H = [1, 0, 0], we observe position only
  // Innovation: y = z - H * x = q_measured - pos
  // S = H * P * H^T + R = P[0][0] + R
  // K = P * H^T * S^{-1} = [P[0][0], P[1][0], P[2][0]]^T / S
  // x = x + K * y
  // P = (I - K * H) * P

  float R = m_measurement_noise;
  Eigen::Vector3f H(1, 0, 0);

  for (int i = 0; i < m_num_dofs; i++) {
    float y = q_measured[i] - m_pos[i]; // innovation
    float S = m_P[i](0, 0) + R;        // innovation covariance

    // Kalman gain
    Eigen::Vector3f K = m_P[i].col(0) / S;

    // State update
    m_pos[i] += K[0] * y;
    m_vel[i] += K[1] * y;
    m_acc[i] += K[2] * y;

    // Covariance update: P = (I - K * H^T) * P
    Eigen::Matrix3f KH = K * H.transpose();
    m_P[i] = (Eigen::Matrix3f::Identity() - KH) * m_P[i];
  }

  // Clamp position to ROM bounds
  for (int i = 0; i < m_num_dofs; i++)
    m_pos[i] = std::clamp(m_pos[i], rom_lower[i], rom_upper[i]);
}

