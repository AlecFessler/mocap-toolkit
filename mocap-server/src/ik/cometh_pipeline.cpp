#include "ik/cometh_pipeline.hpp"
#include <cmath>
#include <cstring>

ComethPipeline::ComethPipeline(const ComethParams& params)
    : m_params(params), m_first_frame(true) {
  m_scaler = std::make_unique<BodyScaler>(m_skeleton, params.scaler_buffer_size,
                                          params.scaler_outlier_tolerance);
  m_solver = std::make_unique<QPIKSolver>(m_skeleton, params.qpik);
  m_observer = std::make_unique<JointObserver>(NUM_DOFS, params.qpik.dt,
                                               params.observer_process_noise,
                                               params.observer_measurement_noise);

  // Set per-DOF process noise
  for (int j = 0; j < NUM_JOINTS; j++) {
    const JointDef& jd = m_skeleton.joints[j];
    float noise;
    if (j == JOINT_ROOT_TRANS)
      noise = params.root_process_noise;
    else if (j >= JOINT_L_THUMB_CMC)
      noise = params.finger_process_noise;
    else
      noise = params.body_process_noise;

    for (int d = 0; d < jd.num_dofs; d++)
      m_observer->set_process_noise(jd.dof_offset + d, noise);
  }
}

void ComethPipeline::extract_targets(const float* kp3d,
                                      const std::vector<bool>& outlier_flags,
                                      std::vector<Eigen::Vector3f>& targets,
                                      std::vector<bool>& valid) const {
  int num_sites = m_skeleton.num_sites();
  targets.resize(num_sites);
  valid.resize(num_sites);

  for (int s = 0; s < num_sites; s++) {
    int coco = m_skeleton.sites[s].coco_index;
    float x = kp3d[coco * 3 + 0];
    float y = kp3d[coco * 3 + 1];
    float z = kp3d[coco * 3 + 2];

    if (std::isnan(x) || outlier_flags[coco]) {
      valid[s] = false;
      targets[s] = Eigen::Vector3f::Zero();
    } else {
      valid[s] = true;
      targets[s] = Eigen::Vector3f(x, y, z);
    }
  }
}

void ComethPipeline::write_output(const float* input_kp3d, float* output_kp3d) const {
  // Start by copying all input (passthrough for head/face)
  std::memcpy(output_kp3d, input_kp3d, 133 * 3 * sizeof(float));

  // Overwrite IK-controlled keypoints with FK output
  for (int s = 0; s < m_skeleton.num_sites(); s++) {
    int coco = m_skeleton.sites[s].coco_index;
    const Eigen::Vector3f& pos = m_skeleton.site_positions[s];
    output_kp3d[coco * 3 + 0] = pos.x();
    output_kp3d[coco * 3 + 1] = pos.y();
    output_kp3d[coco * 3 + 2] = pos.z();
  }
}

// Helper: get a keypoint as Vector3f, check validity
static bool get_kp(const float* kp3d, int idx, Eigen::Vector3f& out) {
  float x = kp3d[idx * 3], y = kp3d[idx * 3 + 1], z = kp3d[idx * 3 + 2];
  if (std::isnan(x)) return false;
  out = Eigen::Vector3f(x, y, z);
  return true;
}

// Compute root translation and rotation from observed body keypoints.
// Sets q[0..5] directly from the data every frame — the IK only solves joint angles.
static void anchor_root(const float* kp3d, Eigen::VectorXf& q) {
  Eigen::Vector3f l_hip, r_hip, l_shoulder, r_shoulder;
  bool have_hips = get_kp(kp3d, 11, l_hip) && get_kp(kp3d, 12, r_hip);
  bool have_shoulders = get_kp(kp3d, 5, l_shoulder) && get_kp(kp3d, 6, r_shoulder);

  if (have_hips) {
    // Root translation = hip midpoint
    Eigen::Vector3f mid = (l_hip + r_hip) / 2.0f;
    q[0] = mid.x();
    q[1] = mid.y();
    q[2] = mid.z();
  }

  if (have_hips && have_shoulders) {
    // Build body frame from observed landmarks
    Eigen::Vector3f hip_right = (r_hip - l_hip).normalized();
    Eigen::Vector3f hip_mid = (l_hip + r_hip) / 2.0f;
    Eigen::Vector3f shoulder_mid = (l_shoulder + r_shoulder) / 2.0f;
    Eigen::Vector3f up = (shoulder_mid - hip_mid).normalized();
    Eigen::Vector3f fwd = hip_right.cross(up).normalized();
    up = fwd.cross(hip_right).normalized();

    // Build rotation matrix: columns are the body frame axes
    Eigen::Matrix3f R;
    R.col(0) = hip_right;
    R.col(1) = up;
    R.col(2) = fwd;

    // Extract ZYX Euler angles
    float pitch = std::asin(-std::clamp(R(2, 0), -1.0f, 1.0f));
    float roll, yaw;
    if (std::abs(R(2, 0)) < 0.999f) {
      roll = std::atan2(R(2, 1), R(2, 2));
      yaw = std::atan2(R(1, 0), R(0, 0));
    } else {
      roll = std::atan2(-R(1, 2), R(1, 1));
      yaw = 0.0f;
    }
    q[3] = roll;
    q[4] = pitch;
    q[5] = yaw;
  }
}

void ComethPipeline::process(const float* input_kp3d, float* output_kp3d) {
  // Step 1: Body scaling — measure bones from DLT, detect outliers
  std::vector<bool> outlier_flags = m_scaler->update(input_kp3d);

  // Step 2: Extract IK targets from input keypoints
  std::vector<Eigen::Vector3f> targets;
  std::vector<bool> valid;
  extract_targets(input_kp3d, outlier_flags, targets, valid);

  // Check if we have enough valid targets to run IK
  int valid_count = 0;
  for (bool v : valid)
    if (v) valid_count++;

  bool has_targets = valid_count >= 4;

  // Step 3: Predict observer state
  m_observer->predict();

  if (has_targets) {
    // Start from observer's predicted joint angles (DOFs 6+)
    m_skeleton.q = m_observer->position();

    // Anchor root position and rotation directly from observed keypoints every frame.
    // Root DOFs (0-5) bypass the observer entirely — set from data, not filtered.
    anchor_root(input_kp3d, m_skeleton.q);

    // First frame: reset observer to the anchored state
    if (m_first_frame) {
      m_observer->reset(m_skeleton.q);
      m_first_frame = false;
    }

    // Step 4: QPIK solve (joint angles only, root is anchored)
    Eigen::VectorXf prev_vel = m_observer->velocity();
    m_solver->solve(targets, valid, prev_vel);

    // Step 5: Update observer with IK result (joint angles)
    m_observer->update(m_skeleton.q, true,
                       m_skeleton.rom_lower(), m_skeleton.rom_upper());
  } else {
    // No valid targets — coast
    m_observer->update(m_skeleton.q, false,
                       m_skeleton.rom_lower(), m_skeleton.rom_upper());

    if (m_observer->coast_count() > m_params.max_coast_frames)
      m_first_frame = true;
  }

  // Step 6: Set skeleton to filtered state and compute final FK
  m_skeleton.q = m_observer->position();
  m_skeleton.compute_fk();

  // Step 7: Write output
  write_output(input_kp3d, output_kp3d);
}
