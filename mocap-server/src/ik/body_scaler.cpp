#include "ik/body_scaler.hpp"
#include <algorithm>
#include <cmath>


BodyScaler::BodyScaler(Skeleton& skeleton, int buffer_size, float outlier_tolerance)
    : m_skeleton(skeleton),
      m_buffer_size(buffer_size),
      m_outlier_tolerance(outlier_tolerance),
      m_frame_count(0) {
  int num_bones = static_cast<int>(skeleton.bones.size());
  m_bone_buffers.resize(num_bones);
  m_bone_write_idx.resize(num_bones, 0);
  for (int i = 0; i < num_bones; i++)
    m_bone_buffers[i].resize(buffer_size, 0.0f);
}

float BodyScaler::median(int bone_idx) const {
  const auto& buf = m_bone_buffers[bone_idx];
  int count = std::min(m_frame_count, m_buffer_size);
  if (count == 0) return 0.0f;

  std::vector<float> valid;
  valid.reserve(count);
  for (int i = 0; i < count; i++) {
    if (buf[i] > 0.0f)
      valid.push_back(buf[i]);
  }
  if (valid.empty()) return 0.0f;

  std::sort(valid.begin(), valid.end());
  int n = static_cast<int>(valid.size());
  if (n % 2 == 0)
    return (valid[n / 2 - 1] + valid[n / 2]) / 2.0f;
  else
    return valid[n / 2];
}

bool BodyScaler::is_outlier(int bone_idx, float measured_length) const {
  const BoneDef& bd = m_skeleton.bones[bone_idx];
  float expected = m_skeleton.joints[bd.joint].bone_length;
  if (expected <= 0.0f) return false; // no reference yet

  // Width bones measure full inter-joint distance but joint stores half
  if (bd.symmetric_pair == -1 &&
      ((bd.coco_a == 11 && bd.coco_b == 12) ||
       (bd.coco_a == 5 && bd.coco_b == 6)))
    expected *= 2.0f;

  float ratio = measured_length / expected;
  return std::abs(ratio - 1.0f) > m_outlier_tolerance;
}

std::vector<bool> BodyScaler::update(const float* kp3d) {
  int num_bones = static_cast<int>(m_skeleton.bones.size());
  std::vector<bool> outlier_flags(133, false);

  // Measure bone lengths from DLT 3D points
  for (int b = 0; b < num_bones; b++) {
    const BoneDef& bd = m_skeleton.bones[b];
    int a3 = bd.coco_a * 3;
    int b3 = bd.coco_b * 3;

    float ax = kp3d[a3], ay = kp3d[a3 + 1], az = kp3d[a3 + 2];
    float bx = kp3d[b3], by = kp3d[b3 + 1], bz = kp3d[b3 + 2];

    // Skip if either endpoint is NaN
    if (std::isnan(ax) || std::isnan(bx))
      continue;

    float dx = ax - bx, dy = ay - by, dz = az - bz;
    float length = std::sqrt(dx * dx + dy * dy + dz * dz);

    // Skip degenerate measurements
    if (length < 1.0f) continue;

    // Check for outlier if we have enough history
    if (m_frame_count >= 10 && is_outlier(b, length)) {
      // Mark both endpoints as outliers
      outlier_flags[bd.coco_a] = true;
      outlier_flags[bd.coco_b] = true;
      continue;
    }

    // Store in circular buffer
    int idx = m_bone_write_idx[b] % m_buffer_size;
    m_bone_buffers[b][idx] = length;
    m_bone_write_idx[b]++;
  }

  m_frame_count++;
  apply_scaling();

  return outlier_flags;
}

void BodyScaler::apply_scaling() {
  int num_bones = static_cast<int>(m_skeleton.bones.size());

  // Compute median per bone
  std::vector<float> medians(num_bones);
  for (int b = 0; b < num_bones; b++)
    medians[b] = median(b);

  // Enforce L/R symmetry: average symmetric pairs
  for (int b = 0; b < num_bones; b++) {
    int pair = m_skeleton.bones[b].symmetric_pair;
    if (pair < 0 || pair <= b) continue; // process each pair once

    float left = medians[b];
    float right = medians[pair];

    if (left > 0.0f && right > 0.0f) {
      float avg = (left + right) / 2.0f;
      medians[b] = avg;
      medians[pair] = avg;
    } else if (left > 0.0f) {
      medians[pair] = left;
    } else if (right > 0.0f) {
      medians[b] = right;
    }
  }

  // Special handling for hip width and shoulder width:
  // These are full-width measurements but the skeleton stores half-width as bone_length
  // Hip width bone (idx 8): COCO 11-12 distance = full hip width, joint stores half
  // Shoulder width bone (idx 9): COCO 5-6 distance = full shoulder width, joint stores half

  // Apply to skeleton bone lengths
  for (int b = 0; b < num_bones; b++) {
    float m = medians[b];
    if (m <= 0.0f) continue;

    int joint_idx = m_skeleton.bones[b].joint;

    // For hip/shoulder width bones: the measured distance is the full width,
    // but each side's joint bone_length is half (distance from pelvis center to hip)
    // We identify these by checking if both coco endpoints map to symmetric joints
    const BoneDef& bd = m_skeleton.bones[b];
    if (bd.symmetric_pair == -1 &&
        (bd.coco_a == 11 && bd.coco_b == 12)) {
      // Hip width: set both L and R hip bone_length to half
      m_skeleton.joints[JOINT_L_HIP].bone_length = m / 2.0f;
      m_skeleton.joints[JOINT_R_HIP].bone_length = m / 2.0f;
      continue;
    }
    if (bd.symmetric_pair == -1 &&
        (bd.coco_a == 5 && bd.coco_b == 6)) {
      // Shoulder width: set both L and R shoulder bone_length to half
      m_skeleton.joints[JOINT_L_SHOULDER].bone_length = m / 2.0f;
      m_skeleton.joints[JOINT_R_SHOULDER].bone_length = m / 2.0f;
      continue;
    }

    m_skeleton.joints[joint_idx].bone_length = m;
  }
}

