#ifndef IK_BODY_SCALER_HPP
#define IK_BODY_SCALER_HPP

#include "ik/skeleton.hpp"
#include <vector>
#include <array>


class BodyScaler {
public:
  // buffer_size: number of frames for running median (e.g., 90 = 3s at 30fps)
  // outlier_tolerance: reject keypoints producing bone lengths outside this factor of expected (e.g., 0.3 = +-30%)
  BodyScaler(Skeleton& skeleton, int buffer_size = 90, float outlier_tolerance = 0.3f);

  // Feed a new frame of 3D keypoints from DLT+Kalman output.
  // kp3d: float[133*3], NaN for invalid.
  // Returns a bitmask-style vector: outlier_flags[coco_idx] = true if that keypoint is an outlier
  // (incompatible bone lengths given current skeleton scaling)
  std::vector<bool> update(const float* kp3d);

  // Has the scaler converged (enough frames accumulated)?
  bool converged() const { return m_frame_count >= m_buffer_size / 2; }

private:
  Skeleton& m_skeleton;
  int m_buffer_size;
  float m_outlier_tolerance;
  int m_frame_count;

  // Per-bone circular buffers of length measurements
  std::vector<std::vector<float>> m_bone_buffers;
  std::vector<int> m_bone_write_idx;

  // Compute median of a circular buffer (only filled entries)
  float median(int bone_idx) const;

  // Apply symmetry and update skeleton bone lengths
  void apply_scaling();

  // Check if a keypoint produces an outlier bone length
  bool is_outlier(int bone_idx, float measured_length) const;
};

#endif // IK_BODY_SCALER_HPP
