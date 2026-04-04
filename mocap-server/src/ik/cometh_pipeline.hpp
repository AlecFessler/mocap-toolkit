#ifndef IK_COMETH_PIPELINE_HPP
#define IK_COMETH_PIPELINE_HPP

#include "ik/skeleton.hpp"
#include "ik/body_scaler.hpp"
#include "ik/qpik_solver.hpp"
#include "ik/joint_observer.hpp"
#include <memory>

struct ComethParams {
  QPIKParams qpik;
  float observer_process_noise = 0.5f;
  float observer_measurement_noise = 1.0f;
  int scaler_buffer_size = 90;
  float scaler_outlier_tolerance = 0.3f;

  // Process noise by joint type (body > fingers since body moves more)
  float body_process_noise = 0.5f;
  float finger_process_noise = 0.1f;
  float root_process_noise = 1.0f;

  // Max coast frames before resetting to neutral
  int max_coast_frames = 30;
};

class ComethPipeline {
public:
  ComethPipeline(const ComethParams& params = ComethParams{});

  // Process one frame of 3D keypoints.
  // input_kp3d: float[133*3] from DLT+Kalman, NaN for invalid
  // output_kp3d: float[133*3] with IK-corrected body/hands/feet + passthrough head/face
  void process(const float* input_kp3d, float* output_kp3d);

private:
  ComethParams m_params;
  Skeleton m_skeleton;
  std::unique_ptr<BodyScaler> m_scaler;
  std::unique_ptr<QPIKSolver> m_solver;
  std::unique_ptr<JointObserver> m_observer;

  bool m_first_frame;

  // Extract IK targets from COCO keypoints, populating per-site targets
  void extract_targets(const float* kp3d,
                       const std::vector<bool>& outlier_flags,
                       std::vector<Eigen::Vector3f>& targets,
                       std::vector<bool>& valid) const;

  // Write FK output back to COCO format
  void write_output(const float* input_kp3d, float* output_kp3d) const;
};

#endif // IK_COMETH_PIPELINE_HPP
