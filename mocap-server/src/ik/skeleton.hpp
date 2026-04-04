#ifndef IK_SKELETON_HPP
#define IK_SKELETON_HPP

#include <Eigen/Dense>
#include <array>
#include <vector>
#include <cmath>

// Skeleton joint indices
enum Joint : int {
  JOINT_ROOT_TRANS = 0,
  JOINT_ROOT_ROT,
  JOINT_L_HIP,
  JOINT_L_KNEE,
  JOINT_L_ANKLE,
  JOINT_L_TOE,
  JOINT_R_HIP,
  JOINT_R_KNEE,
  JOINT_R_ANKLE,
  JOINT_R_TOE,
  JOINT_SPINE,
  JOINT_L_SHOULDER,
  JOINT_L_ELBOW,
  JOINT_L_WRIST,
  JOINT_R_SHOULDER,
  JOINT_R_ELBOW,
  JOINT_R_WRIST,
  // Left hand
  JOINT_L_THUMB_CMC,
  JOINT_L_THUMB_MCP,
  JOINT_L_THUMB_IP,
  JOINT_L_INDEX_MCP,
  JOINT_L_INDEX_PIP,
  JOINT_L_INDEX_DIP,
  JOINT_L_MIDDLE_MCP,
  JOINT_L_MIDDLE_PIP,
  JOINT_L_MIDDLE_DIP,
  JOINT_L_RING_MCP,
  JOINT_L_RING_PIP,
  JOINT_L_RING_DIP,
  JOINT_L_PINKY_MCP,
  JOINT_L_PINKY_PIP,
  JOINT_L_PINKY_DIP,
  // Right hand
  JOINT_R_THUMB_CMC,
  JOINT_R_THUMB_MCP,
  JOINT_R_THUMB_IP,
  JOINT_R_INDEX_MCP,
  JOINT_R_INDEX_PIP,
  JOINT_R_INDEX_DIP,
  JOINT_R_MIDDLE_MCP,
  JOINT_R_MIDDLE_PIP,
  JOINT_R_MIDDLE_DIP,
  JOINT_R_RING_MCP,
  JOINT_R_RING_PIP,
  JOINT_R_RING_DIP,
  JOINT_R_PINKY_MCP,
  JOINT_R_PINKY_PIP,
  JOINT_R_PINKY_DIP,
  NUM_JOINTS
};

static constexpr int NUM_DOFS = 75;

// Per-joint definition
struct JointDef {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int parent;             // parent joint index, -1 for root
  int dof_offset;         // start index in the q vector
  int num_dofs;           // 1, 2, or 3
  Eigen::Vector3f bone_dir;     // unit direction of bone from parent (in parent local frame)
  float bone_length;            // length of bone from parent (0 for root)
  Eigen::Vector3f axes[3];      // rotation axes for each DOF (in local frame)
  float rom_lower[3];           // ROM lower bounds (radians)
  float rom_upper[3];           // ROM upper bounds (radians)
};

// Maps a skeleton site to a COCO keypoint
struct SiteDef {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int joint;              // skeleton joint this site is attached to
  Eigen::Vector3f local_offset; // offset in joint's local frame (for tips, heel, etc.)
  int coco_index;         // COCO WholeBody keypoint index
};

// Defines a bone for length measurement (between two COCO keypoints)
struct BoneDef {
  int coco_a;             // first COCO keypoint
  int coco_b;             // second COCO keypoint
  int joint;              // skeleton joint whose bone_length this sets
  int symmetric_pair;     // index of the L/R counterpart in the bone table (-1 if none)
};

class Skeleton {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Skeleton();

  // Joint definitions (fixed after construction, except bone_length which the scaler updates)
  std::array<JointDef, NUM_JOINTS> joints;

  // IK target sites
  std::vector<SiteDef, Eigen::aligned_allocator<SiteDef>> sites;

  // Bone definitions for the scaler
  std::vector<BoneDef> bones;

  // Current state
  Eigen::VectorXf q;  // [NUM_DOFS] joint angles

  // FK output (updated by compute_fk)
  std::array<Eigen::Matrix4f, NUM_JOINTS> world_transforms;
  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> site_positions; // [sites.size()]

  // Per-DOF world axes (computed during FK, used by Jacobian)
  // Each DOF's rotation axis transformed to world frame at the correct intermediate rotation state
  Eigen::Vector3f dof_world_axes[NUM_DOFS];

  // Compute forward kinematics from current q
  void compute_fk();

  // Compute analytical Jacobian: J is [3*num_sites x NUM_DOFS]
  // Only computes columns for DOFs that are ancestors of each site
  void compute_jacobian(Eigen::MatrixXf& J) const;

  // Get world position of a joint (from cached world_transforms)
  Eigen::Vector3f joint_world_pos(int joint_idx) const;

  // Get world rotation axis for a specific DOF
  Eigen::Vector3f dof_world_axis(int joint_idx, int local_dof) const;

  // ROM bounds as flat vectors for the QP solver
  Eigen::VectorXf rom_lower() const;
  Eigen::VectorXf rom_upper() const;

  int num_sites() const { return static_cast<int>(sites.size()); }

private:
  void init_body_joints();
  void init_hand_joints();
  void init_sites();
  void init_bones();

  // Precomputed ancestor DOF list per site for fast Jacobian
  // ancestors[site_idx] = list of (joint_idx, local_dof_idx) pairs
  std::vector<std::vector<std::pair<int, int>>> m_site_ancestors;
  void build_ancestor_map();
};

// Degree to radian helper
inline constexpr float deg2rad(float d) { return d * static_cast<float>(M_PI) / 180.0f; }

#endif // IK_SKELETON_HPP
