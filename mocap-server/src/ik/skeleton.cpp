#include "ik/skeleton.hpp"
#include <cmath>


// Axis constants
static const Eigen::Vector3f AXIS_X(1, 0, 0);
static const Eigen::Vector3f AXIS_Y(0, 1, 0);
static const Eigen::Vector3f AXIS_Z(0, 0, 1);

// Helper: build a JointDef
static JointDef make_joint(int parent, int dof_offset, int num_dofs,
                           Eigen::Vector3f bone_dir, float bone_length,
                           const Eigen::Vector3f axes[],
                           const float rom_lo[], const float rom_hi[]) {
  JointDef j{};
  j.parent = parent;
  j.dof_offset = dof_offset;
  j.num_dofs = num_dofs;
  j.bone_dir = bone_dir;
  j.bone_length = bone_length;
  for (int i = 0; i < num_dofs; i++) {
    j.axes[i] = axes[i];
    j.rom_lower[i] = rom_lo[i];
    j.rom_upper[i] = rom_hi[i];
  }
  for (int i = num_dofs; i < 3; i++) {
    j.axes[i] = AXIS_X;
    j.rom_lower[i] = 0;
    j.rom_upper[i] = 0;
  }
  return j;
}

void Skeleton::init_body_joints() {
  int dof = 0;

  // Root translation (3 DOF) — no bone, no ROM limits
  {
    Eigen::Vector3f axes[] = {AXIS_X, AXIS_Y, AXIS_Z};
    float lo[] = {-1e6f, -1e6f, -1e6f};
    float hi[] = {1e6f, 1e6f, 1e6f};
    joints[JOINT_ROOT_TRANS] = make_joint(-1, dof, 3,
      Eigen::Vector3f::Zero(), 0.0f, axes, lo, hi);
    dof += 3;
  }

  // Root rotation (3 DOF, ZYX Euler) — no bone from trans to rot, unlimited
  {
    Eigen::Vector3f axes[] = {AXIS_X, AXIS_Y, AXIS_Z};
    float lo[] = {-static_cast<float>(M_PI), -static_cast<float>(M_PI), -static_cast<float>(M_PI)};
    float hi[] = {static_cast<float>(M_PI), static_cast<float>(M_PI), static_cast<float>(M_PI)};
    joints[JOINT_ROOT_ROT] = make_joint(JOINT_ROOT_TRANS, dof, 3,
      Eigen::Vector3f::Zero(), 0.0f, axes, lo, hi);
    dof += 3;
  }

  // Left hip (3 DOF) — bone from pelvis center to left hip
  {
    Eigen::Vector3f axes[] = {AXIS_X, AXIS_Z, AXIS_Y};  // flex/ext, abd/add, int/ext rot
    float lo[] = {deg2rad(-30), deg2rad(-45), deg2rad(-45)};
    float hi[] = {deg2rad(120), deg2rad(45), deg2rad(45)};
    joints[JOINT_L_HIP] = make_joint(JOINT_ROOT_ROT, dof, 3,
      Eigen::Vector3f(-1, 0, 0).normalized(), 100.0f, axes, lo, hi); // default ~100mm half hip width
    dof += 3;
  }

  // Left knee (1 DOF)
  {
    Eigen::Vector3f axes[] = {AXIS_X};
    float lo[] = {deg2rad(0)};
    float hi[] = {deg2rad(150)};
    joints[JOINT_L_KNEE] = make_joint(JOINT_L_HIP, dof, 1,
      Eigen::Vector3f(0, -1, 0).normalized(), 430.0f, axes, lo, hi); // thigh ~430mm
    dof += 1;
  }

  // Left ankle (2 DOF)
  {
    Eigen::Vector3f axes[] = {AXIS_X, AXIS_Z};  // dorsi/plantar, inv/eversion
    float lo[] = {deg2rad(-50), deg2rad(-25)};
    float hi[] = {deg2rad(30), deg2rad(25)};
    joints[JOINT_L_ANKLE] = make_joint(JOINT_L_KNEE, dof, 2,
      Eigen::Vector3f(0, -1, 0).normalized(), 410.0f, axes, lo, hi); // shin ~410mm
    dof += 2;
  }

  // Left toe (1 DOF)
  {
    Eigen::Vector3f axes[] = {AXIS_X};
    float lo[] = {deg2rad(-30)};
    float hi[] = {deg2rad(45)};
    joints[JOINT_L_TOE] = make_joint(JOINT_L_ANKLE, dof, 1,
      Eigen::Vector3f(0, 0, 1).normalized(), 150.0f, axes, lo, hi); // foot ~150mm
    dof += 1;
  }

  // Right hip (3 DOF)
  {
    Eigen::Vector3f axes[] = {AXIS_X, AXIS_Z, AXIS_Y};
    float lo[] = {deg2rad(-30), deg2rad(-45), deg2rad(-45)};
    float hi[] = {deg2rad(120), deg2rad(45), deg2rad(45)};
    joints[JOINT_R_HIP] = make_joint(JOINT_ROOT_ROT, dof, 3,
      Eigen::Vector3f(1, 0, 0).normalized(), 100.0f, axes, lo, hi);
    dof += 3;
  }

  // Right knee (1 DOF)
  {
    Eigen::Vector3f axes[] = {AXIS_X};
    float lo[] = {deg2rad(0)};
    float hi[] = {deg2rad(150)};
    joints[JOINT_R_KNEE] = make_joint(JOINT_R_HIP, dof, 1,
      Eigen::Vector3f(0, -1, 0).normalized(), 430.0f, axes, lo, hi);
    dof += 1;
  }

  // Right ankle (2 DOF)
  {
    Eigen::Vector3f axes[] = {AXIS_X, AXIS_Z};
    float lo[] = {deg2rad(-50), deg2rad(-25)};
    float hi[] = {deg2rad(30), deg2rad(25)};
    joints[JOINT_R_ANKLE] = make_joint(JOINT_R_KNEE, dof, 2,
      Eigen::Vector3f(0, -1, 0).normalized(), 410.0f, axes, lo, hi);
    dof += 2;
  }

  // Right toe (1 DOF)
  {
    Eigen::Vector3f axes[] = {AXIS_X};
    float lo[] = {deg2rad(-30)};
    float hi[] = {deg2rad(45)};
    joints[JOINT_R_TOE] = make_joint(JOINT_R_ANKLE, dof, 1,
      Eigen::Vector3f(0, 0, 1).normalized(), 150.0f, axes, lo, hi);
    dof += 1;
  }

  // Spine (3 DOF) — bone from pelvis up to mid-torso/shoulder level
  {
    Eigen::Vector3f axes[] = {AXIS_X, AXIS_Z, AXIS_Y};  // flex/ext, lat bend, axial rot
    float lo[] = {deg2rad(-30), deg2rad(-30), deg2rad(-45)};
    float hi[] = {deg2rad(45), deg2rad(30), deg2rad(45)};
    joints[JOINT_SPINE] = make_joint(JOINT_ROOT_ROT, dof, 3,
      Eigen::Vector3f(0, 1, 0).normalized(), 450.0f, axes, lo, hi); // torso ~450mm
    dof += 3;
  }

  // Left shoulder (3 DOF) — bone from spine to left shoulder
  {
    Eigen::Vector3f axes[] = {AXIS_X, AXIS_Z, AXIS_Y};  // flex/ext, abd/add, int/ext rot
    float lo[] = {deg2rad(-60), deg2rad(-10), deg2rad(-90)};
    float hi[] = {deg2rad(180), deg2rad(180), deg2rad(90)};
    joints[JOINT_L_SHOULDER] = make_joint(JOINT_SPINE, dof, 3,
      Eigen::Vector3f(-1, 0, 0).normalized(), 190.0f, axes, lo, hi); // half shoulder width ~190mm
    dof += 3;
  }

  // Left elbow (1 DOF)
  {
    Eigen::Vector3f axes[] = {AXIS_X};
    float lo[] = {deg2rad(0)};
    float hi[] = {deg2rad(150)};
    joints[JOINT_L_ELBOW] = make_joint(JOINT_L_SHOULDER, dof, 1,
      Eigen::Vector3f(0, -1, 0).normalized(), 280.0f, axes, lo, hi); // upper arm ~280mm
    dof += 1;
  }

  // Left wrist (2 DOF)
  {
    Eigen::Vector3f axes[] = {AXIS_X, AXIS_Z};  // flex/ext, rad/ulnar dev
    float lo[] = {deg2rad(-80), deg2rad(-30)};
    float hi[] = {deg2rad(80), deg2rad(40)};
    joints[JOINT_L_WRIST] = make_joint(JOINT_L_ELBOW, dof, 2,
      Eigen::Vector3f(0, -1, 0).normalized(), 260.0f, axes, lo, hi); // forearm ~260mm
    dof += 2;
  }

  // Right shoulder (3 DOF)
  {
    Eigen::Vector3f axes[] = {AXIS_X, AXIS_Z, AXIS_Y};
    float lo[] = {deg2rad(-60), deg2rad(-10), deg2rad(-90)};
    float hi[] = {deg2rad(180), deg2rad(180), deg2rad(90)};
    joints[JOINT_R_SHOULDER] = make_joint(JOINT_SPINE, dof, 3,
      Eigen::Vector3f(1, 0, 0).normalized(), 190.0f, axes, lo, hi);
    dof += 3;
  }

  // Right elbow (1 DOF)
  {
    Eigen::Vector3f axes[] = {AXIS_X};
    float lo[] = {deg2rad(0)};
    float hi[] = {deg2rad(150)};
    joints[JOINT_R_ELBOW] = make_joint(JOINT_R_SHOULDER, dof, 1,
      Eigen::Vector3f(0, -1, 0).normalized(), 280.0f, axes, lo, hi);
    dof += 1;
  }

  // Right wrist (2 DOF)
  {
    Eigen::Vector3f axes[] = {AXIS_X, AXIS_Z};
    float lo[] = {deg2rad(-80), deg2rad(-30)};
    float hi[] = {deg2rad(80), deg2rad(40)};
    joints[JOINT_R_WRIST] = make_joint(JOINT_R_ELBOW, dof, 2,
      Eigen::Vector3f(0, -1, 0).normalized(), 260.0f, axes, lo, hi);
    dof += 2;
  }

  // dof should now be 35
  (void)dof;
}

// Helper to init a set of 5 finger joints (thumb + 4 fingers) for one hand
// Returns the next dof offset after all fingers
static int init_fingers(std::array<JointDef, NUM_JOINTS>& joints,
                        int wrist_joint, int first_joint_id, int dof_start,
                        float side) {
  (void)side; // could be used for mirroring bone directions if needed
  int dof = dof_start;

  // Finger layout per hand (from wrist):
  // Thumb: CMC(2) -> MCP(1) -> IP(1) = joints [0..2], 4 DOF
  // Index: MCP(2) -> PIP(1) -> DIP(1) = joints [3..5], 4 DOF
  // Middle: MCP(2) -> PIP(1) -> DIP(1) = joints [6..8], 4 DOF
  // Ring: MCP(2) -> PIP(1) -> DIP(1) = joints [9..11], 4 DOF
  // Pinky: MCP(2) -> PIP(1) -> DIP(1) = joints [12..14], 4 DOF

  // Default finger bone lengths (mm) — proximal, middle, distal phalanges
  // Thumb
  constexpr float THUMB_MC = 45.0f;   // metacarpal
  constexpr float THUMB_PP = 30.0f;   // proximal phalanx
  constexpr float THUMB_DP = 25.0f;   // distal phalanx

  // Index/middle/ring/pinky
  constexpr float FINGER_MC = 65.0f;  // metacarpal (to MCP)
  constexpr float FINGER_PP = 40.0f;  // proximal phalanx
  constexpr float FINGER_MP = 25.0f;  // middle phalanx
  constexpr float FINGER_DP = 18.0f;  // distal phalanx

  // Finger splay angles from wrist center
  // Thumb points ~45 deg lateral from palm, other fingers fan out in Z
  Eigen::Vector3f thumb_dir = Eigen::Vector3f(side * 0.7f, -0.3f, 0.6f).normalized();
  Eigen::Vector3f index_dir = Eigen::Vector3f(side * 0.15f, -0.15f, 0.97f).normalized();
  Eigen::Vector3f middle_dir = Eigen::Vector3f(0.0f, -0.1f, 0.99f).normalized();
  Eigen::Vector3f ring_dir = Eigen::Vector3f(-side * 0.15f, -0.15f, 0.97f).normalized();
  Eigen::Vector3f pinky_dir = Eigen::Vector3f(-side * 0.3f, -0.2f, 0.93f).normalized();

  // Along-finger direction (distal segments continue roughly same direction)
  Eigen::Vector3f fwd = Eigen::Vector3f(0, 0, 1); // simplified: all fingers extend in Z in local wrist frame

  // --- Thumb ---
  int j = first_joint_id;
  // CMC (2 DOF: flex/ext + abd/add)
  {
    Eigen::Vector3f axes[] = {AXIS_X, AXIS_Z};
    float lo[] = {deg2rad(-10), deg2rad(-10)};
    float hi[] = {deg2rad(60), deg2rad(50)};
    joints[j] = make_joint(wrist_joint, dof, 2, thumb_dir, THUMB_MC, axes, lo, hi);
    dof += 2;
  }
  // MCP (1 DOF: flex)
  {
    Eigen::Vector3f axes[] = {AXIS_X};
    float lo[] = {deg2rad(-10)};
    float hi[] = {deg2rad(80)};
    joints[j + 1] = make_joint(j, dof, 1, fwd, THUMB_PP, axes, lo, hi);
    dof += 1;
  }
  // IP (1 DOF: flex)
  {
    Eigen::Vector3f axes[] = {AXIS_X};
    float lo[] = {deg2rad(-10)};
    float hi[] = {deg2rad(80)};
    joints[j + 2] = make_joint(j + 1, dof, 1, fwd, THUMB_DP, axes, lo, hi);
    dof += 1;
  }

  // --- Four fingers (index, middle, ring, pinky) ---
  Eigen::Vector3f finger_dirs[] = {index_dir, middle_dir, ring_dir, pinky_dir};
  // metacarpal lengths vary slightly per finger
  float mc_lengths[] = {FINGER_MC, FINGER_MC * 1.05f, FINGER_MC * 0.95f, FINGER_MC * 0.85f};
  float pp_lengths[] = {FINGER_PP, FINGER_PP * 1.05f, FINGER_PP * 0.95f, FINGER_PP * 0.85f};
  float mp_lengths[] = {FINGER_MP, FINGER_MP * 1.0f, FINGER_MP * 0.95f, FINGER_MP * 0.85f};
  (void)FINGER_DP; // fingertip lengths handled via site offsets, not separate bones

  for (int f = 0; f < 4; f++) {
    int base = first_joint_id + 3 + f * 3; // 3 joints per finger, thumb took slots 0-2
    // MCP (2 DOF: flex/ext + abd/add)
    {
      Eigen::Vector3f axes[] = {AXIS_X, AXIS_Z};
      float lo[] = {deg2rad(-20), deg2rad(-15)};
      float hi[] = {deg2rad(90), deg2rad(15)};
      joints[base] = make_joint(wrist_joint, dof, 2,
        finger_dirs[f], mc_lengths[f], axes, lo, hi);
      dof += 2;
    }
    // PIP (1 DOF: flex)
    {
      Eigen::Vector3f axes[] = {AXIS_X};
      float lo[] = {deg2rad(0)};
      float hi[] = {deg2rad(110)};
      joints[base + 1] = make_joint(base, dof, 1, fwd, pp_lengths[f], axes, lo, hi);
      dof += 1;
    }
    // DIP (1 DOF: flex)
    {
      Eigen::Vector3f axes[] = {AXIS_X};
      float lo[] = {deg2rad(0)};
      float hi[] = {deg2rad(80)};
      joints[base + 2] = make_joint(base + 1, dof, 1, fwd, mp_lengths[f], axes, lo, hi);
      dof += 1;
    }
  }

  return dof;
}

void Skeleton::init_hand_joints() {
  // Left hand: joints JOINT_L_THUMB_CMC through JOINT_L_PINKY_DIP (15 joints)
  // dof_offset starts at 35 (after body)
  int dof_after_left = init_fingers(joints, JOINT_L_WRIST, JOINT_L_THUMB_CMC, 35, -1.0f);

  // Right hand: joints JOINT_R_THUMB_CMC through JOINT_R_PINKY_DIP (15 joints)
  init_fingers(joints, JOINT_R_WRIST, JOINT_R_THUMB_CMC, dof_after_left, 1.0f);
}

void Skeleton::init_sites() {
  sites.clear();

  // Body sites — COCO keypoint index -> skeleton joint
  // Shoulders
  sites.push_back({JOINT_L_SHOULDER, Eigen::Vector3f::Zero(), 5});
  sites.push_back({JOINT_R_SHOULDER, Eigen::Vector3f::Zero(), 6});
  // Elbows
  sites.push_back({JOINT_L_ELBOW, Eigen::Vector3f::Zero(), 7});
  sites.push_back({JOINT_R_ELBOW, Eigen::Vector3f::Zero(), 8});
  // Wrists
  sites.push_back({JOINT_L_WRIST, Eigen::Vector3f::Zero(), 9});
  sites.push_back({JOINT_R_WRIST, Eigen::Vector3f::Zero(), 10});
  // Hips
  sites.push_back({JOINT_L_HIP, Eigen::Vector3f::Zero(), 11});
  sites.push_back({JOINT_R_HIP, Eigen::Vector3f::Zero(), 12});
  // Knees
  sites.push_back({JOINT_L_KNEE, Eigen::Vector3f::Zero(), 13});
  sites.push_back({JOINT_R_KNEE, Eigen::Vector3f::Zero(), 14});
  // Ankles
  sites.push_back({JOINT_L_ANKLE, Eigen::Vector3f::Zero(), 15});
  sites.push_back({JOINT_R_ANKLE, Eigen::Vector3f::Zero(), 16});

  // Feet — toe/heel sites with local offsets from ankle/toe joints
  sites.push_back({JOINT_L_TOE, Eigen::Vector3f(30, 0, 0), 17});   // l_big_toe (lateral offset)
  sites.push_back({JOINT_L_TOE, Eigen::Vector3f(-30, 0, 0), 18});  // l_small_toe
  sites.push_back({JOINT_L_ANKLE, Eigen::Vector3f(0, 0, -50), 19}); // l_heel (behind ankle)
  sites.push_back({JOINT_R_TOE, Eigen::Vector3f(-30, 0, 0), 20});  // r_big_toe
  sites.push_back({JOINT_R_TOE, Eigen::Vector3f(30, 0, 0), 21});   // r_small_toe
  sites.push_back({JOINT_R_ANKLE, Eigen::Vector3f(0, 0, -50), 22}); // r_heel

  // Left hand (COCO 91-111): wrist(91), then 4 keypoints per finger
  // COCO 91 = hand wrist (same position as body wrist, redundant but include for IK target)
  sites.push_back({JOINT_L_WRIST, Eigen::Vector3f::Zero(), 91});

  // Thumb: COCO 92=CMC, 93=MCP, 94=IP, 95=tip
  sites.push_back({JOINT_L_THUMB_CMC, Eigen::Vector3f::Zero(), 92});
  sites.push_back({JOINT_L_THUMB_MCP, Eigen::Vector3f::Zero(), 93});
  sites.push_back({JOINT_L_THUMB_IP, Eigen::Vector3f::Zero(), 94});
  // Tip is end-effector of IP joint (fixed offset along bone direction)
  sites.push_back({JOINT_L_THUMB_IP, Eigen::Vector3f(0, 0, 20), 95});

  // Index: COCO 96=MCP, 97=PIP, 98=DIP, 99=tip
  sites.push_back({JOINT_L_INDEX_MCP, Eigen::Vector3f::Zero(), 96});
  sites.push_back({JOINT_L_INDEX_PIP, Eigen::Vector3f::Zero(), 97});
  sites.push_back({JOINT_L_INDEX_DIP, Eigen::Vector3f::Zero(), 98});
  sites.push_back({JOINT_L_INDEX_DIP, Eigen::Vector3f(0, 0, 18), 99});

  // Middle: COCO 100-103
  sites.push_back({JOINT_L_MIDDLE_MCP, Eigen::Vector3f::Zero(), 100});
  sites.push_back({JOINT_L_MIDDLE_PIP, Eigen::Vector3f::Zero(), 101});
  sites.push_back({JOINT_L_MIDDLE_DIP, Eigen::Vector3f::Zero(), 102});
  sites.push_back({JOINT_L_MIDDLE_DIP, Eigen::Vector3f(0, 0, 18), 103});

  // Ring: COCO 104-107
  sites.push_back({JOINT_L_RING_MCP, Eigen::Vector3f::Zero(), 104});
  sites.push_back({JOINT_L_RING_PIP, Eigen::Vector3f::Zero(), 105});
  sites.push_back({JOINT_L_RING_DIP, Eigen::Vector3f::Zero(), 106});
  sites.push_back({JOINT_L_RING_DIP, Eigen::Vector3f(0, 0, 18), 107});

  // Pinky: COCO 108-111
  sites.push_back({JOINT_L_PINKY_MCP, Eigen::Vector3f::Zero(), 108});
  sites.push_back({JOINT_L_PINKY_PIP, Eigen::Vector3f::Zero(), 109});
  sites.push_back({JOINT_L_PINKY_DIP, Eigen::Vector3f::Zero(), 110});
  sites.push_back({JOINT_L_PINKY_DIP, Eigen::Vector3f(0, 0, 18), 111});

  // Right hand (COCO 112-132): same structure, mirrored
  sites.push_back({JOINT_R_WRIST, Eigen::Vector3f::Zero(), 112});

  // Thumb: COCO 113-116
  sites.push_back({JOINT_R_THUMB_CMC, Eigen::Vector3f::Zero(), 113});
  sites.push_back({JOINT_R_THUMB_MCP, Eigen::Vector3f::Zero(), 114});
  sites.push_back({JOINT_R_THUMB_IP, Eigen::Vector3f::Zero(), 115});
  sites.push_back({JOINT_R_THUMB_IP, Eigen::Vector3f(0, 0, 20), 116});

  // Index: COCO 117-120
  sites.push_back({JOINT_R_INDEX_MCP, Eigen::Vector3f::Zero(), 117});
  sites.push_back({JOINT_R_INDEX_PIP, Eigen::Vector3f::Zero(), 118});
  sites.push_back({JOINT_R_INDEX_DIP, Eigen::Vector3f::Zero(), 119});
  sites.push_back({JOINT_R_INDEX_DIP, Eigen::Vector3f(0, 0, 18), 120});

  // Middle: COCO 121-124
  sites.push_back({JOINT_R_MIDDLE_MCP, Eigen::Vector3f::Zero(), 121});
  sites.push_back({JOINT_R_MIDDLE_PIP, Eigen::Vector3f::Zero(), 122});
  sites.push_back({JOINT_R_MIDDLE_DIP, Eigen::Vector3f::Zero(), 123});
  sites.push_back({JOINT_R_MIDDLE_DIP, Eigen::Vector3f(0, 0, 18), 124});

  // Ring: COCO 125-128
  sites.push_back({JOINT_R_RING_MCP, Eigen::Vector3f::Zero(), 125});
  sites.push_back({JOINT_R_RING_PIP, Eigen::Vector3f::Zero(), 126});
  sites.push_back({JOINT_R_RING_DIP, Eigen::Vector3f::Zero(), 127});
  sites.push_back({JOINT_R_RING_DIP, Eigen::Vector3f(0, 0, 18), 128});

  // Pinky: COCO 129-132
  sites.push_back({JOINT_R_PINKY_MCP, Eigen::Vector3f::Zero(), 129});
  sites.push_back({JOINT_R_PINKY_PIP, Eigen::Vector3f::Zero(), 130});
  sites.push_back({JOINT_R_PINKY_DIP, Eigen::Vector3f::Zero(), 131});
  sites.push_back({JOINT_R_PINKY_DIP, Eigen::Vector3f(0, 0, 18), 132});

  site_positions.resize(sites.size());
}

void Skeleton::init_bones() {
  bones.clear();

  // Body bones (measured between COCO keypoints)
  // Index:  coco_a, coco_b, skeleton_joint (whose bone_length this sets), symmetric_pair_idx

  int idx = 0;
  // Upper arms
  bones.push_back({5, 7, JOINT_L_ELBOW, idx + 1});    // 0: l_shoulder -> l_elbow = l_upper_arm
  bones.push_back({6, 8, JOINT_R_ELBOW, idx});         // 1: r_shoulder -> r_elbow = r_upper_arm
  idx += 2;
  // Forearms
  bones.push_back({7, 9, JOINT_L_WRIST, idx + 1});     // 2: l_elbow -> l_wrist = l_forearm
  bones.push_back({8, 10, JOINT_R_WRIST, idx});         // 3: r_elbow -> r_wrist = r_forearm
  idx += 2;
  // Thighs
  bones.push_back({11, 13, JOINT_L_KNEE, idx + 1});    // 4: l_hip -> l_knee = l_thigh
  bones.push_back({12, 14, JOINT_R_KNEE, idx});         // 5: r_hip -> r_knee = r_thigh
  idx += 2;
  // Shins
  bones.push_back({13, 15, JOINT_L_ANKLE, idx + 1});   // 6: l_knee -> l_ankle = l_shin
  bones.push_back({14, 16, JOINT_R_ANKLE, idx});        // 7: r_knee -> r_ankle = r_shin
  idx += 2;
  // Hips (half hip width: pelvis center to hip joint)
  bones.push_back({11, 12, JOINT_L_HIP, -1});          // 8: hip width (special: symmetric by definition)
  idx += 1;
  // Shoulder width (spine top to shoulder)
  bones.push_back({5, 6, JOINT_L_SHOULDER, -1});        // 9: shoulder width (special)
  idx += 1;
  // Feet (ankle to toe)
  bones.push_back({15, 17, JOINT_L_TOE, idx + 1});     // 10: l_ankle -> l_big_toe
  bones.push_back({16, 20, JOINT_R_TOE, idx});          // 11: r_ankle -> r_big_toe
  idx += 2;

  // Hand bones — measured from COCO hand keypoints
  // Left hand: COCO 91=wrist, 92-95=thumb, 96-99=index, 100-103=middle, 104-107=ring, 108-111=pinky
  // Thumb metacarpal (wrist to CMC)
  bones.push_back({91, 92, JOINT_L_THUMB_CMC, idx + 1});  // 12
  bones.push_back({112, 113, JOINT_R_THUMB_CMC, idx});     // 13
  idx += 2;
  // Thumb proximal (CMC to MCP)
  bones.push_back({92, 93, JOINT_L_THUMB_MCP, idx + 1});   // 14
  bones.push_back({113, 114, JOINT_R_THUMB_MCP, idx});      // 15
  idx += 2;
  // Thumb distal (MCP to IP)
  bones.push_back({93, 94, JOINT_L_THUMB_IP, idx + 1});    // 16
  bones.push_back({114, 115, JOINT_R_THUMB_IP, idx});       // 17
  idx += 2;

  // Index finger
  bones.push_back({91, 96, JOINT_L_INDEX_MCP, idx + 1});   // 18: wrist to MCP
  bones.push_back({112, 117, JOINT_R_INDEX_MCP, idx});      // 19
  idx += 2;
  bones.push_back({96, 97, JOINT_L_INDEX_PIP, idx + 1});   // 20: MCP to PIP
  bones.push_back({117, 118, JOINT_R_INDEX_PIP, idx});      // 21
  idx += 2;
  bones.push_back({97, 98, JOINT_L_INDEX_DIP, idx + 1});   // 22: PIP to DIP
  bones.push_back({118, 119, JOINT_R_INDEX_DIP, idx});      // 23
  idx += 2;

  // Middle finger
  bones.push_back({91, 100, JOINT_L_MIDDLE_MCP, idx + 1});
  bones.push_back({112, 121, JOINT_R_MIDDLE_MCP, idx});
  idx += 2;
  bones.push_back({100, 101, JOINT_L_MIDDLE_PIP, idx + 1});
  bones.push_back({121, 122, JOINT_R_MIDDLE_PIP, idx});
  idx += 2;
  bones.push_back({101, 102, JOINT_L_MIDDLE_DIP, idx + 1});
  bones.push_back({122, 123, JOINT_R_MIDDLE_DIP, idx});
  idx += 2;

  // Ring finger
  bones.push_back({91, 104, JOINT_L_RING_MCP, idx + 1});
  bones.push_back({112, 125, JOINT_R_RING_MCP, idx});
  idx += 2;
  bones.push_back({104, 105, JOINT_L_RING_PIP, idx + 1});
  bones.push_back({125, 126, JOINT_R_RING_PIP, idx});
  idx += 2;
  bones.push_back({105, 106, JOINT_L_RING_DIP, idx + 1});
  bones.push_back({126, 127, JOINT_R_RING_DIP, idx});
  idx += 2;

  // Pinky finger
  bones.push_back({91, 108, JOINT_L_PINKY_MCP, idx + 1});
  bones.push_back({112, 129, JOINT_R_PINKY_MCP, idx});
  idx += 2;
  bones.push_back({108, 109, JOINT_L_PINKY_PIP, idx + 1});
  bones.push_back({129, 130, JOINT_R_PINKY_PIP, idx});
  idx += 2;
  bones.push_back({109, 110, JOINT_L_PINKY_DIP, idx + 1});
  bones.push_back({130, 131, JOINT_R_PINKY_DIP, idx});
  idx += 2;

  (void)idx;
}

Skeleton::Skeleton() {
  q = Eigen::VectorXf::Zero(NUM_DOFS);

  init_body_joints();
  init_hand_joints();
  init_sites();
  init_bones();
  build_ancestor_map();

  // Initialize transforms to identity
  for (auto& t : world_transforms)
    t = Eigen::Matrix4f::Identity();
}

// Build rotation matrix from axis-angle
static Eigen::Matrix3f axis_angle_rotation(const Eigen::Vector3f& axis, float angle) {
  float c = std::cos(angle);
  float s = std::sin(angle);
  float t = 1.0f - c;
  float x = axis.x(), y = axis.y(), z = axis.z();
  Eigen::Matrix3f R;
  R << t*x*x + c,   t*x*y - s*z, t*x*z + s*y,
       t*x*y + s*z, t*y*y + c,   t*y*z - s*x,
       t*x*z - s*y, t*y*z + s*x, t*z*z + c;
  return R;
}

void Skeleton::compute_fk() {
  for (int j = 0; j < NUM_JOINTS; j++) {
    const JointDef& jd = joints[j];

    // Start with parent transform (identity for root)
    Eigen::Matrix4f T;
    if (jd.parent >= 0)
      T = world_transforms[jd.parent];
    else
      T = Eigen::Matrix4f::Identity();

    // Apply bone translation (in parent's local frame)
    if (jd.bone_length > 0.0f) {
      Eigen::Vector3f offset = jd.bone_dir * jd.bone_length;
      T(0, 3) += T(0, 0) * offset.x() + T(0, 1) * offset.y() + T(0, 2) * offset.z();
      T(1, 3) += T(1, 0) * offset.x() + T(1, 1) * offset.y() + T(1, 2) * offset.z();
      T(2, 3) += T(2, 0) * offset.x() + T(2, 1) * offset.y() + T(2, 2) * offset.z();
    }

    // Apply rotations for each DOF
    // For translational root: DOFs directly set position
    if (j == JOINT_ROOT_TRANS) {
      T(0, 3) = q[jd.dof_offset + 0];
      T(1, 3) = q[jd.dof_offset + 1];
      T(2, 3) = q[jd.dof_offset + 2];
      // Translational DOF world axes are just identity directions
      dof_world_axes[jd.dof_offset + 0] = Eigen::Vector3f(1, 0, 0);
      dof_world_axes[jd.dof_offset + 1] = Eigen::Vector3f(0, 1, 0);
      dof_world_axes[jd.dof_offset + 2] = Eigen::Vector3f(0, 0, 1);
    } else {
      // For each DOF, record the world axis BEFORE this DOF's rotation is applied,
      // then apply the rotation. This gives the Jacobian the correct intermediate frames.
      for (int d = 0; d < jd.num_dofs; d++) {
        // World axis for this DOF = current rotation frame * local axis
        dof_world_axes[jd.dof_offset + d] = T.block<3, 3>(0, 0) * jd.axes[d];

        float angle = q[jd.dof_offset + d];
        if (std::abs(angle) < 1e-8f) continue;

        Eigen::Matrix3f R_local = axis_angle_rotation(jd.axes[d], angle);
        T.block<3, 3>(0, 0) = T.block<3, 3>(0, 0) * R_local;
      }
    }

    world_transforms[j] = T;
  }

  // Compute site positions
  for (int s = 0; s < static_cast<int>(sites.size()); s++) {
    const SiteDef& sd = sites[s];
    const Eigen::Matrix4f& T = world_transforms[sd.joint];

    if (sd.local_offset.isZero()) {
      site_positions[s] = T.block<3, 1>(0, 3);
    } else {
      // Transform local offset to world
      Eigen::Vector3f world_offset = T.block<3, 3>(0, 0) * sd.local_offset;
      site_positions[s] = T.block<3, 1>(0, 3) + world_offset;
    }
  }
}

void Skeleton::build_ancestor_map() {
  m_site_ancestors.resize(sites.size());

  for (int s = 0; s < static_cast<int>(sites.size()); s++) {
    auto& ancestors = m_site_ancestors[s];
    ancestors.clear();

    // Walk up the joint tree from this site's joint to root
    int j = sites[s].joint;
    while (j >= 0) {
      const JointDef& jd = joints[j];
      // Add each DOF of this joint (in reverse order so root DOFs come first)
      for (int d = jd.num_dofs - 1; d >= 0; d--) {
        ancestors.push_back({j, d});
      }
      j = jd.parent;
    }

    // Reverse so ancestors are root-to-leaf order
    std::reverse(ancestors.begin(), ancestors.end());
  }
}

void Skeleton::compute_jacobian(Eigen::MatrixXf& J) const {
  int num_sites = static_cast<int>(sites.size());
  J.setZero(3 * num_sites, NUM_DOFS);

  for (int s = 0; s < num_sites; s++) {
    const Eigen::Vector3f& p_site = site_positions[s];

    for (const auto& [joint_idx, local_dof] : m_site_ancestors[s]) {
      const JointDef& jd = joints[joint_idx];
      int col = jd.dof_offset + local_dof;

      // Use the precomputed per-DOF world axis (computed at the correct intermediate
      // rotation state during FK, before this DOF's rotation was applied)
      const Eigen::Vector3f& world_axis = dof_world_axes[col];

      if (joint_idx == JOINT_ROOT_TRANS) {
        // Translational DOF: site moves along the axis direction
        J.block<3, 1>(3 * s, col) = world_axis;
      } else {
        // Rotational DOF: J = axis_world x (p_site - p_joint)
        Eigen::Vector3f p_joint = world_transforms[joint_idx].block<3, 1>(0, 3);
        J.block<3, 1>(3 * s, col) = world_axis.cross(p_site - p_joint);
      }
    }
  }
}

Eigen::Vector3f Skeleton::joint_world_pos(int joint_idx) const {
  return world_transforms[joint_idx].block<3, 1>(0, 3);
}

Eigen::Vector3f Skeleton::dof_world_axis(int joint_idx, int local_dof) const {
  return world_transforms[joint_idx].block<3, 3>(0, 0) * joints[joint_idx].axes[local_dof];
}

Eigen::VectorXf Skeleton::rom_lower() const {
  Eigen::VectorXf lo(NUM_DOFS);
  for (int j = 0; j < NUM_JOINTS; j++) {
    const JointDef& jd = joints[j];
    for (int d = 0; d < jd.num_dofs; d++)
      lo[jd.dof_offset + d] = jd.rom_lower[d];
  }
  return lo;
}

Eigen::VectorXf Skeleton::rom_upper() const {
  Eigen::VectorXf hi(NUM_DOFS);
  for (int j = 0; j < NUM_JOINTS; j++) {
    const JointDef& jd = joints[j];
    for (int d = 0; d < jd.num_dofs; d++)
      hi[jd.dof_offset + d] = jd.rom_upper[d];
  }
  return hi;
}

