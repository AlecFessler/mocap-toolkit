#!/usr/bin/env python3
"""
Analyze mocap capture data: per-keypoint visibility, reprojection error,
bone length consistency, and pairwise triangulation spread.
"""

import struct
import numpy as np

NUM_KEYPOINTS = 133
NUM_CAMERAS = 3
NUM_BONES = 12
CONF_THRESHOLD = 0.5

KP_BYTES = NUM_KEYPOINTS * 3 * 4
CONF_BYTES = NUM_CAMERAS * NUM_KEYPOINTS * 4
# frame_metrics: 308 reproj + 308 spread + 308 int num_views + 12 bone_lengths
METRICS_BYTES = NUM_KEYPOINTS * 4 + NUM_KEYPOINTS * 4 + NUM_KEYPOINTS * 4 + NUM_BONES * 4

KP_NAMES = {
    0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
    5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
    9: "left_hip", 10: "right_hip", 11: "left_knee", 12: "right_knee",
    13: "left_ankle", 14: "right_ankle",
    15: "left_big_toe", 16: "left_small_toe", 17: "left_heel",
    18: "right_big_toe", 19: "right_small_toe", 20: "right_heel",
    41: "right_wrist", 62: "left_wrist", 69: "neck",
}

BONE_NAMES = [
    "L shoulder-elbow", "L elbow-wrist", "R shoulder-elbow", "R elbow-wrist",
    "L shoulder-hip", "R shoulder-hip", "L hip-knee", "L knee-ankle",
    "R hip-knee", "R knee-ankle", "shoulder width", "hip width",
]

BODY_GROUPS = {
    "Head":       [0, 1, 2, 3, 4],
    "Torso":      [5, 6, 9, 10, 69],
    "Left Arm":   [5, 7, 62],
    "Right Arm":  [6, 8, 41],
    "Left Hand":  list(range(42, 63)),
    "Right Hand": list(range(21, 42)),
    "Left Leg":   [9, 11, 13],
    "Right Leg":  [10, 12, 14],
    "Left Foot":  [13, 15, 16, 17],
    "Right Foot": [14, 18, 19, 20],
}


def parse_metrics(data, num_frames):
    """Parse metrics binary: reproj[308] + spread[308] + num_views[308 as int32] + bones[12]"""
    reproj = np.zeros((num_frames, NUM_KEYPOINTS), dtype=np.float32)
    spread = np.zeros((num_frames, NUM_KEYPOINTS), dtype=np.float32)
    views = np.zeros((num_frames, NUM_KEYPOINTS), dtype=np.int32)
    bones = np.zeros((num_frames, NUM_BONES), dtype=np.float32)

    for f in range(num_frames):
        offset = f * METRICS_BYTES
        chunk = data[offset:offset + METRICS_BYTES]
        if len(chunk) < METRICS_BYTES:
            break
        pos = 0
        reproj[f] = np.frombuffer(chunk[pos:pos + NUM_KEYPOINTS * 4], dtype=np.float32)
        pos += NUM_KEYPOINTS * 4
        spread[f] = np.frombuffer(chunk[pos:pos + NUM_KEYPOINTS * 4], dtype=np.float32)
        pos += NUM_KEYPOINTS * 4
        views[f] = np.frombuffer(chunk[pos:pos + NUM_KEYPOINTS * 4], dtype=np.int32)
        pos += NUM_KEYPOINTS * 4
        bones[f] = np.frombuffer(chunk[pos:pos + NUM_BONES * 4], dtype=np.float32)

    return reproj, spread, views, bones


def main():
    kp_path = "/tmp/mocap_keypoints_alec.bin"
    conf_path = "/tmp/mocap_confidence_alec.bin"
    metrics_path = "/tmp/mocap_metrics_alec.bin"

    with open(kp_path, "rb") as f:
        kp_data = f.read()
    with open(conf_path, "rb") as f:
        conf_data = f.read()
    with open(metrics_path, "rb") as f:
        metrics_data = f.read()

    num_frames_kp = len(kp_data) // KP_BYTES
    num_frames_conf = len(conf_data) // CONF_BYTES
    num_frames_met = len(metrics_data) // METRICS_BYTES
    num_frames = min(num_frames_kp, num_frames_conf, num_frames_met)

    if num_frames == 0:
        print("No data found.")
        return

    print(f"{'=' * 85}")
    print(f"  MOCAP CAPTURE ANALYSIS — {num_frames} frames")
    print(f"{'=' * 85}")

    # Parse confidence
    all_conf = np.zeros((num_frames, NUM_CAMERAS, NUM_KEYPOINTS), dtype=np.float32)
    for f in range(num_frames):
        offset = f * CONF_BYTES
        floats = np.frombuffer(conf_data[offset:offset + CONF_BYTES], dtype=np.float32)
        all_conf[f] = floats.reshape(NUM_CAMERAS, NUM_KEYPOINTS)

    # Parse 3D keypoints
    all_kp = np.zeros((num_frames, NUM_KEYPOINTS, 3), dtype=np.float32)
    for f in range(num_frames):
        offset = f * KP_BYTES
        floats = np.frombuffer(kp_data[offset:offset + KP_BYTES], dtype=np.float32)
        all_kp[f] = floats.reshape(NUM_KEYPOINTS, 3)

    # Parse metrics
    reproj, spread, views, bones = parse_metrics(metrics_data, num_frames)

    # ========== CAMERA VISIBILITY ==========
    print(f"\n{'=' * 85}")
    print("  CAMERA VISIBILITY (per keypoint, % of frames)")
    print(f"{'=' * 85}")
    print(f"{'Keypoint':<20} {'0 cams':>8} {'1 cam':>8} {'2 cams':>8} {'3 cams':>8} {'triang':>8}")
    print(f"{'-' * 85}")

    triangulated = ~np.isnan(all_kp[:, :, 0])
    pct_tri = triangulated.mean(axis=0) * 100

    for idx in sorted(KP_NAMES.keys()):
        name = KP_NAMES[idx]
        v = views[:, idx]
        p0 = (v == 0).mean() * 100
        p1 = (v == 1).mean() * 100
        p2 = (v == 2).mean() * 100
        p3 = (v == 3).mean() * 100
        print(f"{name:<20} {p0:7.1f}% {p1:7.1f}% {p2:7.1f}% {p3:7.1f}% {pct_tri[idx]:7.1f}%")

    # Per body group
    print(f"\n{'Body Group':<20} {'0 cams':>8} {'1 cam':>8} {'2 cams':>8} {'3 cams':>8} {'triang':>8}")
    print(f"{'-' * 85}")
    for group_name, indices in BODY_GROUPS.items():
        v = views[:, indices]
        p0 = (v == 0).mean() * 100
        p1 = (v == 1).mean() * 100
        p2 = (v == 2).mean() * 100
        p3 = (v == 3).mean() * 100
        gt = pct_tri[indices].mean()
        print(f"{group_name:<20} {p0:7.1f}% {p1:7.1f}% {p2:7.1f}% {p3:7.1f}% {gt:7.1f}%")

    # ========== REPROJECTION ERROR ==========
    print(f"\n{'=' * 85}")
    print("  REPROJECTION ERROR (pixels)")
    print(f"{'=' * 85}")
    print(f"{'Keypoint':<20} {'mean':>8} {'median':>8} {'p95':>8}")
    print(f"{'-' * 50}")

    for idx in sorted(KP_NAMES.keys()):
        name = KP_NAMES[idx]
        r = reproj[:, idx]
        valid = r[~np.isnan(r)]
        if len(valid) > 0:
            print(f"{name:<20} {valid.mean():7.1f}px {np.median(valid):7.1f}px {np.percentile(valid, 95):7.1f}px")
        else:
            print(f"{name:<20}      N/A      N/A      N/A")

    # Per body group
    print(f"\n{'Body Group':<20} {'mean':>8} {'median':>8} {'p95':>8}")
    print(f"{'-' * 50}")
    for group_name, indices in BODY_GROUPS.items():
        r = reproj[:, indices].flatten()
        valid = r[~np.isnan(r)]
        if len(valid) > 0:
            print(f"{group_name:<20} {valid.mean():7.1f}px {np.median(valid):7.1f}px {np.percentile(valid, 95):7.1f}px")
        else:
            print(f"{group_name:<20}      N/A      N/A      N/A")

    # ========== BONE LENGTH CONSISTENCY ==========
    print(f"\n{'=' * 85}")
    print("  BONE LENGTH CONSISTENCY (mm)")
    print(f"{'=' * 85}")
    print(f"{'Bone':<25} {'mean':>8} {'stddev':>8} {'CV':>8} {'status':>10}")
    print(f"{'-' * 65}")

    for b in range(NUM_BONES):
        bl = bones[:, b]
        valid = bl[~np.isnan(bl)]
        if len(valid) > 5:
            mean = valid.mean()
            std = valid.std()
            cv = std / mean if mean > 0 else float('inf')
            status = "GOOD" if cv < 0.05 else "OK" if cv < 0.1 else "UNSTABLE" if cv < 0.2 else "BAD"
            print(f"{BONE_NAMES[b]:<25} {mean:7.1f} {std:7.1f} {cv:7.3f}   {status:>8}")
        else:
            print(f"{BONE_NAMES[b]:<25}      N/A      N/A      N/A        N/A")

    # ========== PAIRWISE SPREAD ==========
    print(f"\n{'=' * 85}")
    print("  PAIRWISE TRIANGULATION SPREAD (mm, 3-camera keypoints only)")
    print(f"{'=' * 85}")
    print(f"{'Body Group':<20} {'mean':>8} {'median':>8} {'p95':>8}")
    print(f"{'-' * 50}")

    for group_name, indices in BODY_GROUPS.items():
        s = spread[:, indices].flatten()
        valid = s[~np.isnan(s)]
        if len(valid) > 0:
            print(f"{group_name:<20} {valid.mean():7.1f} {np.median(valid):7.1f} {np.percentile(valid, 95):7.1f}")
        else:
            print(f"{group_name:<20}      N/A      N/A      N/A")

    # ========== SUMMARY SCORE ==========
    print(f"\n{'=' * 85}")
    print("  SUMMARY")
    print(f"{'=' * 85}")

    all_reproj = reproj.flatten()
    valid_reproj = all_reproj[~np.isnan(all_reproj)]
    avg_reproj = valid_reproj.mean() if len(valid_reproj) > 0 else float('nan')

    bone_cvs = []
    for b in range(NUM_BONES):
        valid = bones[:, b][~np.isnan(bones[:, b])]
        if len(valid) > 5:
            bone_cvs.append(valid.std() / valid.mean() if valid.mean() > 0 else 1.0)
    avg_bone_cv = np.mean(bone_cvs) if bone_cvs else float('nan')

    all_spread = spread.flatten()
    valid_spread = all_spread[~np.isnan(all_spread)]
    avg_spread = valid_spread.mean() if len(valid_spread) > 0 else float('nan')

    avg_tri = triangulated.sum(axis=1).mean()
    good_frames = (triangulated.sum(axis=1) >= 50).sum()

    print(f"  Avg reprojection error:  {avg_reproj:.1f} px  {'GOOD' if avg_reproj < 5 else 'OK' if avg_reproj < 10 else 'BAD'}")
    print(f"  Avg bone length CV:      {avg_bone_cv:.3f}    {'GOOD' if avg_bone_cv < 0.05 else 'OK' if avg_bone_cv < 0.1 else 'BAD'}")
    print(f"  Avg pairwise spread:     {avg_spread:.1f} mm  {'GOOD' if avg_spread < 10 else 'OK' if avg_spread < 50 else 'BAD'}")
    print(f"  Avg keypoints/frame:     {avg_tri:.0f} / {NUM_KEYPOINTS}")
    print(f"  Good frames (>=50 kps):  {good_frames} / {num_frames} ({good_frames/num_frames*100:.0f}%)")
    print()


if __name__ == "__main__":
    main()
