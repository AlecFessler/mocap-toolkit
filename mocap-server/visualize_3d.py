#!/usr/bin/env python3
"""
Real-time 3D keypoint visualizer for mocap-toolkit.

Reads /tmp/mocap_keypoints_alec.bin (written by mocap-toolkit mocap mode)
and animates the 3D skeleton as new frames arrive.

Binary format: each frame is 308 keypoints * 3 floats (x,y,z) = 3696 bytes.
Invalid keypoints have NaN coordinates.
"""

import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib.animation import FuncAnimation

NUM_KEYPOINTS = 308
FLOATS_PER_FRAME = NUM_KEYPOINTS * 3
FRAME_BYTES = FLOATS_PER_FRAME * 4  # 3696 bytes

KEYPOINTS_PATH = "/tmp/mocap_keypoints_alec.bin"

# Sapiens Goliath 308 keypoint skeleton edges
# Extracted from facebookresearch/sapiens pose/configs/_base_/datasets/goliath.py
SKELETON_EDGES = [
    # body
    (13, 11),  # left_ankle -> left_knee
    (11, 9),   # left_knee -> left_hip
    (14, 12),  # right_ankle -> right_knee
    (12, 10),  # right_knee -> right_hip
    (9, 10),   # left_hip -> right_hip
    (5, 9),    # left_shoulder -> left_hip
    (6, 10),   # right_shoulder -> right_hip
    (5, 6),    # left_shoulder -> right_shoulder
    (5, 7),    # left_shoulder -> left_elbow
    (6, 8),    # right_shoulder -> right_elbow
    (7, 62),   # left_elbow -> left_wrist
    (8, 41),   # right_elbow -> right_wrist
    # head
    (1, 2),    # left_eye -> right_eye
    (0, 1),    # nose -> left_eye
    (0, 2),    # nose -> right_eye
    (1, 3),    # left_eye -> left_ear
    (2, 4),    # right_eye -> right_ear
    (3, 5),    # left_ear -> left_shoulder
    (4, 6),    # right_ear -> right_shoulder
    # feet
    (13, 15),  # left_ankle -> left_big_toe
    (13, 16),  # left_ankle -> left_small_toe
    (13, 17),  # left_ankle -> left_heel
    (14, 18),  # right_ankle -> right_big_toe
    (14, 19),  # right_ankle -> right_small_toe
    (14, 20),  # right_ankle -> right_heel
    # left hand
    (62, 45),  # left_wrist -> left_thumb_third_joint
    (45, 44),  (44, 43),  (43, 42),
    (62, 49),  # left_wrist -> left_forefinger_third_joint
    (49, 48),  (48, 47),  (47, 46),
    (62, 53),  # left_wrist -> left_middle_finger_third_joint
    (53, 52),  (52, 51),  (51, 50),
    (62, 57),  # left_wrist -> left_ring_finger_third_joint
    (57, 56),  (56, 55),  (55, 54),
    (62, 61),  # left_wrist -> left_pinky_finger_third_joint
    (61, 60),  (60, 59),  (59, 58),
    # right hand
    (41, 24),  # right_wrist -> right_thumb_third_joint
    (24, 23),  (23, 22),  (22, 21),
    (41, 28),  # right_wrist -> right_forefinger_third_joint
    (28, 27),  (27, 26),  (26, 25),
    (41, 32),  # right_wrist -> right_middle_finger_third_joint
    (32, 31),  (31, 30),  (30, 29),
    (41, 36),  # right_wrist -> right_ring_finger_third_joint
    (36, 35),  (35, 34),  (34, 33),
    (41, 40),  # right_wrist -> right_pinky_finger_third_joint
    (40, 39),  (39, 38),  (38, 37),
]

# Color coding for body parts
BODY_EDGES = set(range(0, 12))   # torso + limbs
HEAD_EDGES = set(range(12, 19))  # head
FEET_EDGES = set(range(19, 25))  # feet
LHAND_EDGES = set(range(25, 45)) # left hand
RHAND_EDGES = set(range(45, 65)) # right hand

# Body part groups: name -> (keypoint indices, color, anchor keypoint for label)
BODY_PARTS = {
    "Head":       (list(range(0, 5)) + list(range(70, 220)), '#ffaa00', 0),       # nose
    "Torso":      ([5, 6, 9, 10, 69], '#00ff88', 69),                              # neck
    "L Arm":      ([5, 7, 62, 67, 65], '#44ff44', 7),                              # left_elbow
    "R Arm":      ([6, 8, 41, 68, 66], '#44ff44', 8),                              # right_elbow
    "L Hand":     (list(range(42, 63)), '#44aaff', 62),                             # left_wrist
    "R Hand":     (list(range(21, 42)), '#ff44ff', 41),                             # right_wrist
    "L Leg":      ([9, 11, 13], '#00ffcc', 11),                                    # left_knee
    "R Leg":      ([10, 12, 14], '#00ffcc', 12),                                   # right_knee
    "L Foot":     ([13, 15, 16, 17], '#ff4444', 13),                               # left_ankle
    "R Foot":     ([14, 18, 19, 20], '#ff4444', 14),                               # right_ankle
}


class MocapVisualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(12, 9), facecolor='black')
        self.fig.canvas.manager.set_window_title("Mocap 3D Reconstruction")
        self.ax = self.fig.add_subplot(111, projection="3d", facecolor='black')
        self.ax.set_xlabel("X (mm)", color='white')
        self.ax.set_ylabel("Y (mm)", color='white')
        self.ax.set_zlabel("Z (mm)", color='white')
        self.ax.tick_params(colors='white')
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False

        self.scatter = None
        self.skeleton_lines = []
        self.labels = []
        self.fd = None
        self.frame_count = 0
        self.total_frames = 0
        self.last_points = None

        # auto-scale tracking
        self.bounds_initialized = False
        self.center = np.zeros(3)
        self.extent = 1000.0

    def open_file(self):
        if not os.path.exists(KEYPOINTS_PATH):
            return False
        self.fd = open(KEYPOINTS_PATH, "rb")
        self.total_frames = os.path.getsize(KEYPOINTS_PATH) // FRAME_BYTES
        # start at the latest frame
        if self.total_frames > 0:
            self.fd.seek((self.total_frames - 1) * FRAME_BYTES)
            self.frame_count = self.total_frames - 1
        return True

    def read_latest_frame(self):
        if self.fd is None:
            if not self.open_file():
                return None

        # always jump to the latest complete frame
        file_size = os.path.getsize(KEYPOINTS_PATH)
        new_total = file_size // FRAME_BYTES
        if new_total == 0:
            return self.last_points

        if new_total == self.total_frames and self.last_points is not None:
            return self.last_points  # no new data

        self.total_frames = new_total
        self.fd.seek((new_total - 1) * FRAME_BYTES)
        data = self.fd.read(FRAME_BYTES)
        if len(data) < FRAME_BYTES:
            return self.last_points

        self.frame_count = new_total

        floats = struct.unpack(f"{FLOATS_PER_FRAME}f", data)
        points = np.array(floats).reshape(NUM_KEYPOINTS, 3)
        # flip Y axis — camera convention has Y pointing down
        points[:, 1] = -points[:, 1]
        return points

    def update_bounds(self, valid_points):
        if len(valid_points) < 3:
            return

        new_center = np.mean(valid_points, axis=0)
        ranges = np.ptp(valid_points, axis=0)
        new_extent = max(np.max(ranges) * 0.8, 500.0)  # at least 500mm, pad by 80%

        if not self.bounds_initialized:
            self.center = new_center
            self.extent = new_extent
            self.bounds_initialized = True
        else:
            alpha = 0.3  # faster tracking
            self.center = self.center * (1 - alpha) + new_center * alpha
            self.extent = self.extent * (1 - alpha) + new_extent * alpha

        self.ax.set_xlim(self.center[0] - self.extent, self.center[0] + self.extent)
        self.ax.set_ylim(self.center[1] - self.extent, self.center[1] + self.extent)
        self.ax.set_zlim(self.center[2] - self.extent, self.center[2] + self.extent)

    def get_edge_color(self, edge_idx):
        if edge_idx in BODY_EDGES:
            return '#00ff88'   # green - body
        elif edge_idx in HEAD_EDGES:
            return '#ffaa00'   # orange - head
        elif edge_idx in FEET_EDGES:
            return '#ff4444'   # red - feet
        elif edge_idx in LHAND_EDGES:
            return '#44aaff'   # blue - left hand
        elif edge_idx in RHAND_EDGES:
            return '#ff44ff'   # magenta - right hand
        return '#ffffff'

    def animate(self, frame_idx):
        points = self.read_latest_frame()
        if points is None:
            self.ax.set_title("Waiting for data...", color='white', fontsize=14)
            return []

        self.last_points = points
        valid_mask = ~np.isnan(points[:, 0])
        valid_points = points[valid_mask]

        # filter outliers: remove points far from median
        if len(valid_points) > 3:
            median = np.median(valid_points, axis=0)
            dists = np.linalg.norm(valid_points - median, axis=1)
            inlier_mask = dists < 2000  # keep points within 2m of median
            valid_points = valid_points[inlier_mask]

            # rebuild valid_mask for skeleton drawing
            filtered_mask = np.zeros(NUM_KEYPOINTS, dtype=bool)
            for idx in np.where(~np.isnan(points[:, 0]))[0]:
                if np.linalg.norm(points[idx] - median) < 2000:
                    filtered_mask[idx] = True
            valid_mask = filtered_mask

        valid_count = len(valid_points)

        self.ax.set_title(
            f"3D Pose | Frame {self.frame_count}/{self.total_frames} | {valid_count}/{NUM_KEYPOINTS} keypoints",
            color='white', fontsize=14
        )

        # clear old drawings
        for line in self.skeleton_lines:
            line.remove()
        self.skeleton_lines.clear()
        for label in self.labels:
            label.remove()
        self.labels.clear()
        if self.scatter is not None:
            self.scatter.remove()
            self.scatter = None

        if len(valid_points) == 0:
            return []

        # draw skeleton edges
        for edge_idx, (i, j) in enumerate(SKELETON_EDGES):
            if i < NUM_KEYPOINTS and j < NUM_KEYPOINTS:
                if valid_mask[i] and valid_mask[j]:
                    color = self.get_edge_color(edge_idx)
                    line, = self.ax.plot(
                        [points[i, 0], points[j, 0]],
                        [points[i, 1], points[j, 1]],
                        [points[i, 2], points[j, 2]],
                        c=color, alpha=0.9, linewidth=2,
                    )
                    self.skeleton_lines.append(line)

        # draw joints as points
        # color body keypoints differently
        colors = np.full(len(valid_points), '#ffffff')
        self.scatter = self.ax.scatter(
            valid_points[:, 0],
            valid_points[:, 1],
            valid_points[:, 2],
            c='white', s=6, alpha=0.6, depthshade=True,
        )

        # draw body part labels at anchor keypoints
        for name, (indices, color, anchor) in BODY_PARTS.items():
            if anchor < NUM_KEYPOINTS and valid_mask[anchor]:
                pt = points[anchor]
                label = self.ax.text(
                    pt[0], pt[1], pt[2], f" {name}",
                    color=color, fontsize=7, alpha=0.8,
                    ha='left', va='center',
                )
                self.labels.append(label)

        self.update_bounds(valid_points)
        return []

    def run(self):
        print(f"Watching {KEYPOINTS_PATH} for 3D keypoint data...")
        print("Start './mocap-server/bin/mocap-toolkit mocap' to generate data.")
        print("Press Ctrl+C to quit.\n")

        ani = FuncAnimation(
            self.fig,
            self.animate,
            interval=80,  # ~12 fps, matches inference rate
            blit=False,
            cache_frame_data=False,
        )
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    viz = MocapVisualizer()
    viz.run()
