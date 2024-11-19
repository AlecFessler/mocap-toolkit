# © 2024 Alec Fessler
# MIT License
# See LICENSE file in the project root for full license information.

import cv2
import json
import sys
import math

def find_aruco_corners(img):
    """
    Returns the full list of corners of the aruco markers in the img.
    Also validates that there are 3 markers as expected by this script.

    Parameters:
    - img: the img to detect markers in

    Returns:
    - list: a list containing a list for each marker, which contains the markers corners
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    corners_list, ids, _ = detector.detectMarkers(img)

    if ids is not None:
        print("Detected marker corners:")
        print(corners_list)
        print("Detected marker IDs:")
        print(ids)
    else:
        print("No markers detected.")
        sys.exit(1)

    num_corners = len(corners_list)
    if num_corners != 3:
        print(f"There should be 3 corners, there are {num_corners}")
        sys.exit(1)

    return corners_list

def find_outer_corner(height, width, corners):
    """
    Returns the outermost corner from the four corners of an aruco marker.
    Outmost is defined as being furthest from the center of the img.

    Parameters:
    - height (int): Height of the img
    - width (int): Width of the img
    - corners (list): List of 4 (x, y) coordinates of a single ArUco marker

    Returns:
    - list: [x, y] coordinates of the outermost corner
    """
    mid_height = height // 2
    mid_width = width // 2

    distances = []
    for corner in corners:
        x, y = corner
        distance = math.sqrt((x - mid_width) ** 2 + (y - mid_height) ** 2)
        distances.append(distance)

    max_dist_idx = distances.index(max(distances))
    return corners[max_dist_idx]

def compute_rot_err(corners):
    """
    Computes the mean rotation error from the ideal (90 degree vertical, 0 degree horizontal)

    Parameters:
    - corners: List of 3 (x, y) coordinates representing the outermost corners of the markers.

    Returns:
    - rotation_angle (float): Negative rotation angle of the camera.
    """
    # sort corners to identify vertical and horizontal sides
    corners = sorted(corners, key=lambda p: p[0])

    x1, y1 = corners[0]
    x2, y2 = corners[1]
    x3, y3 = corners[2]

    # vertical side, first two points, closest in x
    slope_v = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
    angle_v = math.degrees(math.atan(slope_v))

    # horizontal side, first and third points
    slope_h = (y3 - y1) / (x3 - x1) if x3 != x1 else 0
    angle_h = math.degrees(math.atan(slope_h))

    rot = ((90 - abs(angle_v)) + abs(angle_h)) / 2
    return -rot

def main():
    """
    The calibration script assumes there will be exactly 3 aruco markers
    which are positioned along 3 of the 4 corners of a square which represents
    the crop region.

    It works by first determining the rotation error of the camera based on
    the slope of the lines formed by the crop region, and then corrects for
    this and redetects the markers. It then uses these straightened out
    markers to determine the crop region. And lastly, with the crop region
    defined, it determines the cameras translation error, which must be subtracted
    from any coordinates in the image to achieve a virtually aligned camera. These
    parameters are then saved to a json file for usage by the video preprocessing script.
    """
    if len(sys.argv) < 3:
        print("Usage: python script.py <img_path> <cam_name>")
        sys.exit(1)

    cam_name = sys.argv[2]

    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load img at {img_path}")
        sys.exit(1)

    height, width = img.shape[:2]

    # get lens distortion matrix from the other calibration script and apply it here for accuracy

    corners_list = find_aruco_corners(img)
    outer_corners = [find_outer_corner(height, width, corners) for corners in corners_list]

    rot_err = compute_rot_err(outer_corners)
    center = (width // 2, height // 2)
    rot_mat = cv2.getRotationMatrix2D(center, rot_err, 1.0)
    rot_img = cv2.warpAffine(img, rot_mat, (width, height))

    corners_list = find_aruco_corners(rot_img)
    outer_corners = [find_outer_corner(height, width, corners) for corners in corners_list]


    x_min = min([p[0] for p in outer_corners])
    y_min = min([p[1] for p in outer_corners])
    x_max = max([p[0] for p in outer_corners])
    y_max = max([p[1] for p in outer_corners])

    crop_width = x_max - x_min
    crop_height = y_max - y_min
    crop_center = (x_min + crop_width // 2, y_min + crop_height // 2)

    x_alignment_err = center[0] - crop_center[0]
    y_alignment_err = center[1] - crop_center[1]

    alignment_params = {
        'rot_err': rot_err,
        'crop_region': (int(x_min), int(y_min), int(crop_width), int(crop_height)),
        'alignment_err': (int(x_alignment_err), int(y_alignment_err))
    }

    with open(f'{cam_name}_alignment_params.json', 'w') as f:
        json.dump(alignment_params, f, indent=2)

if __name__ == "__main__":
    main()
