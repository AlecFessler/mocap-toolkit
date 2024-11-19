import cv2
import sys
import math

def find_outer_corner(height, width, corners):
    """
    Returns the outermost corner from the four corners of an aruco marker.
    Outmost is defined as being furthest from the center of the image.

    This is working under the assumption that a marker will prominantly be
    in one of the four quadrants of the image, which will the the case
    for the way this script is calibrating the camera.

    Parameters:
    - height (int): Height of the image
    - width (int): Width of the image
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
    - rotation_angle (float): Camera rotation angle in degrees.
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
    return rot

    if len(sys.argv) < 2:
        print("Usage: python script.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        sys.exit(1)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    corners_list, ids, _ = detector.detectMarkers(image)

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

    outer_corners = []
    for corners in corners_list:
        outer_corner = find_outer_corner(height, width, corners)
        outer_corners.append(outer_corner)

    rot_angle = compute_rot_err(outer_corners)
    center = (width // 2, height // 2)
    rot_mat = cv2.getRotationMatrix2D(center, rot_angle, 1.0)
    rot_img = cv2.warpAffine(image, rot_mat, (width, height))

    # now that we've corrected for the rotation error, we need to
    # redetect the markers, and then use their positioning to determine
    # where to crop the image around them

if __name__ == "__main__":
    main()
