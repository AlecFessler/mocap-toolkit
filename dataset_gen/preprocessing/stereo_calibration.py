import cv2
import numpy as np
import json
import sys
from pathlib import Path

def load_calibration(cam_name):
    """
    Loads and returns the camera matrix and distortion coefficients

    Parameters:
    - cam_name (str): the name of the camera to load

    Returns:
    - tuple (np.array, np.array): the camera matrix and distortion coefficients
    """
    with open(f'{cam_name}_calibration.json', 'r') as f:
        data = json.load(f)
    return np.array(data['camera_matrix']), np.array(data['dist_coeffs'])

def find_chessboard_corners(img_path, chess_rows=6, chess_cols=9):
    """
    Finds the chessboard corners in an img

    Parameters:
    - img_path (str): the path of the img to load
    - chess_rows (int): the number of rows in the chessboard pattern
    - chess_cols (int): the number of cols in the chessboard pattern

    Returns:
    - tuple (np.array, np.array): the corners array and the img h,w
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f'Failed to read img: {img_path}')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (chess_cols, chess_rows), None)

    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return corners, img.shape[:2][::-1]  # width, height
    return None, None

def main():
    if len(sys.argv) != 4:
        print("Usage: python stereo_calibration.py <stereo_img_dir> <cam1_name> <cam2_name>")
        sys.exit(1)

    stereo_dir = sys.argv[1]
    cam1_name = sys.argv[2]
    cam2_name = sys.argv[3]

    print(f'Calibrating stereo pair: {cam1_name} - {cam2_name}')

    cam1_matrix, cam1_dist = load_calibration(cam1_name)
    cam2_matrix, cam2_dist = load_calibration(cam2_name)

    CHESS_ROWS = 6
    CHESS_COLS = 9
    SQUARE_SIZE = 25.0  # mm

    objp = np.zeros((CHESS_ROWS * CHESS_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESS_COLS, 0:CHESS_ROWS].T.reshape(-1, 2) * SQUARE_SIZE

    cam1_imgs = sorted(Path(stereo_dir).glob(f'{cam1_name}*.png'))
    cam2_imgs = sorted(Path(stereo_dir).glob(f'{cam2_name}*.png'))

    if len(cam1_imgs) != len(cam2_imgs):
        raise ValueError(f'Unequal number of imgs for {cam1_name} and {cam2_name}')

    obj_points = []
    cam1_points = []
    cam2_points = []
    img_size = None

    for img1_path, img2_path in zip(cam1_imgs, cam2_imgs):
        corners1, size1 = find_chessboard_corners(img1_path)
        corners2, size2 = find_chessboard_corners(img2_path)

        if corners1 is not None and corners2 is not None:
            if img_size is None:
                img_size = size1
            elif size1 != img_size or size2 != img_size:
                raise ValueError('Inconsistent img sizes')

            obj_points.append(objp)
            cam1_points.append(corners1)
            cam2_points.append(corners2)

    if not obj_points:
        raise ValueError('No valid img pairs found')

    print(f'Found {len(obj_points)} valid img pairs')

    ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        obj_points,
        cam1_points,
        cam2_points,
        cam1_matrix,
        cam1_dist,
        cam2_matrix,
        cam2_dist,
        img_size,
        flags=cv2.CALIB_FIX_INTRINSIC
    )

    distance = np.linalg.norm(T)

    stereo_data = {
        'R': R.tolist(),
        'T': T.tolist(),
        'E': E.tolist(),
        'F': F.tolist(),
        'distance_mm': float(distance),
        'img_size': img_size
    }

    calibration_file = f'stereo_calibration_{cam1_name}_{cam2_name}.json'
    with open(calibration_file, 'w') as f:
        json.dump(stereo_data, f, indent=2)

    print(f'Distance between cameras: {distance:.2f}mm')
    print(f'Calibration saved to {calibration_file}')

if __name__ == "__main__":
    main()
