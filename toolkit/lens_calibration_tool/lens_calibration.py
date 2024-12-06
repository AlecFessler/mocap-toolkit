import cv2
import numpy as np
import sys
import glob
import json
from pathlib import Path

def main():
    if len(sys.argv) != 3:
        print("Usage: python lens_calibration.py <calibration_img_dir> <cam_name>")
        sys.exit(1)

    cam_name = sys.argv[2]
    img_dir = sys.argv[1]

    CHESS_ROWS = 6
    CHESS_COLS = 9
    SQUARE_SIZE = 25.0 #mm

    objp = np.zeros((CHESS_ROWS * CHESS_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESS_COLS, 0:CHESS_ROWS].T.reshape(-1, 2) * SQUARE_SIZE

    # lists to store object points and img points
    obj_points = [] # 3d points in real world space
    img_points = [] # 2d points in img plane

    img_paths = glob.glob(str(Path(img_dir) / '*.png'))
    if not img_paths:
        raise ValueError(f'No images found in {img_dir}')

    img = cv2.imread(img_paths[0])
    if img is None:
        raise ValueError(f'Failed to read image: {img_paths[0]}')
    img_size = img.shape[:2][::-1] # width, height

    print(f'Processing {len(img_paths)} images...')

    for file in img_paths:
        img = cv2.imread(file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (CHESS_COLS, CHESS_ROWS), None)

        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            obj_points.append(objp)
            img_points.append(corners)

            # cv2.drawChessboardCorners(img, (CHESS_COLS, CHESS_ROWS), corners, ret)
            # cv2.imshow('Corners', img)
            # cv2.waitKey(500)
        else:
            print(f'Failed to find chessboard corners in {file}')

    print(f'Successfully processed {len(obj_points)} images')

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points,
        img_points,
        img_size,
        None,
        None
    )

    mean_error = 0
    for i in range(len(obj_points)):
        img_points_2, _ = cv2.projectPoints(
            obj_points[i],
            rvecs[i],
            tvecs[i],
            camera_matrix,
            dist_coeffs
        )
        error = cv2.norm(
            img_points[i],
            img_points_2,
            cv2.NORM_L2
        ) / len(img_points_2)
        mean_error += error

    total_error = mean_error / len(obj_points)

    print(f'Total reprojection error: {total_error}')

    calibration_data = {
        'camera_matrix': camera_matrix.tolist(),
        'dist_coeffs': dist_coeffs.tolist(),
    }

    with open(f'{cam_name}_calibration.json', 'w') as f:
        json.dump(calibration_data, f, indent=2)

if __name__ == "__main__":
    main()
