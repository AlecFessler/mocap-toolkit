# © 2024 Alec Fessler
# MIT License
# See LICENSE file in the project root for full license information.

import cv2
import numpy as np

def main():
    marker_size = 200
    border_bits = 1
    byte_array_size = marker_size + border_bits * 2

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    for marker_id in range(3):
        marker_buf = np.zeros((byte_array_size, byte_array_size), dtype=np.uint8)
        marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size, marker_buf, border_bits)
        cv2.imwrite(f"marker_{marker_id}.png", marker)

if __name__ == "__main__":
    main()
