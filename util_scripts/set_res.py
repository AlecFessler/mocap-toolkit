import cv2
import os
import sys

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python set_res.py <video_path> <resolution (e.g. 1920x1080)>")
        sys.exit(1)

    video_path = sys.argv[1]
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Error: Could not open video.")
        sys.exit(1)

    resolution = sys.argv[2]
    width, height = map(int, resolution.split("x"))
    if width <= 0 or height <= 0:
        print("Error: Invalid resolution.")
        sys.exit(1)
    resolution = (width, height)

    video_name = os.path.basename(video_path)
    video_name = os.path.splitext(video_name)[0]
    video_name += f"_{width}x{height}.mp4"
    video_name = os.path.join(os.path.dirname(video_path), video_name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_name, fourcc, video.get(cv2.CAP_PROP_FPS), resolution)

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame = cv2.resize(frame, resolution)
        video_writer.write(frame)

    video_writer.release()
    video.release()
