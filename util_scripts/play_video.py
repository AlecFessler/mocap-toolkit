import cv2
import os
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python play_video.py <video_file>")
        sys.exit(1)

    video_file = sys.argv[1]
    if not os.path.exists(video_file):
        print("File not found: %s" % video_file)
        sys.exit(1)

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error opening video file")
        sys.exit(1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Video", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
