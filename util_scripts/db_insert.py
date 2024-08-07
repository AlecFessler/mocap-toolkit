import cv2
import h5py
import mediapipe as mp
import numpy as np
import os
import subprocess
import sys

if __name__ == '__main__':
    try:
        if len(sys.argv) != 2:
            print('Usage: python db_insert.py <db_path>')
            sys.exit(1)

        mp_hands = mp.solutions.hands
        db_path = '../hand_motions_labels.h5'
        open_mode = 'a' if os.path.exists(db_path) else 'w'
        with h5py.File(db_path, open_mode) as db:
            if 'version' not in db.attrs:
                db.attrs['version'] = 1.0
            if 'description' not in db.attrs:
                db.attrs['description'] = 'Dataset containing videos of hand motions, with the joint positions of the hand labeled for each frame.'
            if 'num_videos' not in db.attrs:
                db.attrs['num_videos'] = 0
            if 'video_resolution' not in db.attrs:
                db.attrs['video_resolution'] = (640, 360)

            num_videos = db.attrs['num_videos']
            db_video_resolution = db.attrs['video_resolution']

            video_path = sys.argv[1]
            if not video_path.endswith(('mp4', 'avi', 'mov', 'mkv')):
                print(f'Error: Invalid video file {video_path}, must be of type mp4, avi, mov, or mkv')
                sys.exit(1)

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f'Error: Could not open video file {video_path}')
                sys.exit(1)

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if frame_width < db_video_resolution[0] or frame_height < db_video_resolution[1]:
                print(f'Error: Video resolution {frame_width}x{frame_height} is less than the database resolution {db_video_resolution[0]}x{db_video_resolution[1]}')
                sys.exit(1)

            grp_name = str(num_videos)
            video_grp = db.create_group(grp_name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            video_grp.attrs['fps'] = fps

            # FFmpeg command to encode raw video frames:
            # -y: overwrite output file if exists
            # -f rawvideo: input format is raw video
            # -vcodec rawvideo: specify input codec as raw video
            # -s {frame_width}x{frame_height}: input frame resolution
            # -pix_fmt bgr24: pixel format is BGR with 8 bits per channel
            # -r {fps}: input frame rate
            # -i -: read input from stdin
            # -vf scale={db_video_resolution[0]}:{db_video_resolution[1]}: resize to target resolution
            # -an: disable audio
            # -c:v libx265: encode video with H.265 codec
            # -crf 28: set quality level (lower is better quality)
            # {grp_name}.mkv: output file name in MKV format
            ffmpeg_command = [
                    'ffmpeg',
                    '-y',
                    '-f', 'rawvideo',
                    '-vcodec', 'rawvideo',
                    '-s', f'{frame_width}x{frame_height}',
                    '-pix_fmt', 'bgr24',
                    '-r', str(int(fps)),
                    '-i', '-',
                    '-vf', f'scale={db_video_resolution[0]}:{db_video_resolution[1]}',
                    '-an',
                    '-c:v', 'libx265',
                    '-crf', '28',
                    f'../hand_motions_videos/{grp_name}.mkv'
            ]
            ffmpeg = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)

            frame_count = 0
            labels = []

            with mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ) as hands:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb_frame)
                    if not results.multi_hand_landmarks:
                        print(f'Error: Could not detect hand in frame {frame_count} of video file {video_path}')
                        sys.exit(1)
                    hand_landmarks = results.multi_hand_landmarks[0]
                    landmarks_arr = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                    labels.append(landmarks_arr)

                    ffmpeg.stdin.write(frame.tobytes())

            # casting to f16 to save space
            video_grp.create_dataset('labels', data=np.array(labels, dtype=np.float16), compression='gzip', compression_opts=9)
            video_grp.attrs['num_frames'] = frame_count
            db.attrs['num_videos'] = num_videos + 1
    finally:
        cap.release()
        if ffmpeg.stdin:
            ffmpeg.stdin.close()
        ffmpeg.terminate()
        try:
            ffmpeg.wait(timeout=2)
        except subprocess.TimeoutExpired:
            ffmpeg.kill()
            ffmpeg.wait()
        print(f'Inserted video {video_path} into database {db_path} as group {grp_name}')
