import cv2
import numpy as np
import os
import sqlite3
import torch
from torch.utils.data import Dataset
import json

class HandMotionVideoDataset(Dataset):
    """
    A PyTorch Dataset class for loading hand motion videos from a SQLite database.

    Videos are loaded as tensors of shape (T, H, W, C), where T is the number of frames,
    H is the height of the frame, W is the width of the frame, and C is the number of channels.

    Videos are loaded lazily, so only the frames needed for the current batch are loaded into memory.

    Only videos with a duration greater than or equal to the specified clip_duration are included in the dataset.
    There is no method to reset the clip_duration after initialization by design, as this may lead to unexpected behavior.
    Instead, reinitialize the dataset with the new clip_duration.

    Videos will be clipped to the specified duration, starting at a random frame within the video.
    This may result in a slight bias towards the beginning of the video, but it is a small bias.
    The upside is that it effectively increases the size of the dataset by allowing multiple clips from a single video.

    The clip_duration can be used to enable cirriculum learning, where the model is trained on shorter clips first,
    and then gradually increases the clip duration as the model learns.

    :member db_path: The path to the SQLite database file.
    :member clip_duration: The duration of the video clips to load in seconds.
    :member dataset_len: The number of videos in the dataset.
    :member video_ids: A list of video IDs for videos with a duration greater than or equal to clip_duration.

    :method __init__: Initializes the dataset with the specified database path and clip duration.
    :method __len__: Returns the number of videos in the dataset.
    :method __getitem__: Loads a video clip from the dataset at the specified index.
    """
    def __init__(self, db_path, clip_duration=2.0):
        """
        Initializes the dataset with the specified database path and clip duration.

        :param db_path: The path to the SQLite database file.
        :param clip_duration: The duration of the video clips to load in seconds.
        """
        self.db_path = db_path
        self.clip_duration = clip_duration
        conn = sqlite3.connect('file:{}'.format(self.db_path), uri=True)
        cursor = conn.cursor()
        self.dataset_len = cursor.execute(
            'SELECT COUNT(*) FROM video WHERE duration >= ?', (self.clip_duration,)
        ).fetchone()[0]
        if self.dataset_len == 0:
            min_duration = cursor.execute('SELECT MIN(duration) FROM video').fetchone()[0]
            avg_duration = cursor.execute('SELECT AVG(duration) FROM video').fetchone()[0]
            raise ValueError(f'No videos of {self.clip_duration} seconds or longer. \
                             The shortest video is {min_duration} seconds. \
                             The average video duration is {avg_duration} seconds. \
                             Select a longer clip_duration.')
        self.video_ids = [id[0] for id in cursor.execute('SELECT id FROM video WHERE duration >= ?', (self.clip_duration,)).fetchall()]
        conn.close()

    def __len__(self):
        """
        Returns the number of videos in the dataset that meet the clip_duration requirement.
        """
        return self.dataset_len

    def __getitem__(self, idx):
        """
        Loads a video clip from the dataset at the specified index.

        The index is used to select a video ID from the list of video IDs.
        A random clip of the specified duration is extracted from the video.
        The remaining frames are discarded if the video is shorter than the clip duration.

        :param idx: The index of the video to load.
        :return: A tensor of shape (T, H, W, C) containing the video clip.
        """
        conn = sqlite3.connect('file:{}'.format(self.db_path), uri=True)
        cursor = conn.cursor()

        video_id = self.video_ids[idx]
        video = cursor.execute(
            'SELECT * FROM video WHERE id = ?', (video_id,)
        ).fetchone()
        if video is None:
            raise IndexError('Index out of range')

        # first column is the video id
        _, video_path, timestamp, duration, frame_rate, number_of_frames, resolution, camera_angle = video

        if not os.path.exists(video_path):
            raise FileNotFoundError(f'Video file not found: {video_path}')

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f'Error opening video file: {video_path}')

        # Start at a random frame within the video, or the first frame to avoid truncation errors
        # This technically biases the dataset towards the beginning of the video, but it's a small bias
        fps = cap.get(cv2.CAP_PROP_FPS) if frame_rate is None else frame_rate
        target_frame_count = int(self.clip_duration * fps)
        last_valid_frame = int((duration - self.clip_duration) * fps)
        start_frame_range = range(0, max(1, last_valid_frame))
        start_frame = np.random.choice(start_frame_range)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Clip the frames to the target duration
        frames = []
        for _ in range(target_frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            # Optional preprocessing here
            frames.append(frame)
        cap.release()
        data = torch.tensor(np.array(frames), dtype=torch.float32)

        # Load the hand labels for frames in the clip
        frame_numbers = range(start_frame, start_frame + target_frame_count)
        labels = cursor.execute(
            'SELECT * FROM hand JOIN frame on hand.id = frame.hand_landmarks WHERE frame_number IN ({}) AND video_id = ?'.format(','.join('?' * target_frame_count)), (*frame_numbers, video_id)).fetchall()
        labels = [json.loads(label[1]) for label in labels]

        conn.close()

        return data, labels
