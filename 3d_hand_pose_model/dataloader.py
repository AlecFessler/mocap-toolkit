# © 2024 Alec Fessler
# MIT License
# See LICENSE file in the project root for full license information.

import h5py
import numpy as np
import os
import subprocess
import torch
from torch.utils.data import Dataset

class HandMotionVideoDataset(Dataset):
    def __init__(self, labels_path='../hand_motions_labels.h5', data_path='../hand_motions_videos/', clip_duration=2.0):
        self.labels_path = labels_path
        self.data_path = data_path
        self.clip_duration = clip_duration
        try:
            self.db = h5py.File(self.labels_path, 'r')
            self.db_video_resolution = self.db.attrs['video_resolution']
            self.valid_video_ids = []
            for video_id in self.db:
                video_grp = self.db[video_id]
                fps = video_grp.attrs['fps']
                num_frames = video_grp.attrs['num_frames']
                duration = num_frames / fps
                if duration >= self.clip_duration:
                    self.valid_video_ids.append(video_id)
            self.num_samples = len(self.valid_video_ids)
        except Exception as e:
            raise RuntimeError(f'Error loading labels file: {e}')
        finally:
            # close db connection so that it can be reopened in __getitem__
            # so each worker in DataLoader has its own connection
            if self.db is not None:
                self.db.close()
                self.db = None

    def __del__(self):
        if self.db is not None:
            self.db.close()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.db is None:
            self.db = h5py.File(self.labels_path, 'r')

        video_id = self.valid_video_ids[idx]
        video_path = os.path.join(self.data_path, f'{video_id}.mkv')
        video_grp = self.db[video_id]
        fps = video_grp.attrs['fps']
        num_frames = video_grp.attrs['num_frames']

        clip_length = int(self.clip_duration * fps)
        last_possible_start = num_frames - clip_length
        start_frame = np.random.randint(0, last_possible_start)
        end_frame = start_frame + clip_length
        start_time = start_frame / fps
        num_frames = end_frame - start_frame

        # ffmpeg command explained:
        # -ss: start time
        # -i: input file
        # -vf: video filter
        # -frames:v: number of frames to extract
        # -f: output format
        # -pix_fmt: pixel format
        # - : pipes the output to stdout
        ffmpeg_cmd = [
            'ffmpeg',
            '-ss', str(start_time),
            '-i', video_path,
            '-vf', 'format=rgb24',
            '-frames:v', str(num_frames),
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-'
        ]
        try:
            ffmpeg = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            raw_video, err = ffmpeg.communicate()
            if ffmpeg.returncode != 0:
                raise RuntimeError(f'FFmpeg error: {err.decode("utf-8")}')
        except Exception as e:
            raise RuntimeError(f'Error running ffmpeg command: {e}')
        finally:
            if ffmpeg:
                ffmpeg.terminate()
                try:
                    ffmpeg.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    ffmpeg.kill()
                    ffmpeg.wait()
        num_bytes = num_frames * self.db_video_resolution[0] * self.db_video_resolution[1] * 3
        assert len(raw_video) == num_bytes, f'Expected {num_bytes} bytes from ffmpeg output but got {len(raw_video)} bytes'
        # copy array to make writeable before passing to pytorch
        frames = np.copy(np.frombuffer(raw_video, dtype=np.uint8).reshape(num_frames, self.db_video_resolution[1], self.db_video_resolution[0], 3))

        labels = video_grp['labels'][start_frame:end_frame]

        # labels are already fp16
        return torch.from_numpy(frames).half(), torch.from_numpy(labels)

if __name__ == '__main__':
    dataset = HandMotionVideoDataset()
    print(f'Number of samples: {len(dataset)}')
    print(f'Shape of first sample: {dataset[0][0].shape}')
    print(f'Dtype of first sample: {dataset[0][0].dtype}')
    print(f'Shape of first sample labels: {dataset[0][1].shape}')
    print(f'Dtype of first sample labels: {dataset[0][1].dtype}')
