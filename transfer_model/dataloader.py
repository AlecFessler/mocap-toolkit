import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class HandMotionVideoDataset(Dataset):
    def __init__(self, file_path='../hand_motions.h5', clip_duration=2.0):
        """
        Initialize the Hand Motion Video Dataset.

        The database connection is initially made just to gather the valid video IDs,
        and the number of samples, and then is closed until the first sample is requested.
        This is done to avoid issues with multiprocessing, ensuring that each process has
        it's own connection to the database.

        :param file_path: Path to the HDF5 file containing the hand motion videos
        :param clip_duration: Duration of the clips to extract from the videos
        """
        self.file_path = file_path
        self.clip_duration = clip_duration
        self.db = h5py.File(self.file_path, 'r')
        self.valid_video_ids = []
        for video_id in self.db:
            video_grp = self.db[video_id]
            fps = video_grp.attrs['fps']
            num_frames = video_grp.attrs['num_frames']
            duration = num_frames / fps
            if duration >= self.clip_duration:
                self.valid_video_ids.append(video_id)
        self.num_samples = len(self.valid_video_ids)
        self.db.close()
        self.db = None

    def __del__(self):
        if self.db is not None:
            self.db.close()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        The clip will be cut from a random location in the video,
        effectively increasing the number of samples that can be extracted
        from each video.

        :param idx: Index of the sample to retrieve
        """
        if self.db is None:
            self.db = h5py.File(self.file_path, 'r')

        video_grp = self.db[self.valid_video_ids[idx]]
        fps = video_grp.attrs['fps']
        num_frames = video_grp.attrs['num_frames']

        if num_frames < fps * self.clip_duration:
            # This should not happen if we have filtered out videos that are too short
            raise ValueError(f'Video {idx} is too short for {self.clip_duration} second clips')

        clip_length = int(self.clip_duration * fps)
        last_possible_start = num_frames - clip_length
        start_frame = np.random.randint(0, last_possible_start)
        end_frame = start_frame + clip_length

        frames = video_grp['frames'][start_frame:end_frame]
        labels = video_grp['labels'][start_frame:end_frame]

        return torch.from_numpy(frames), torch.from_numpy(labels)

if __name__ == '__main__':
    dataset = HandMotionVideoDataset()
    print(f'Number of samples: {len(dataset)}')
    print(f'Shape of first sample: {dataset[0][0].shape}')
    print(f'Shape of first sample labels: {dataset[0][1].shape}')
