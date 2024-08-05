import h5py
import torch
from torch.utils.data import Dataset

class HandMotionVideoDataset(Dataset):
    def __init__(self):
        with h5py.File('../hand_motions.h5', 'r') as db:
            self.num_samples = db.attrs['num_videos']

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        with h5py.File('../hand_motions.h5', 'r') as db:
            video_grp = db[str(idx)]
            frames = video_grp['frames'][:]
            labels = video_grp['labels'][:]
        # Optional preprocessing here
        return torch.from_numpy(frames), torch.from_numpy(labels)

if __name__ == '__main__':
    dataset = HandMotionVideoDataset()
    print(len(dataset))
    frames, labels = dataset[1]
    print(frames.shape, labels.shape)
