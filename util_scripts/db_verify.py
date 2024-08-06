import h5py

def verify_hdf5_file(db_path):
    with h5py.File(db_path, 'r') as db:
        # Print general attributes
        if 'version' in db.attrs:
            print(f"Version: {db.attrs['version']}")
        if 'description' in db.attrs:
            print(f"Description: {db.attrs['description']}")
        if 'num_videos' in db.attrs:
            print(f"Number of videos: {db.attrs['num_videos']}")
        if 'video_resolution' in db.attrs:
            print(f"Video resolution: {db.attrs['video_resolution']}")

        # Iterate over each video group
        for video_name in db:
            video_grp = db[video_name]
            print(f"\nVideo {video_name} information:")
            print(f"  FPS: {video_grp.attrs['fps']}")
            print(f"  Number of frames: {video_grp.attrs['num_frames']}")

            # Verify labels dataset
            if 'labels' in video_grp:
                labels_shape = video_grp['labels'].shape
                print(f"  Labels dataset shape: {labels_shape}")

if __name__ == '__main__':
    db_path = '../hand_motions_labels.h5'
    verify_hdf5_file(db_path)
