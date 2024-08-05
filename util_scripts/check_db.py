import sqlite3 as sql
import json

# Connect to the database
conn = sql.connect("../hand_motion.db")
cursor = conn.cursor()

# Query for the video
cursor.execute("SELECT * FROM video")
videos_data = cursor.fetchall()

if videos_data:
    for video in videos_data:
        video_id, file_path, timestamp, duration, frame_rate, number_of_frames, resolution, camera_angle = video
        print(f"Video ID: {video_id}")
        print(f"File Path: {file_path}")
        print(f"Timestamp: {timestamp}")
        print(f"Duration: {duration} seconds")
        print(f"Frame Rate: {frame_rate} fps")
        print(f"Number of Frames: {number_of_frames}")
        print(f"Resolution: {resolution}")
        print(f"Camera Angle: {camera_angle}\n")

        # Iterate through each frame number
        for frame_number in range(number_of_frames):
            cursor.execute("SELECT * FROM frame WHERE video_id = ? AND frame_number = ?", (video_id, frame_number))
            frame = cursor.fetchone()

            if frame:
                frame_id, video_id, hand_landmarks_id, frame_number, frame_timestamp = frame
                # print(f"Frame Number: {frame_number}")
                # print(f"Timestamp: {frame_timestamp}")

                # Query the associated hand landmarks
                cursor.execute("SELECT hand_landmarks FROM hand WHERE id = ?", (hand_landmarks_id,))
                hand_data = cursor.fetchone()

                if hand_data:
                    hand_landmarks_json = hand_data[0]
                    hand_landmarks = json.loads(hand_landmarks_json)
                    # print(f"Hand Landmarks: {json.dumps(hand_landmarks, indent=2)}")
                else:
                    print("No hand landmarks found.")
            else:
                print(f"No data found for Frame Number: {frame_number}")

            # print("-" * 50)
else:
    print("No video found in the database.")

# Close the database connection
conn.close()
