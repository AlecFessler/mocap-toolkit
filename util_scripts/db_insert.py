import cv2
from datetime import datetime, timedelta
import mediapipe as mp
import sqlite3 as sql
import json
import os
import sys

def insert_video(video_path):
    """
    Insert a video into the database

    :param video_path: The path to the video file
    """
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Error opening video file")
        exit()

    with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as hands:
        video_file_path = os.path.abspath(video_path)
        video_timestamp = datetime.now()
        video_duration = 0 # count the number of frames and divide by the frame rate
        video_frame_rate = video.get(cv2.CAP_PROP_FPS)
        video_number_of_frames = 0 # count the number of frames
        video_resolution = f"{int(video.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
        video_camera_angle = "front"
        frame_timestamp = video_timestamp

        cursor.execute("""
        INSERT INTO video (file_path, timestamp, duration, frame_rate, number_of_frames, resolution, camera_angle)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (video_file_path, video_timestamp.strftime("%Y-%m-%d %H:%M:%S"), video_duration, video_frame_rate, video_number_of_frames, video_resolution, video_camera_angle))
        video_id = cursor.lastrowid

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                # Convert the hand landmarks to a JSON string
                hand_landmarks_json = {}
                for i, landmark in enumerate(hand_landmarks.landmark):
                    hand_landmarks_json[hand_landmarks_dict[list(hand_landmarks_dict.keys())[i]]] = {
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z
                    }
                hand_landmarks_json = json.dumps(hand_landmarks_json)

                # Save the hand landmarks
                cursor.execute("""
                INSERT INTO hand (hand_landmarks)
                VALUES (?)
                """, (hand_landmarks_json,))
                hand_id = cursor.lastrowid

                # Save the frame
                cursor.execute("""
                INSERT INTO frame (video_id, hand_landmarks, frame_number, timestamp)
                VALUES (?, ?, ?, ?)
                """, (video_id, hand_id, video_number_of_frames, frame_timestamp.strftime("%Y-%m-%d %H:%M:%S")))

                # Draw the hand landmarks
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(rgb_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # increment the frame number and timestamp
            video_number_of_frames += 1
            frame_duration = 1 / video_frame_rate
            video_duration += frame_duration
            frame_timestamp += timedelta(seconds=frame_duration)

            # Convert back to BGR
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Hand Pose Recognizer", bgr_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Save the final video duration and number of frames
    cursor.execute("""
    UPDATE video
    SET duration = ?,
    number_of_frames = ?
    WHERE id = ?
    """, (video_duration, video_number_of_frames, video_id))

    video.release()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python db_insert.py <video_path>")
        exit()

    # Connect to the database
    conn = sql.connect("../hand_motion.db")
    cursor = conn.cursor()

    # Create the tables if they don't exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS video (
    id INTEGER PRIMARY KEY,
    file_path TEXT,
    timestamp TEXT,
    duration REAL,
    frame_rate REAL,
    number_of_frames INTEGER,
    resolution TEXT,
    camera_angle TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS frame (
    id INTEGER PRIMARY KEY,
    video_id INTEGER,
    hand_landmarks INTEGER,
    frame_number INTEGER,
    timestamp TEXT,
    FOREIGN KEY (video_id) REFERENCES video (id),
    FOREIGN KEY (hand_landmarks) REFERENCES hand (id)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS hand (
    id INTEGER PRIMARY KEY,
    hand_landmarks TEXT
    )
    """)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    """
    cmc: Carpometacarpal joint -- the joint where the thumb meets the wrist
    mcp: Metacarpophalangeal joint -- the joint where fingers (and thumb) meet the palm
    ip: Interphalangeal joint -- the joint between the two segments of a finger (or thumb)
    pip: Proximal interphalangeal joint -- the second joint from the tip of a finger
    dip: Distal interphalangeal joint -- the joint nearest to the tip of a finger
    tip: The tip of a finger (or thumb)
    """
    hand_landmarks_dict = {
        "wrist": 0,
        "thumb_cmc": 1,
        "thumb_mcp": 2,
        "thumb_ip": 3,
        "thumb_tip": 4,
        "index_finger_mcp": 5,
        "index_finger_pip": 6,
        "index_finger_dip": 7,
        "index_finger_tip": 8,
        "middle_finger_mcp": 9,
        "middle_finger_pip": 10,
        "middle_finger_dip": 11,
        "middle_finger_tip": 12,
        "ring_finger_mcp": 13,
        "ring_finger_pip": 14,
        "ring_finger_dip": 15,
        "ring_finger_tip": 16,
        "pinky_mcp": 17,
        "pinky_pip": 18,
        "pinky_dip": 19,
        "pinky_tip": 20
    }

    video_path = sys.argv[1]
    # if the video path is a directory, insert all videos in the directory
    if os.path.isdir(video_path):
        for video_file in os.listdir(video_path):
            if video_file.endswith(("mp4", "avi", "mov", "mkv")):
                insert_video(os.path.join(video_path, video_file))
    # if the video path is a file, insert the video
    elif os.path.isfile(video_path):
        insert_video(video_path)
    else:
        print("Invalid video path")
        exit()

    cv2.destroyAllWindows()
    conn.commit()
    conn.close()
