import streamlit as st
import os
import zipfile
import tempfile
import pandas as pd
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp

# --- Pose estimation helper ---
def extract_joint_angles(video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    joint_angles = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            def get_coords(idx): return np.array([landmarks[idx].x, landmarks[idx].y])

            left_ankle = get_coords(27)
            right_ankle = get_coords(28)
            left_foot_index = get_coords(29)
            right_foot_index = get_coords(30)
            left_foot = get_coords(31)
            right_foot = get_coords(32)
            left_heel = get_coords(33)
            right_heel = get_coords(34)
            
            left_hip = get_coords(23)
            right_hip = get_coords(24)
            left_knee = get_coords(25)
            right_knee = get_coords(26)
            left_ankle = get_coords(27)
            right_ankle = get_coords(28)

            # Vectors
            left_hip = left_hip - left_knee
            right_hip = right_hip - right_knee
            left_knee = left_knee - left_ankle
            right_knee = right_knee - right_ankle
            
            left_ankle = left_ankle - left_foot
            right_ankle = right_ankle - right_foot
            
            left_foot = left_foot - left_heel
            right_foot = right_foot - right_heel
            
            left_heel = left_heel - left_ankle
            right_heel = right_heel - right_ankle

            left_foot_index = left_foot_index - left_ankle
            right_foot_index = right_foot_index - right_ankle

            left_thigh = left_hip - left_knee
            left_shank = left_knee - left_ankle
            right_thigh = right_hip - right_knee
            right_shank = right_knee - right_ankle

            # Simple angle calculation (dot product formula)
            def calculate_angle(v1, v2):
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
                return angle
            
            left_ankle_angle = calculate_angle(left_foot, left_ankle)
            right_ankle_angle = calculate_angle(right_foot, right_ankle)

            left_hip_angle = calculate_angle(left_thigh, left_hip)
            right_hip_angle = calculate_angle(right_thigh, right_hip)

            left_knee_angle = calculate_angle(left_thigh, left_shank)
            right_knee_angle = calculate_angle(right_thigh, right_shank)

            joint_angles.append({
                "Frame": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                "Left Ankle Angle": left_ankle_angle,
                "Right Ankle Angle": right_ankle_angle,

                "Left Knee Angle": left_knee_angle,
                "Right Knee Angle": right_knee_angle,
                "Left Hip Angle": left_hip_angle,
                "Right Hip Angle": right_hip_angle

            })

    cap.release()
    return pd.DataFrame(joint_angles)

# --- Streamlit Interface ---
st.title("🎥 Batch Video Uploader and Pose Estimator")

uploaded_files = st.file_uploader(
    "Upload your .mov or .mp4 video files",
    type=["mov", "mp4"],
    accept_multiple_files=True
)

if uploaded_files:
    with tempfile.TemporaryDirectory() as temp_dir:
        result_dir = Path(temp_dir) / "results"
        result_dir.mkdir(parents=True, exist_ok=True)

        for uploaded_file in uploaded_files:
            video_path = result_dir / uploaded_file.name
            with open(video_path, "wb") as f:
                f.write(uploaded_file.read())

            st.info(f"Processing {uploaded_file.name}...")
            df_angles = extract_joint_angles(str(video_path))

            # Save each CSV
            csv_filename = video_path.stem + "_results.csv"
            csv_path = result_dir / csv_filename
            df_angles.to_csv(csv_path, index=False)

        # Zip all CSVs
        zip_path = Path(temp_dir) / "pose_results.zip"
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for csv_file in result_dir.glob("*.csv"):
                zipf.write(csv_file, arcname=csv_file.name)

        # Offer download
        with open(zip_path, "rb") as f:
            st.success("✅ Videos processed! Download your results below.")
            st.download_button(
                label="📦 Download Zipped Results",
                data=f,
                file_name="pose_results.zip",
                mime="application/zip"
            )

st.markdown('''
## Instructions ''')
st.write('''
            1. Drag and drop batch upload .mov or .mp4 files
            2. Process each video → Run pose estimation → Extract joint angles (Ankle, Knee, Hip, Spine))
            3. Export a CSV for each video individually
            4. Zip all the results automatically
            5. Download the zipped file containing all the results
            6. Unzip the file and open the CSV files in Excel
            7. Analyze the data and visualize the joint angles using your preferred tools (e.g., MATLAB, Python, R)

            have any questions or need assistance? Reach out to David Hollinger at dh25587@essex.ac.uk ''')

