import streamlit as st
import os
import zipfile
import tempfile
from pathlib import Path

st.title("🎞️ Batch Video Uploader and Zipper")

uploaded_files = st.file_uploader(
    "Drag and drop .mov or .mp4 files here",
    type=["mov", "mp4"],
    accept_multiple_files=True
)

if uploaded_files:
    with tempfile.TemporaryDirectory() as temp_dir:
        video_dir = Path(temp_dir) / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)

        # Save each uploaded file
        for uploaded_file in uploaded_files:
            file_path = video_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

        # Create zip file
        zip_path = Path(temp_dir) / "uploaded_videos.zip"
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for video_file in video_dir.iterdir():
                zipf.write(video_file, arcname=video_file.name)

        # Provide download button
        with open(zip_path, "rb") as f:
            st.download_button(
                label="📦 Download Zipped Videos",
                data=f,
                file_name="uploaded_videos.zip",
                mime="application/zip"
            )

st.text('TO DO: Process the videos then saved them individually into a zip file')
