

import streamlit as st

st.title('Welcome to 5G4PHealth!')

import cv2
import mediapipe as mp
from mediapipe import solutions
import numpy as np
import tempfile
import os
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import pandas as pd
from scipy.signal import butter, lfilter
from sklearn.decomposition import PCA
from scipy.signal import find_peaks
import tempfile
import requests
from io import BytesIO
from fpdf import FPDF
import matplotlib.colors as mcolors
from PIL import Image, ImageOps
from datetime import datetime
import qrcode
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from email.message import EmailMessage
from dotenv import load_dotenv
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Sample data: replace with your actual model scores
# categories = ['Code', 'Factuality', 'Reasoning', 'Science', 'Multilingual', 'Vision']
# N = len(categories)

# # Example model scores
# data_gemma2 = [40, 65, 50, 55, 30, 20]
# data_gemma3 = [95, 90, 85, 85, 85, 90]

# # Repeat the first value to close the radar chart
# data_gemma2 += data_gemma2[:1]
# data_gemma3 += data_gemma3[:1]
# angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
# angles += angles[:1]

# # Plotting
# fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# ax.plot(angles, data_gemma2, color='red', linewidth=2, label='Gemma 2')
# ax.fill(angles, data_gemma2, color='red', alpha=0.25)

# ax.plot(angles, data_gemma3, color='blue', linewidth=2, label='Gemma 3')
# ax.fill(angles, data_gemma3, color='blue', alpha=0.25)

# ax.set_yticks([20, 40, 60, 80, 100])
# ax.set_yticklabels(["20", "40", "60", "80", "100"])
# ax.set_xticks(angles[:-1])
# ax.set_xticklabels(categories, fontsize=12)
# ax.set_title('Model Performance Comparison', fontsize=16, pad=20)
# ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))

# st.pyplot(fig)


# TO DO

# WALKING
# - Add an upload side and back walking video
# - Update the spider plot ranges for walking
# - Update the asymmetry plot for walking
# - Update the text insights for walking
# - Update the joint target analysis for walking
# 
# GENERAL
# - Fix when video is uploaded 90 deg sideways (gives wrong results, uh oh!) --> it should be vertical, not landscape recording
# - Try to merge the side and back videos into one report (if feasible)
# - Add more personliazed insights based on the data (text and exercise recommendations as decision trees)
class CustomPDF(FPDF):
    def header(self):
        # Set black background on every page automatically
        self.set_fill_color(0, 0, 0)
        self.rect(0, 0, 210, 297, 'F')
        
def generate_pdf(pose_image_path, df_rom, spider_plot, asymmetry_plot, text_info, camera_side, gait_type):
    """Generates a PDF with the pose estimation, given plots, and text. FPDF document (A4 size, 210mm width x 297mm height)"""
    pdf = CustomPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ✅ Add Date and Location (Top Left)
    pdf.set_text_color(255, 255, 255)  # White text
    pdf.set_font("Arial", size=9)  # Small font
    current_date = datetime.today().strftime("%m/%d/%Y")  # Automatically fetch today's date
    location_text = f"Date: {current_date}\nLocation: Tri N Run Mobile\nGait Type: {gait_type.capitalize()}"
    pdf.multi_cell(0, 3.5, location_text)  # Multi-line cell to properly format text

    # ✅ Report Title (Centered)
    pdf.set_xy(10, 10)  # Reset cursor
    pdf.set_font("Arial", style='BU', size=20)
    pdf.cell(190, 10, "Your Stride Sync Report", ln=True, align='C')

    # add logo in the top right corner
    github_url = "https://raw.githubusercontent.com/dholling4/PolarPlotter/main/"
    logo_path = github_url + "logo/stride sync logo.png"
    logo = requests.get(logo_path)
    logo_img = Image.open(BytesIO(logo.content))
    logo_img_path = tempfile.mktemp(suffix=".png")
    logo_img.save(logo_img_path)
    pdf.image(logo_img_path, x=165, y=10, w=30)  # Adjusted placement


    pdf.ln(10)  # Spacing before the next section

    # Add padding to the image
    if pose_image_path:
        pose_img = Image.open(pose_image_path)
        width, height = pose_img.size

        # Create a new image with padding
        padded_img = ImageOps.expand(pose_img, border=(0, 1, 0, 1), fill=(0, 0, 0))  # Add black padding
        padded_pose_path = tempfile.mktemp(suffix=".png")
        padded_img.save(padded_pose_path)

        # 🔹 Reduce image size in the PDF
        pdf.image(padded_pose_path, x=10, y=25, h=88, w=49)  # Make it smaller (1/8 of the page)

    # ✅ Spider Plot (Top Right)
    spider_plot_path = tempfile.mktemp(suffix=".png")
    spider_plot.update_layout(paper_bgcolor="black", font_color="white") 
    spider_plot.write_image(spider_plot_path)
    pdf.image(spider_plot_path, x=75, y=31, w=125)  # Adjusted placement

    pdf.ln(40)  # Increase spacing before middle section

    # ✅ Asymmetry Plot (Middle Left)
    asymmetry_plot_path = tempfile.mktemp(suffix=".png")
    asymmetry_plot.update_layout(paper_bgcolor="black", plot_bgcolor="black", font_color="white")
    asymmetry_plot.write_image(asymmetry_plot_path)
    pdf.image(asymmetry_plot_path, x=10, y=125, w=125)  # Placed on the left

    pdf.ln(5)  # Extra spacing before next plot

    # ✅ Generate Styled ROM Table (Middle Right)
    rom_chart_path = tempfile.mktemp(suffix=".png")
    fig, ax = plt.subplots(figsize=(4.5, 2.3))  # Adjust size
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df_rom.values, colLabels=df_rom.columns, cellLoc='center', loc='center')
    # table.auto_set_font_size(True)
    # table.auto_set_column_width([0, 1, 2, 3])  # Adjust column width
    # increase size of the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.25, 1.25)  # Increase size by 1.5
    table.auto_set_column_width([0, 1, 2, 3])  # Adjust column width

    # Define ranges for color classification
    # Define ranges for color classification
    if camera_side == "side" and gait_type == "walking": 
        ankle_good = (20, 45)
        ankle_moderate = (15, 20)
        ankle_bad = (0, 55)#(0, 10)

        knee_good = (50, 70)
        knee_moderate = (40, 50)
        knee_bad = (0, 80) #(0, 40)

        hip_good = (25, 45)
        hip_moderate = (15, 25)
        hip_bad = (0, 15)

        spine_good = (0, 5)
        spine_moderate = (5, 10)
        spine_bad = (10, 30)

    if camera_side == "back" and gait_type == "walking": 
        ankle_good = (20, 50)
        ankle_moderate = (15, 20)
        ankle_bad = (0, 15)

        knee_good = (0, 5)
        knee_moderate = (5, 10)
        knee_bad = (10, 30)

        hip_good = (0, 10)
        hip_moderate = (10, 15)
        hip_bad = (15, 50)

        spine_good = (0, 5)
        spine_moderate = (5, 10)
        spine_bad = (10, 30)

    if camera_side == "side" and gait_type == "running": 
        ankle_good = (65, 75)
        ankle_moderate = (55, 85)
        ankle_bad = (55, 95)

        knee_good = (120, 130)
        knee_moderate = (90, 175)
        knee_bad = (90, 175)

        hip_good = (60, 70)
        hip_moderate = (40, 90)
        hip_bad = (40, 90)

        spine_good = (5, 15)
        spine_moderate = (2, 20)
        spine_bad = (0, 30)

    if camera_side == "back" and gait_type == "running":
        ankle_good = (20, 60)
        ankle_moderate = (15, 20)
        ankle_bad = (0, 15)

        knee_good = (0, 5)
        knee_moderate = (5, 12)
        knee_bad = (12, 30)

        hip_good = (0, 10)
        hip_moderate = (10, 20)
        hip_bad = (20, 40)

        spine_good = (1, 10)
        spine_moderate = (10, 20)
        spine_bad = (20, 30)

    def get_color(value, good_range, moderate_range):
        """Assigns a color based on the ROM classification."""
        if good_range[0] <= value <= good_range[1]:
            # return a light green color
            return "lightgreen"
        
        elif moderate_range[0] <= value <= moderate_range[1]:
            return 'yellow'
        else:
            return "lightcoral"

    # Apply colors to the first and last columns
    for i, joint in enumerate(df_rom['Joint']):
        if joint == 'Spine Segment':
            good_range, moderate_range = spine_good, spine_moderate
        elif 'Hip' in joint:
            good_range, moderate_range = hip_good, hip_moderate
        elif 'Knee' in joint:
            good_range, moderate_range = knee_good, knee_moderate
        elif 'Ankle' in joint:
            good_range, moderate_range = ankle_good, ankle_moderate

        rom_value = df_rom['Range of Motion (°)'].iloc[i]
        color = get_color(rom_value, good_range, moderate_range)

        # Change color of the first column (Joint names)
        cell = table[(i + 1, 0)]
        cell.set_text_props(color=color)

        # Change color of the last column (Range of Motion)
        cell = table[(i + 1, len(df_rom.columns) - 1)]
        cell.set_text_props(color=color)

        # change color of 2nd, 3rd columns
        cell = table[(i + 1, 1)]
        cell.set_text_props(color="white")
        cell = table[(i + 1, 2)]
        cell.set_text_props(color="white")

        for i in range(4):
            cell = table[(0, i)]
            cell.set_text_props(color="white", weight='bold')     

    # Set the background color of the chart to black
    for key, cell in table._cells.items():
        cell.set_edgecolor("white")
        # cell.set_text_props(color=color, weight='bold')
        cell.set_facecolor("black")

    plt.savefig(rom_chart_path, bbox_inches='tight', dpi=300, facecolor='black') 
    plt.close(fig)
    
    # Place ROM Table
    pdf.image(rom_chart_path, x=10, y=190, w=130) 

    pdf.ln(155)  # Adjust based on vertical layout
   
    joint_color_map = {
        "spine": (200, 162, 200),  # Purple
        "left hip": (144, 238, 144),    # Green
        "right hip": (144, 238, 144),    # Green
        "left knee": (173, 216, 230),   # Blue
        "right knee": (173, 216, 230),   # Blue
        "left ankle": (255, 182, 193),   # Red
        "right ankle": (255, 182, 193)   # Red
    }
    font_size=12

    for joint in ["spine segment summary", "left hip summary", "right hip summary", "left knee summary", "right knee summary", "left ankle summary", "right ankle summary"]:
        summary = text_info.get(joint, "")
        if summary:
            color = (255, 255, 255)
            pdf.set_text_color(*color)
            pdf.set_font("Arial", style='B', size=font_size)
            label = joint.title() + ": "
            pdf.write(font_size / 2, label)
            pdf.set_font("Arial", size=font_size)
            pdf.write(font_size / 2, summary + "\n")
            pdf.ln(1)    

    pdf.ln(200)  # Spacing before bottom text section
    
    pdf.set_text_color(255, 215, 0)  # Gold Text for Highlights
    pdf.set_font("Arial", style='B', size=14)
    pdf.cell(0, 10, "Key Insights from Your Gait", ln=True)
    
    for joint in ["spine", "left hip", "right hip", "left knee", "right knee", "left ankle", "right ankle"]:
        insight = text_info.get(joint, "")
        if insight:
            color = joint_color_map[joint]
            pdf.set_text_color(*color)
            pdf.set_font("Arial", style='B', size=font_size)
            label = joint.title() + ": "
            pdf.write(font_size / 2, label)
            pdf.set_font("Arial", size=font_size)
            pdf.write(font_size / 2, insight + "\n")
            pdf.ln(2)
        
    spine_text = '''A key indicator of your posture and alignment. A consistent angle of about 5-15 degrees throughout your stride is ideal, and any significant deviations may indicate potential issues with your core stability or posture.'''
    hip_text = '''A critical joint for power generation and stability. A consistent angle of about 30-50 degrees throughout your stride is ideal, and any significant deviations may indicate potential issues with your hip flexor or glute strength.'''
    knee_text = 'A key joint for shock absorption and propulsion. A consistent angle of about 160-180 degrees at heel strike and 120-140 degrees at toe-off is ideal, and any significant deviations may indicate potential issues with your quadriceps or hamstrings.'
    ankle_text = 'Plays an essential roll for push-off and stability. A consistent angle of about 90-100 degrees at heel strike and 20-30 degrees at toe-off is ideal, and any significant deviations may indicate potential issues with your calf or Achilles tendon.'

    pdf.ln(5)

    pdf.set_text_color(255, 215, 0)  # Gold for Header
    pdf.set_font("Arial", 'b', size=14)
    pdf.cell(0, 10, "Joint Target Analysis", ln=True)

    joint_targets = {
    "Spine Segment Angle": {
        "text": spine_text,
        "color": (200, 162, 200)  # Purple
    },
    "Hips": {
        "text": hip_text,
        "color": (144, 238, 144)  # Green
    },
    "Knees": {
        "text": knee_text,
        "color": (173, 216, 230)  # Blue
    },
    "Ankles": {
        "text": ankle_text,
        "color": (255, 182, 193)  # Red
    }
}

    for joint_label, data in joint_targets.items():
        pdf.set_text_color(*data["color"])
        pdf.set_font("Arial", style='B', size=font_size)
        pdf.write(font_size / 2, f"{joint_label}: ")

        pdf.set_font("Arial", size=font_size)
        pdf.write(font_size / 2, data["text"] + "\n")
        pdf.ln(1)

    pdf.ln(10)

    # ✅ Invitation to Optional Coaching Session
    coaching_invite = "Want to Take Your Running to the Next Level? Consider scheduling an advanced gait analysis or personalized coaching session. Our expert team can help fine-tune your stride, optimize efficiency, and reduce injury risk."
    pdf.set_text_color(255, 215, 0)  # Gold color for the title
    pdf.set_font("Arial", style='B', size=14)  # Bold and slightly larger
    pdf.cell(0, 10, "Coaching & Gait Review", ln=True)

    pdf.set_text_color(255, 255, 255)  # White text for readability
    pdf.set_font("Arial", size=font_size)
    pdf.multi_cell(0, 7, coaching_invite)

    # Highlight Contact Info with Bigger, Bold White Text
    pdf.set_text_color(255, 255, 255)  # Bright green for attention
    pdf.set_font("Arial", style='B', size=13)  # Bigger and bold

    pdf.ln(5)

    pdf.cell(0, 10, "Contact: digitalathlete80@gmail.com", ln=True)

    pdf.set_text_color(255, 255, 255)  # Bright red for the email
    pdf.set_font("Arial", style='B', size=13)
    pdf.cell(0, 10, "Website: stride-sync.streamlit.app", ln=True)
    pdf.set_text_color(255, 255, 255)  # Bright red for the email
    pdf.set_font("Arial", style='B', size=13)
    pdf.cell(0, 10, "Scan the QR Code for recommended training videos", ln=True)
    pdf.ln(10)

    # ✅ Add a QR Code for the Website
    qr_code_url = "https://stride-sync.streamlit.app"
    qr_code_path = tempfile.mktemp(suffix=".png")
    qr_code = qrcode.make(qr_code_url)
    qr_code.save(qr_code_path)
    pdf.image(qr_code_path, x=160, y=265, w=30)

    # ✅ Save PDF
    pdf_file_path = tempfile.mktemp(suffix=".pdf")
    pdf.output(pdf_file_path)
    
    return pdf_file_path


def detect_peaks(data, column, prominence, distance):
    peaks, _ = find_peaks(data[column], prominence=prominence, distance=distance)
    return peaks

def detect_mins(data, column, prominence, distance):
    mins, _ = find_peaks(-data[column], prominence=prominence, distance=distance)
    return mins

def compute_stats(data, peaks, column):
    cycle_stats = []
    for i in range(len(peaks) - 1):
        cycle_data = data[column][peaks[i]:peaks[i + 1]]
        cycle_stats.append({
            "Cycle": i + 1,
            "Mean": np.mean(cycle_data),
            "Std Dev": np.std(cycle_data),
            "Max": np.max(cycle_data),
            "Min": np.min(cycle_data)
        })
    return pd.DataFrame(cycle_stats)

# Setup MediaPipe Pose model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

KEYPOINTS_OF_INTEREST = {
    23: "Left Hip",
    24: "Right Hip",
    25: "Left Knee",
    26: "Right Knee",
    27: "Left Ankle",
    28: "Right Ankle",
    29: "Left Heel",
    30: "Right Heel",
    31: "Left Foot",
    32: "Right Foot"
}

def process_first_frame_report(video_path, video_index):
    """Use pose estimation overlay for generate pdf report."""
    neon_green = (57, 255, 20)
    cool_blue = (0, 91, 255)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps

    # ➕ Check if video is rotated based on first frame
    ret, test_frame = cap.read()
    if not ret:
        raise ValueError("Couldn't read from video.")
    
    rotated = False
    if test_frame.shape[0] > test_frame.shape[1]:  # height > width → probably rotated
        rotated = True
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset back to start

    # If the video is longer than 10 seconds, capture only the middle 5 seconds
    if duration > 10:
        start_frame = total_frames // 2 - (5 * fps)
        end_frame = total_frames // 2 + (5 * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        total_frames = int(end_frame - start_frame)
        duration = total_frames / fps

    elif duration <= 10:
        start_frame = 0  # Start from the beginning
        end_frame = total_frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        total_frames = int(end_frame - start_frame)
        duration = total_frames / fps

    frame_number = start_frame

    time = frame_number / fps

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    ret, frame = cap.read()
    if rotated:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    if not ret:
        st.error("Failed to read the selected frame.")
        cap.release()
        return None, None, None  # Return None if no valid frame

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            annotated_frame = frame.copy()
            mp_drawing.draw_landmarks(
                annotated_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=solutions.drawing_styles.DrawingSpec(color=neon_green, thickness=10, circle_radius=7),
                connection_drawing_spec=solutions.drawing_styles.DrawingSpec(color=cool_blue, thickness=10)
            )

            # Save the processed frame as an image
            image_path = tempfile.mktemp(suffix=".png")
            cv2.imwrite(image_path, annotated_frame)
            
            # st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption=f"Frame {frame_number}")
            
            cap.release()
            return frame_number, time, image_path  # Return image path

    cap.release()
    return None, None, None

def process_first_frame(video_path, video_index):
    """Processes the first frame and returns the frame number, time, and saved image path."""

    neon_green = (57, 255, 20)
    cool_blue = (0, 91, 255)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps

    # ➕ Check if video is rotated based on first frame
    ret, test_frame = cap.read()
    if not ret:
        raise ValueError("Couldn't read from video.")

    rotated = False

    if test_frame.shape[0] > test_frame.shape[1]:  # height < width → probably rotated
        rotated = True

    # If the video is longer than 10 seconds, capture only the middle 5 seconds
    # if duration > 10:
    duration = total_frames / fps

    st.write(f"Total frames: {total_frames}, FPS: {fps:.1f}, Duration: {duration:.2f} seconds")
    default_frame = min(5, total_frames - 1)  # Ensure default frame is within range
    # frame_number_selected = st.slider(f"Select frame for video ({video_index+1})", 0, total_frames - 1, 5, key=f"frame_{video_index}_{video_path}_{hash(video_path)}")
    frame_number_selected = st.slider("Select video frame", 0, total_frames - 1, default_frame, key=f"frame_{video_index}_{video_path}_{hash(video_path)}")
    time = frame_number_selected / fps

    st.write(f'Frame Number: {frame_number_selected} | Time: {time:.2f} sec')
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number_selected)
    
    ret, frame = cap.read()
    if rotated:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if not ret:
        st.error("Failed to read the selected frame.")
        cap.release()
        return None, None, None  # Return None if no valid frame

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            annotated_frame = frame.copy()
            mp_drawing.draw_landmarks(
                annotated_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=solutions.drawing_styles.DrawingSpec(color=neon_green, thickness=10, circle_radius=7),
                connection_drawing_spec=solutions.drawing_styles.DrawingSpec(color=cool_blue, thickness=10)
            )
            
            # Save the processed frame as an image
            image_path = tempfile.mktemp(suffix=".png")
            cv2.imwrite(image_path, annotated_frame)            
            st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption=f"Frame {frame_number_selected}")
            cap.release()
    return frame_number_selected, time, image_path  # Return image path
    # cap.release()
    # return None, None, None


def calculate_angle(v1, v2):
    """
    Calculate the angle between two vectors using the dot product.
    """
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    angle_radians = np.arccos(dot_product / (magnitude_v1 * magnitude_v2))
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees

###
def plot_joint_angles(time, angles, label, frame_time):
    fig = go.Figure()
    
    # Add the joint angle curve
    fig.add_trace(go.Scatter(x=time, y=angles, mode='lines', name=label))
    
    # Add vertical line for selected frame
    fig.add_trace(go.Scatter(
        x=[frame_time, frame_time],
        y=[min(angles), max(angles)],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Selected Frame'
    ))
    
    fig.update_layout(
        title=f"{label} Joint Angles",
        xaxis_title="Time (s)",
        yaxis_title="Angle (degrees)"
    )
    
    st.plotly_chart(fig)


def perform_pca(df, video_index):
    st.write("### Principal Component Analysis (PCA)")

    # Extract numerical joint angle data
    X = df.iloc[:, 1:].values
    
    # User selects number of principal components
    pcs = st.slider('Select the number of Principal Components:', 1, min(30, X.shape[1]), 3)
    st.write(f"Number of Principal Components Selected: {pcs}")
    
    # Perform PCA
    pca = PCA(n_components=pcs)
    principal_components = pca.fit_transform(X)

    # Explained variance
    explained_variance = pca.explained_variance_ratio_ 
    cumulative_variance = np.cumsum(explained_variance) 

    # dataframe for explained variance
    pca_df = pd.DataFrame({
        "Principal Component": [f"PC{i+1}" for i in range(len(explained_variance))],
        "Explained Variance (%)": explained_variance * 100,
        "Cumulative Variance (%)": cumulative_variance * 100
    })

    # Get absolute loadings (importance of each feature in each PC)
    loadings = np.abs(pca.components_)

    # Get top contributing features for each PC
    feature_labels = ["Left Hip", "Right Hip", "Left Knee", "Right Knee", "Left Ankle", "Right Ankle", "Spine Angle"]

    top_features_per_pc = []
    for i in range(pcs):
        top_feature_idx = np.argsort(-loadings[i])  # Sort in descending order
        top_features_per_pc.append([feature_labels[j] for j in top_feature_idx])

    # Create DataFrame
    pca_feature_df = pd.DataFrame(top_features_per_pc, index=[f"PC{i+1}" for i in range(pcs)])
    pca_feature_df.columns = [f"Rank {i+1}" for i in range(len(feature_labels))]  # Rank features

        
    top_features_per_pc = []
    for i in range(pcs):
        top_feature_idx = np.argsort(-loadings[i])  # Sort in descending order
        top_features_per_pc.append([feature_labels[j] for j in top_feature_idx])

    # Create DataFrame
    pca_feature_df = pd.DataFrame(top_features_per_pc, 
                                index=[f"PC{i+1}" for i in range(pcs)])
    pca_feature_df.columns = [f"Rank {i+1}" for i in range(len(feature_labels))]  # Rank features
    top_features = pca_feature_df

    fig = go.Figure()
    for i, feature in enumerate(top_features.iloc[:, 0]):  # Use only the top contributing feature
        fig.add_trace(go.Bar(
            x=[f"PC{i+1} ({feature})"],  # Label PC with the top feature
            y=[explained_variance[i] * 100],
            name=f"PC{i+1} ({feature})"
        ))

    explained_variance = pca.explained_variance_ratio_ 
    cumulative_variance = np.cumsum(explained_variance) 
    feature_labels = ["Left Hip", "Right Hip", "Left Knee", "Right Knee", "Left Ankle", "Right Ankle", "Spine Angle"]
    loadings = np.abs(pca.components_)
    top_features_ = [feature_labels[np.argmax(loadings[i])] for i in range(pcs)]

    pca_df = pd.DataFrame({
        "Principal Component": [f"PC{i+1} ({top_features_[i]})" for i in range(len(explained_variance))],
        "Explained Variance (%)": explained_variance * 100,
        "Cumulative Variance (%)": cumulative_variance * 100,
    })

   
    fig.add_trace(go.Scatter(
        x=[f"PC{i+1} ({feature})" for i, feature in enumerate(top_features.iloc[:, 0])],
        y=cumulative_variance * 100,
        mode="lines+markers",
        name="Cumulative Variance (%)",
        line=dict(color='red', dash="dash")
    ))

    fig.update_layout(
        title="Explained Variance with Top Contributing Feature",
        xaxis_title="Principal Components",
        yaxis_title="Explained Variance (%)",
        legend_title="Legend"
    )
    st.plotly_chart(fig)
    # download plot data 
    pca_plot_csv = pca_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download PCA Plot Data",
        data=pca_plot_csv,
        file_name="pca_plot_data.csv",
        mime="text/csv",
        key=f"pca_plot_{video_index}"
    )

    # st.dataframe(top_features)

    st.dataframe(pca_df)

    # combine the two dataframes and download
    pca_data = pd.concat([pca_df, top_features], axis=1)
    pca_feature_csv = pca_data.to_csv(index=False).encode('utf-8')

  
    # 2D PCA Scatter Plot (Only if at least 2 PCs are selected)
    if pcs >= 2:
        fig_2d = go.Figure()
        fig_2d.add_trace(go.Scatter(
            x=principal_components[:, 0],
            y=principal_components[:, 1],
            mode='markers',
            marker=dict(size=6, color=df["Time"], colorscale='Blues', showscale=True, colorbar=dict(title="Time", tickmode="array", tickvals=[df["Time"].min(), df["Time"].max()], ticktext=["Start", "End"])),
            text=df["Time"]
        ))

        fig_2d.update_layout(title="PCA Projection (2D)", xaxis_title="PC1", yaxis_title="PC2")
        st.plotly_chart(fig_2d)

        # download plot button
        pca_2d_csv = pd.DataFrame(principal_components[:, :2], columns=[f"PC{i+1}" for i in range(2)]).to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="Download 2D PCA Data",
            data=pca_2d_csv,
            file_name="pca_2d_data.csv",
            mime="text/csv",
            key=f"pca_2d_{video_index}"
        )

    # 3D PCA Scatter Plot (Only if at least 3 PCs are selected)
    if pcs >= 3:
        fig_3d = go.Figure(data=[go.Scatter3d(
            x=principal_components[:, 0], 
            y=principal_components[:, 1], 
            z=principal_components[:, 2],
            mode='markers', 
            marker=dict(size=4, color=df["Time"], colorscale='Blues', showscale=True, colorbar=dict(title="Time", tickmode="array", tickvals=[df["Time"].min(), df["Time"].max()], ticktext=["Start", "End"])),
            text=df["Time"]
        )])
        fig_3d.update_layout(title="PCA Projection (3D)",
                             scene_xaxis_title="PC1",
                             scene_yaxis_title="PC2",
                             scene_zaxis_title="PC3")
        st.plotly_chart(fig_3d)
        # download plot button
        pca_3d_csv = pd.DataFrame(principal_components, columns=[f"PC{i+1}" for i in range(pcs)]).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download 3D PCA Data",
            data=pca_3d_csv,
            file_name="pca_3d_data.csv",
            mime="text/csv",
            key=f"pca_3d_{video_index}"
        )

def plot_asymmetry_bar_chart(left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle):
    # Calculate the range of motion differences (right - left)
    hip_asymmetry = right_hip - left_hip
    knee_asymmetry = right_knee - left_knee
    ankle_asymmetry = right_ankle - left_ankle
    
    # Create a dictionary to hold the values for each joint
    asymmetry_data = {
        "Ankle": ankle_asymmetry,
        "Knee": knee_asymmetry,
        "Hip": hip_asymmetry
    }

    # Set thresholds for excessive asymmetry
    threshold = 10  # degrees

    # Create a color scale based on the absolute difference
    colors = []
    for value in asymmetry_data.values():
        abs_value = abs(value)  # Use absolute value to determine the color
        if abs_value > threshold:
            colors.append('red')  # If the absolute difference is larger than threshold, color red
        else:
            colors.append('green')  # If the difference is smaller, color green
    
    # Create the plot
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=list(asymmetry_data.keys()),
        x=list(asymmetry_data.values()),
        orientation='h',
        marker=dict(
            color=[abs(value) for value in asymmetry_data.values()],  # Color by absolute difference
            colorscale='RdYlGn',  # Red to Green color scale, but will reverse it to make higher values red
            colorbar=dict(title="Asymmetry (°)"),  # Add colorbar
            cmin=0,  # Minimum value for color scale
            cmax=40,  # Maximum value for color scale
            reversescale=True,  # Reverse the color scale
            colorbar_tickfont=dict(size=18)
        ),
        name="Left vs Right Asymmetry",
        text=[f"{value:.1f}°" for value in asymmetry_data.values()],  # Add text labels
        textfont=dict(size=16),  # Increase font size for text labels
        textposition='outside'  # Position text labels outside the bars
    ))

    fig.update_layout(
        title="Range of Motion",
        # increaes title fontsize
        title_font_size=42,
        xaxis_title="← Left Asymmetry (°)           Right Asymmetry (°) →",
        xaxis_title_font_size=22,
        yaxis_title="",
        showlegend=False,
        xaxis=dict(
            zeroline=True,
            zerolinecolor="white",
            zerolinewidth=2,
            range=[-30, 30],  # Fixed range from -30 to 30 for the x-axis
            tickvals=[-30, -20, -10, 0, 10, 20, 30],  # Tick labels for the fixed range
            ticktext=["-30", "-20", "-10", "0", "10", "20", "30"],  # Custom tick labels
            tickfont=dict(size=22)  # Increase tick font size

        ),
        yaxis=dict(tickvals=[0, 1, 2], ticktext=["Ankle", "Knee", "Hip"], tickfont=dict(size=22)),
        height=310,  # Shorten the graph height
        bargap=0.1  # Reduce the gap between bars to make them thinner
    )

    return fig

# Butterworth lowpass filter functions
def butter_lowpass_filter(data, cutoff=6, fs=30, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

def process_video(gait_type, camera_side, video_path, output_txt_path, frame_time, video_index):
    # add after uploading 
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps  

    # ➕ Check if video is rotated based on first frame
    ret, test_frame = cap.read()
    if not ret:
        raise ValueError("Couldn't read from video.")

    rotated = False
    if test_frame.shape[0] > test_frame.shape[1]:  # height < width → probably rotated
        rotated = True
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset back to start
    ### done add after uploading
  
    # start_frame = 0
    # end_frame = total_frames
    
    left_knee_angles, right_knee_angles = [], []
    left_hip_angles, right_hip_angles = [], []
    left_ankle_angles, right_ankle_angles = [], []
    spine_segment_angles = []
    thorax_angles, lumbar_angles = [], []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # if the length is greater than 10 seconds, only capture the middle N seconds
    # start_frame_crop = int(total_frames // 2 - (5 * fps))
    # end_frame_crop = int(total_frames // 2 + (5 * fps))
    # else capture the whole video
    # if duration > 10:
    # start_frame_crop = int(total_frames // 2 - (5 * fps))
    # end_frame_crop = int(total_frames // 2 + (5 * fps))
    # cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_crop)
    # total_frames = int(end_frame_crop - start_frame_crop)
    duration = total_frames / fps
    # else:
    start_frame_crop = 0
    end_frame_crop = total_frames
 
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_crop)
    total_frames = int(end_frame_crop - start_frame_crop)
    duration = total_frames / fps

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for _ in range(start_frame_crop, end_frame_crop):
            ret, frame = cap.read()
            if rotated:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                def get_coords(landmark):
                    return np.array([landmark.x, landmark.y])
                
                left_shoulder = get_coords(landmarks[11])
                right_shoulder = get_coords(landmarks[12])                 
                
                left_hip = get_coords(landmarks[23])
                right_hip = get_coords(landmarks[24])
                left_knee = get_coords(landmarks[25])
                right_knee = get_coords(landmarks[26])
                left_ankle = get_coords(landmarks[27])
                right_ankle = get_coords(landmarks[28])
                left_foot = get_coords(landmarks[31])
                right_foot = get_coords(landmarks[32])

                # midpoint of trunk vector
                shoulder_mid = (left_shoulder + right_shoulder) / 2
                hip_mid = (left_hip + right_hip) / 2
                
                trunk_vector = shoulder_mid - hip_mid

                # Upward vertical in image coordinates
                vertical_vector = np.array([0, -1])  
                left_trunk_vector = left_shoulder - left_hip
                right_trunk_vector = right_shoulder - right_hip
                left_thigh_vector = left_hip - left_knee
                left_shank_vector = left_knee - left_ankle
                right_thigh_vector = right_hip - right_knee
                right_shank_vector = right_knee - right_ankle
                left_foot_vector = left_ankle - left_foot
                right_foot_vector = right_ankle - right_foot

                # spine segment angle
                spine_segment_angles.append(calculate_angle(trunk_vector, vertical_vector))                
                left_hip_angles.append(calculate_angle(left_trunk_vector, left_thigh_vector))
                right_hip_angles.append(calculate_angle(right_trunk_vector, right_thigh_vector))
                left_knee_angles.append(calculate_angle(left_thigh_vector, left_shank_vector))
                right_knee_angles.append(calculate_angle(right_thigh_vector, right_shank_vector))
                left_ankle_angles.append(calculate_angle(left_shank_vector, left_foot_vector))
                right_ankle_angles.append(calculate_angle(right_shank_vector, right_foot_vector))

    time = np.arange(0, len(left_hip_angles)) / fps # Time in seconds  
    cap.release()

    # Apply lowpass filter to smooth angles
    cutoff_frequency = 6  # Adjust cutoff frequency based on signal characteristics
    left_hip_angles = butter_lowpass_filter(left_hip_angles, cutoff_frequency, fps)
    right_hip_angles = butter_lowpass_filter(right_hip_angles, cutoff_frequency, fps)
    left_knee_angles = butter_lowpass_filter(left_knee_angles, cutoff_frequency, fps)
    right_knee_angles = butter_lowpass_filter(right_knee_angles, cutoff_frequency, fps)
    left_ankle_angles = butter_lowpass_filter(left_ankle_angles, cutoff_frequency, fps)
    right_ankle_angles = butter_lowpass_filter(right_ankle_angles, cutoff_frequency, fps)
    spine_segment_angles = butter_lowpass_filter(spine_segment_angles, cutoff_frequency, fps) 

    ### CROP HERE ###
    start_time, end_time = st.slider(
    "Select time range",
    min_value=float(0),
    max_value=float(total_frames/fps - 1),
    value=(float(0), float(total_frames/fps - 1)),
    key=f"side_time_range_{video_index}_{camera_side}_{hash(video_path)}")
    
    start_frame_crop = int(start_time * fps)
    end_frame_crop = int(end_time * fps)
    st.write(f"Selected frame range: {start_frame_crop} to {end_frame_crop}")
    st.write(f"Selected time range: {start_frame_crop/fps:.2f}s to {end_frame_crop/fps:.2f}s")

    mask = (time >= start_frame_crop) & (time <= end_frame_crop)
    filtered_time = time[mask]

    filtered_spine_segment_angles = np.array(spine_segment_angles)[mask]
    filtered_left_hip_angles = np.array(left_hip_angles)[mask]
    filtered_right_hip_angles = np.array(right_hip_angles)[mask]
    filtered_left_knee_angles = np.array(left_knee_angles)[mask]
    filtered_right_knee_angles = np.array(right_knee_angles)[mask]
    filtered_left_ankle_angles = np.array(left_ankle_angles)[mask]
    filtered_right_ankle_angles = np.array(right_ankle_angles)[mask]

    hip_data = {
    "Time (s)": filtered_time,
    "Left Hip Angle (degrees)": filtered_left_hip_angles,
    "Right Hip Angle (degrees)": filtered_right_hip_angles
    }

    knee_data = {
        "Time (s)": filtered_time,
        "Left Knee Angle (degrees)": filtered_left_knee_angles,
        "Right Knee Angle (degrees)": filtered_right_knee_angles
    }

    ankle_data = {
        "Time (s)": filtered_time,
        "Left Ankle Angle (degrees)": filtered_left_ankle_angles,
        "Right Ankle Angle (degrees)": filtered_right_ankle_angles
    }

    # Create a DataFrame
    hip_df = pd.DataFrame(hip_data)
    knee_df = pd.DataFrame(knee_data)
    ankle_df = pd.DataFrame(ankle_data)

     # HIP RANGES
    column_left = "Left Hip Angle (degrees)"
    prominence = 4
    distance = fps / 2  # Assuming fps/2 equivalent    
    peaks_left = detect_peaks(hip_df, column_left, prominence, distance)
    mins_left = detect_mins(hip_df, column_left, prominence, distance)
    hip_left_mins_mean = np.mean(hip_df[column_left].iloc[mins_left])
    hip_left_mins_std = np.std(hip_df[column_left].iloc[mins_left])
    hip_left_peaks_mean = np.mean(hip_df[column_left].iloc[peaks_left])
    hip_left_peaks_std = np.std(hip_df[column_left].iloc[peaks_left])    
    column_right = "Right Hip Angle (degrees)"
    peaks_right = detect_peaks(hip_df, column_right, prominence, distance)
    mins_right = detect_mins(hip_df, column_right, prominence, distance)
    hip_right_mins_mean = np.mean(hip_df[column_right].iloc[mins_right])
    hip_right_mins_std = np.std(hip_df[column_right].iloc[mins_right])
    hip_right_peaks_mean = np.mean(hip_df[column_right].iloc[peaks_right])
    hip_right_peaks_std = np.std(hip_df[column_right].iloc[peaks_right])
        
    # KNEE CYCLES
    column_left = "Left Knee Angle (degrees)"
    prominence = 3
    distance = fps / 2
    peaks_left = detect_peaks(knee_df, column_left, prominence, distance)
    mins_left = detect_mins(knee_df, column_left, prominence, distance)
    knee_left_mins_mean = np.mean(knee_df[column_left].iloc[mins_left])
    knee_left_mins_std = np.std(knee_df[column_left].iloc[mins_left])
    knee_left_peaks_mean = np.mean(knee_df[column_left].iloc[peaks_left])
    knee_left_peaks_std = np.std(knee_df[column_left].iloc[peaks_left])
    column_right = "Right Knee Angle (degrees)"
    peaks_right = detect_peaks(knee_df, column_right, prominence, distance)
    mins_right = detect_mins(knee_df, column_right, prominence, distance)
    knee_right_mins_mean = np.mean(knee_df[column_right].iloc[mins_right])
    knee_right_mins_std = np.std(knee_df[column_right].iloc[mins_right])
    knee_right_peaks_mean = np.mean(knee_df[column_right].iloc[peaks_right])
    knee_right_peaks_std = np.std(knee_df[column_right].iloc[peaks_right])
    
    # ANKLE CYCLES
    column_left = "Left Ankle Angle (degrees)"
    prominence = 4
    distance = fps / 2
    peaks_left = detect_peaks(ankle_df, column_left, prominence, distance)
    mins_left = detect_mins(ankle_df, column_left, prominence, distance)
    ankle_left_mins_mean = np.mean(ankle_df[column_left].iloc[mins_left])
    ankle_left_mins_std = np.std(ankle_df[column_left].iloc[mins_left])
    ankle_left_peaks_mean = np.mean(ankle_df[column_left].iloc[peaks_left])
    ankle_left_peaks_std = np.std(ankle_df[column_left].iloc[peaks_left])
    column_right = "Right Ankle Angle (degrees)"
    peaks_right = detect_peaks(ankle_df, column_right, prominence, distance)
    mins_right = detect_mins(ankle_df, column_right, prominence, distance)
    ankle_right_mins_mean = np.mean(ankle_df[column_right].iloc[mins_right])
    ankle_right_mins_std = np.std(ankle_df[column_right].iloc[mins_right])
    ankle_right_peaks_mean = np.mean(ankle_df[column_right].iloc[peaks_right])
    ankle_right_peaks_std = np.std(ankle_df[column_right].iloc[peaks_right])
   
    try:
        rom_values = [
        np.ptp(filtered_right_knee_angles),
        np.ptp(filtered_right_hip_angles),
        np.ptp(filtered_spine_segment_angles),
        np.ptp(filtered_left_hip_angles),
        np.ptp(filtered_left_knee_angles),
        np.ptp(filtered_left_ankle_angles),
        np.ptp(filtered_right_ankle_angles)
            ]
    except ValueError:
        st.error("Your cropped out too much data. Please include more data by expanding the cropping slider.")
    
    joint_labels = ['Right Joint Knee', 'Right Joint Hip', 'Spine Segment', 'Left Joint Hip', 'Left Joint Knee', 'Left Joint Ankle', 'Right Joint Ankle']
    knee_right_rom_mean = knee_right_peaks_mean - knee_right_mins_mean
    knee_left_rom_mean = knee_left_peaks_mean - knee_left_mins_mean
    hip_right_rom_mean = hip_right_peaks_mean - hip_right_mins_mean
    hip_left_rom_mean = hip_left_peaks_mean - hip_left_mins_mean
    ankle_right_rom_mean = ankle_right_peaks_mean - ankle_right_mins_mean
    ankle_left_rom_mean = ankle_left_peaks_mean - ankle_left_mins_mean
    spine_segment_rom_mean = np.ptp(filtered_spine_segment_angles)

    spine_rom_good = 10 # 5 to 15 
    ankle_plantar_good = 55 # 40 to 55
    ankle_dorsi_good = 20 # 15 to 25 (<15 is moderate (https://www.runnersworld.com/uk/health/injury/a41329624/dorsiflexion/) <10 is bad)

    ankle_inv_good = 23
    ankle_evert_good = 12
    ankle_rom_good = 70 # 65 to 75; another study said 86
    ankle_rom_walk_good = 30

    knee_flex_good = 125
    knee_ext_good = 0
    knee_rom_good = 125


    hip_flex_good = 55
    hip_ext_good = 10
    hip_rom_good = 65 # <60 deg total flexion-extension ROM is bad

    def get_color(value, good_range, moderate_range):
        """Assigns a gradient color based on the ROM classification."""
        norm = mcolors.Normalize(vmin=good_range[0] - 20, vmax=good_range[1] + 20)  # Normalize scale
        cmap = plt.cm.RdYlGn  # Red-Yellow-Green colormap
        return mcolors.to_hex(cmap(norm(value)))

    # Define ranges for color classification
    if camera_side == "side" and gait_type == "walking": 
        ankle_good = (20, 45)
        ankle_moderate = (15, 20)
        ankle_bad = (0, 55)#(0, 10)

        knee_good = (50, 70)
        knee_moderate = (40, 50)
        knee_bad = (0, 80) #(0, 40)

        hip_good = (25, 45)
        hip_moderate = (15, 25)
        hip_bad = (0, 15)

        spine_good = (0, 5)
        spine_moderate = (5, 10)
        spine_bad = (10, 30)

    if camera_side == "back" and gait_type == "walking": 
        ankle_good = (20, 50)
        ankle_moderate = (15, 20)
        ankle_bad = (0, 15)

        knee_good = (0, 5)
        knee_moderate = (5, 10)
        knee_bad = (10, 30)

        hip_good = (0, 10)
        hip_moderate = (10, 15)
        hip_bad = (15, 50)

        spine_good = (0, 5)
        spine_moderate = (5, 10)
        spine_bad = (10, 30)

    if camera_side == "side" and gait_type == "running": 
        ankle_good = (65, 75)
        ankle_moderate = (55, 85)
        ankle_bad = (55, 95)

        knee_good = (120, 130)
        knee_moderate = (90, 175)
        knee_bad = (90, 175)

        hip_good = (60, 70)
        hip_moderate = (40, 90)
        hip_bad = (40, 90)

        spine_good = (5, 15)
        spine_moderate = (2, 20)
        spine_bad = (0, 30)

    if camera_side == "back" and gait_type == "running":
        ankle_good = (20, 60)
        ankle_moderate = (15, 20)
        ankle_bad = (0, 15)

        knee_good = (0, 5)
        knee_moderate = (5, 10)
        knee_bad = (10, 30)

        hip_good = (0, 10)
        hip_moderate = (10, 20)
        hip_bad = (20, 40)

        spine_good = (1, 10)
        spine_moderate = (10, 20)
        spine_bad = (20, 30)

    if camera_side == "side" and gait_type == "pickup pen": 
        ankle_good = (65, 75)
        ankle_moderate = (55, 85)
        ankle_bad = (55, 95)

        knee_good = (120, 130)
        knee_moderate = (90, 175)
        knee_bad = (90, 175)

        hip_good = (60, 70)
        hip_moderate = (40, 90)
        hip_bad = (40, 90)

        spine_good = (5, 15)
        spine_moderate = (2, 20)
        spine_bad = (0, 30)

    rom_values = [knee_right_rom_mean, hip_right_rom_mean, spine_segment_rom_mean, 
                hip_left_rom_mean, knee_left_rom_mean, ankle_left_rom_mean, ankle_right_rom_mean]

    joint_labels = ["Knee Right", "Hip Right", "Spine", "Hip Left", "Knee Left", "Ankle Left", "Ankle Right"]

    # Assign colors based on ROM value classifications using a gradient
    colors = [
        get_color(rom_values[0], knee_good, knee_moderate),  # Knee Right
        get_color(rom_values[1], hip_good, hip_moderate),    # Hip Right
        get_color(rom_values[2], spine_good, spine_moderate), # Spine
        get_color(rom_values[3], hip_good, hip_moderate),    # Hip Left
        get_color(rom_values[4], knee_good, knee_moderate),  # Knee Left
        get_color(rom_values[5], ankle_good, ankle_moderate),  # Ankle Left (Custom range)
        get_color(rom_values[6], ankle_good, ankle_moderate)   # Ankle Right (Custom range)
    ]

    # Define ideal ROM values (midpoint of the good range)
    ideal_rom_outer = [knee_good[1], hip_good[1], spine_good[1], hip_good[1], knee_good[1], ankle_good[1], ankle_good[1]]
    ideal_rom_inner = [knee_good[0], hip_good[0], spine_good[0], hip_good[0], knee_good[0], ankle_good[0], ankle_good[0]]
    moderate_rom_outer = [knee_moderate[1], hip_moderate[1], spine_moderate[1], hip_moderate[1], knee_moderate[1], ankle_moderate[1], ankle_moderate[1]]
    moderate_rom_inner = [knee_moderate[0], hip_moderate[0], spine_moderate[0], hip_moderate[0], knee_moderate[0], ankle_moderate[0], ankle_moderate[0]]
    bad_rom_outer = [knee_bad[1], hip_bad[1], spine_bad[1], hip_bad[1], knee_bad[1], ankle_bad[1], ankle_bad[1]]
    bad_rom_inner = [knee_bad[0], hip_bad[0], spine_bad[0], hip_bad[0], knee_bad[0], ankle_bad[0], ankle_bad[0]]
      
    # Create polar scatter plot with color-coded points
    spider_plot = go.Figure()
    # side walk --> moderate, poor, ideal, yours, poor inner
   
    # back walk --> poor 0.9, moderate 0.9, ideal 0.85, yours 0.75

    # side run --> moderate, poor, ideal, yours, poor inner

    # back run --> poor, moderate, ideal, yours

    # Plot ideal target ROM values

    spider_plot.add_trace(go.Scatterpolar(
        r=bad_rom_outer,
        theta=joint_labels,
        fill= 'toself', # 'toself' if side walking
        fillcolor='rgba(255, 76, 76, 0.9)', # fillcolor='rgba(255, 0, 0, 0.6)'  # red with 60% opacity
        name='Poor',
        marker=dict(color='#FF4C4C', size=0.1),
        line=dict(color='#FF4C4C', width=2)  # Dashed green outline for ideal ROM
    ))

    spider_plot.add_trace(go.Scatterpolar(
        r=moderate_rom_outer,
        theta=joint_labels,
        fill = 'toself',
        fillcolor='rgba(255, 215, 0, 0.9)',  # gold
        name='Moderate',
        marker=dict(color='#FFD700', size=0.1),
        line=dict(color='#FFD700', width=2)  # Dashed green outline for ideal ROM
    ))

    # Plot ideal target ROM values
    spider_plot.add_trace(go.Scatterpolar(
        r=ideal_rom_outer,
        theta=joint_labels,
        fill='toself',
        fillcolor='rgba(0, 255, 171, 0.85)',  # mint green
        name='Ideal Target',
        marker=dict(color='#00FFAB', size=0.1),
        line=dict(color='#00FFAB', width=2)  # Dashed green outline for ideal ROM
    ))

    # Plot actual values
    spider_plot.add_trace(go.Scatterpolar(
        r=rom_values,
        theta=joint_labels,
        fill='toself',
        name = 'Yours',
        fillcolor='rgba(30, 144, 255, 0.75)',  
        marker=dict(color='#1E90FF', size=0.01),
        line=dict(color='#1E90FF', width=2)
    ))

    # spider_plot.add_trace(go.Scatterpolar(
    #     r=bad_rom_inner,
    #     theta=joint_labels,
    #     fill='toself',
    #     fillcolor='rgba(255, 76, 76, 0.95)', # fillcolor='rgba(255, 0, 0, 0.6)'  # red with 60% opacity
    #     name='Poor Inner',  # Empty name to hide from legend
    #     marker=dict(color='#FF4C4C', size=0.1),
    #     line=dict(color='#FF4C4C', width=0.1)  # Dashed green outline for ideal ROM
    # ))

    # Get max range of motion value
    max_all_joint_angles = max(max(rom_values), max(bad_rom_outer), max(bad_rom_inner), max(ideal_rom_outer)) + 10

    # Update layout
    spider_plot.update_layout(
        title="Range of Motion (°) vs Ideal Target",
        #update title fontsize
        title_font=dict(size=36),
        polar=dict(
            angularaxis=dict(
            tickfont=dict(size=26)  # Increase font size for theta labels
        ),
            radialaxis=dict(
                visible=True,
                range=[0, max_all_joint_angles],
                # only show every other tickfont value
                tickvals=[0, 30, 60, 90, 120, 150, 180],
                tickfont=dict(size=16, color='black')
            )
        ),
        showlegend=True,
        #increase legend fontsize
        legend=dict(
            font=dict(
                size=16
            ))
    )

    st.plotly_chart(spider_plot, key=f"spider_plot_{video_index}_{camera_side}_{hash(video_path)}")

    # st.markdown('### title')
    
    # KEY INSIGHT: Frontal and transverse plane motions (e.g., eversion, adduction) often play a more critical role in injury risk than sagittal plane mechanics

    # Asymmetries in ROM (e.g., >10–15% difference between limbs) are significant predictors of injury across joints (https://pmc.ncbi.nlm.nih.gov/articles/PMC11144664/)
    # Asymmetry ≥6.5° between ankles raises musculoskeletal injury risk by 4–5× in athletes (https://www.ejgm.co.uk/download/role-of-ankle-dorsiflexion-in-sports-performance-and-injury-risk-a-narrative-review-13412.pdf)
    # Asymmetry matters more than absolute values: ≥6.5° ankle dorsiflexion asymmetry quadruples injury risk (https://www.ejgm.co.uk/download/role-of-ankle-dorsiflexion-in-sports-performance-and-injury-risk-a-narrative-review-13412.pdf).
    # Muscle flexibility: Gastrocnemius-soleus tightness limits ankle ROM, altering proximal joint mechanics: https://pmc.ncbi.nlm.nih.gov/articles/PMC9865943/

    # Mean ROM for the assymetry bar plot
    left_hip = hip_left_peaks_mean - hip_left_mins_mean
    right_hip = hip_right_peaks_mean - hip_right_mins_mean
    left_knee = knee_left_peaks_mean - knee_left_mins_mean
    right_knee = knee_right_peaks_mean - knee_right_mins_mean
    left_ankle = ankle_left_peaks_mean - ankle_left_mins_mean
    right_ankle = ankle_right_peaks_mean - ankle_right_mins_mean

    asymmetry_bar_plot = plot_asymmetry_bar_chart(left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle)
    st.plotly_chart(asymmetry_bar_plot, key=f"asymmetry_bar_plot_{video_index}_{camera_side}_{hash(video_path)}")

    # update with decision trees (if elif, for each category)
    st.title('💡 How to improve your range of motion:')

    ankle_text_info = ""
    knee_text_info = ""
    hip_text_info = ""
    spine_text_info = ""

    # HIP FEEDBACK
    if hip_good[0] <= hip_right_rom_mean <= hip_good[1]:
        right_hip_text_summary = "GOOD"
        if gait_type == "walking" and camera_side == "side":
            right_hip_text_info = "Hip flexion at initial contact (~30°) and extension during stance optimize propulsion."
        if gait_type == "running" and camera_side == "side":
            right_hip_text_info = "Hip flexion at initial contact (~50°) and extension during stance optimize propulsion"
        if gait_type == "walking" and camera_side == "back":
            right_hip_text_info = "Minimal motion maintains coronal alignment and reduces hip abductor fatigue."
        if gait_type == "running" and camera_side == "back":
            right_hip_text_info = "Minimal motion maintains coronal alignment and reduces hip abductor fatigue."

    elif hip_moderate[0] <= hip_right_rom_mean <= hip_moderate[1]:
        right_hip_text_summary = "MODERATE"
        if gait_type == "walking" and camera_side == "side":
            right_hip_text_info = "Moderately limited hip range of motion increases lumbar spine compensation and hamstring strain."
        if gait_type == "running" and camera_side == "side":
            right_hip_text_info = "Moderately limited range of motion increases lumbar spine compensation and hamstring strain."
        if gait_type == "walking" and camera_side == "back":
            right_hip_text_info = "Moderate levels of increased pelvic drop heightens iliotibial band syndrome risk."
        if gait_type == "running" and camera_side == "back": 
            right_hip_text_info = "Moderate levels of increased pelvic drop heightens iliotibial band syndrome risk."

    elif hip_bad[0] <= hip_right_rom_mean <= hip_bad[1]:
        right_hip_text_summary = "BAD"
        if gait_type == "walking" and camera_side == "side":
            right_hip_text_info = "Bad (<15° flexion-extension): Severe restriction (<10°) alters pelvic tilt and elevates lower back pain risk."
        if gait_type == "running" and camera_side == "side":
            right_hip_text_info = "Severe restriction (<40°) or poorly controlled motion (>90°) alters pelvic tilt and elevates lower back pain risk."
        if gait_type == "walking" and camera_side == "back":
            right_hip_text_info = "Excessive adduction correlates with tibial stress fractures and labral impingement."
        if gait_type == "running" and camera_side == "back": 
            right_hip_text_info = "Excessive adduction correlates with tibial stress fractures and labral impingement."

    if hip_good[0] <= hip_left_rom_mean <= hip_good[1]:
        left_hip_text_summary = "GOOD"
        if gait_type == "walking" and camera_side == "side":
            left_hip_text_info = "Hip flexion at initial contact (~30°) and extension during stance optimize propulsion"
        if gait_type == "running" and camera_side == "side":
            left_hip_text_info = "Hip flexion at initial contact (~50°) and extension during stance optimize propulsion"
        if gait_type == "walking" and camera_side == "back":
            left_hip_text_info = "Minimal motion maintains coronal alignment and reduces hip abductor fatigue."
        if gait_type == "running" and camera_side == "back":
            left_hip_text_info = "Minimal motion maintains coronal alignment and reduces hip abductor fatigue."

    elif hip_moderate[0] <= hip_left_rom_mean <= hip_moderate[1]:
        left_hip_text_summary = "MODERATE"
        if gait_type == "walking" and camera_side == "side":
            left_hip_text_info = "Moderately limited hip range of motion increases lumbar spine compensation and hamstring strain."
        if gait_type == "running" and camera_side == "side":
            left_hip_text_info = "Moderately limited ROM increases lumbar spine compensation and hamstring strain."
        if gait_type == "walking" and camera_side == "back":
            left_hip_text_info = "Moderate levels of increased pelvic drop heightens iliotibial band syndrome risk."
        if gait_type == "running" and camera_side == "back": 
            left_hip_text_info = "Moderate levels of increased pelvic drop heightens iliotibial band syndrome risk."

    elif hip_bad[0] <= hip_left_rom_mean <= hip_bad[1]:
        left_hip_text_summary = "BAD"
        if gait_type == "walking" and camera_side == "side":
            left_hip_text_info = "Bad (<15° flexion-extension): Severe restriction (<10°) alters pelvic tilt and elevates lower back pain risk."
        if gait_type == "running" and camera_side == "side":
            left_hip_text_info = "Severe restriction (<40°) or poorly controlled motion (>90°) alters pelvic tilt and elevates lower back pain risk."
        if gait_type == "walking" and camera_side == "back":
            left_hip_text_info = "Excessive adduction correlates with tibial stress fractures and labral impingement."
        if gait_type == "running" and camera_side == "back": 
            left_hip_text_info = "Excessive adduction correlates with tibial stress fractures and labral impingement."

   # KNEE FEEDBACK
    if knee_good[0] <= knee_right_rom_mean <= knee_good[1]:
        right_knee_text_summary = "GOOD"
        if gait_type == "walking" and camera_side == "side":
            right_knee_text_info = "50-70° flexion during stance phase optimizes shock absorption."
        if gait_type == "running" and camera_side == "side":
            right_knee_text_info = "Good knee flexion during stance phase optimizes shock absorption."
        if gait_type == "walking" and camera_side == "back":
            right_knee_text_info = "Minimal medial knee deviation protects against patellofemoral knee pain."
        if gait_type == "running" and camera_side == "back":
            right_knee_text_info = "Minimal medial knee deviation protects against patellofemoral knee pain."

    elif knee_moderate[0] <= knee_right_rom_mean <= knee_moderate[1]:
        right_knee_text_summary = "MODERATE"
        if gait_type == "walking" and camera_side == "side":
            right_knee_text_info = "Moderately reduced flexion increases patellofemoral joint stress."
        if gait_type == "running" and camera_side == "side":
            right_knee_text_info = "Reduced flexion increases patellofemoral joint stress."
        if gait_type == "walking" and camera_side == "back":
            right_knee_text_info = "Moderate adduction/abduction correlates with early cartilage wear."
        if gait_type == "running" and camera_side == "back":
            right_knee_text_info = "Moderate adduction/abduction correlates with early cartilage wear."

    elif knee_bad[0] < knee_right_rom_mean:
        right_knee_text_summary = "BAD"
        if gait_type == "walking" and camera_side == "side":
            right_knee_text_info = "You have limited knee flexion, which may reduce running efficiency. Consider deep squats, hamstring stretches, and eccentric loading to improve flexibility."
        if gait_type == "running" and camera_side == "side":
            right_knee_text_info = "Stiff-knee gait raises ACL injury risk due to poor energy dissipation."
        if gait_type == "walking" and camera_side == "back":
            right_knee_text_info = "High knee adduction valgus/varus motion can result in patellofemoral knee pain."
        if gait_type == "running" and camera_side == "back":
            right_knee_text_info = "High knee adduction valgus/varus motion can result in patellofemoral knee pain."

    if knee_good[0] < knee_left_rom_mean:
        left_knee_text_summary = "GOOD"
        if gait_type == "walking" and camera_side == "side":
            left_knee_text_info = "50-70° knee flexion during stance phase optimizes shock absorption."
        if gait_type == "running" and camera_side == "side":
            left_knee_text_info = "Good knee flexion during stance phase optimizes shock absorption."
        if gait_type == "walking" and camera_side == "back":
            left_knee_text_info = "Minimal valgus/varus motion protects against patellofemoral knee pain."
        if gait_type == "running" and camera_side == "back":
            left_knee_text_info = "Minimal valgus/varus motion protects against patellofemoral knee pain."

    elif knee_moderate[0] <= knee_left_rom_mean <= knee_moderate[1]:
        left_knee_text_summary = "MODERATE"
        if gait_type == "walking" and camera_side == "side":
            left_knee_text_info = "Moderately reduced flexion increases patellofemoral joint stress."
        if gait_type == "running" and camera_side == "side":
            left_knee_text_info = "Moderately reduced flexion increases patellofemoral joint stress."
        if gait_type == "walking" and camera_side == "back":
            left_knee_text_info = "Moderate adduction/abduction correlates with early cartilage wear."
        if gait_type == "running" and camera_side == "back":
            left_knee_text_info = "Moderate adduction/abduction correlates with early cartilage wear."

    elif knee_bad[0] >= knee_left_rom_mean or knee_left_rom_mean >= knee_bad[1]:
        left_knee_text_summary = "BAD"
        if gait_type == "walking" and camera_side == "side":
            left_knee_text_info = "You have limited knee flexion, which may reduce running efficiency. Consider deep squats, hamstring stretches, and eccentric loading to improve flexibility."
        if gait_type == "running" and camera_side == "side":
            left_knee_text_info = "Stiff-knee gait raises ACL injury risk due to poor energy dissipation."
        if gait_type == "walking" and camera_side == "back":
            left_knee_text_info = "High knee adduction valgus/varus motion can result in patellofemoral knee pain."
        if gait_type == "running" and camera_side == "back":
            left_knee_text_info = "High knee adduction valgus/varus motion can result in patellofemoral knee pain."

    # ANKLE FEEDBACK ---
    if ankle_good[0] <= ankle_right_rom_mean <= ankle_good[1]:
        right_ankle_text_summary = "GOOD"
        if gait_type == "walking" and camera_side == "side":
            right_ankle_text_info = "Good ankle motion facilitates smooth heel-to-toe transition and shock absorption."
        if gait_type == "running" and camera_side == "side":
            right_ankle_text_info = "Good ankle motion facilitates smooth heel-to-toe transition and shock absorption."
        if gait_type == "walking" and camera_side == "back":
            right_ankle_text_info = "Linked to stable foot placement and reduced step-width variability. Recommended Shoe: Motion Control. Features:  Rigid heel counters 4/5 stiffness score) and reinforced arches.  High torsional rigidity 4/5 score) to restrict transverse plane tibial rotation. Extended medial posts for severe overpronators. Best for: Runners with chronic overpronation, posterior tibial tendon dysfunction, or ACL injury  risks."
        if gait_type == "running" and camera_side == "back":
                        right_ankle_text_info = "Linked to stable foot placement and reduced step-width variability. Recommended Shoe: Motion Control. Features:  Rigid heel counters 4/5 stiffness score) and reinforced arches.  High torsional rigidity 4/5 score) to restrict transverse plane tibial rotation. Extended medial posts for severe overpronators. Best for: Runners with chronic overpronation, posterior tibial tendon dysfunction, or ACL injury  risks."

    elif ankle_moderate[0] <= ankle_right_rom_mean <= ankle_moderate[1]:
        right_ankle_text_summary = "MODERATE"
        if gait_type == "walking" and camera_side == "side":
            right_ankle_text_info = "Slightly reduced ankle range of motion increases forefoot loading and compensatory knee motion."
        if gait_type == "running" and camera_side == "side":
            right_ankle_text_info = "Slightly reduced ankle range of motion increases forefoot loading and compensatory knee motion."
        if gait_type == "walking" and camera_side == "back":
            right_ankle_text_info = "Moderately limits lateral balance control, reducing walking stablity in older adults."
        if gait_type == "running" and camera_side == "back":
            right_ankle_text_info = "Moderately limits lateral balance control, reducing walking stablity in older adults."

    elif ankle_bad[0] >= ankle_right_rom_mean or ankle_right_rom_mean >= ankle_bad[1]:
        right_ankle_text_summary = "BAD"
        if gait_type == "walking" and camera_side == "side":
            right_ankle_text_info = "Severe dorsiflexion deficits (<5°) elevate risks of plantar fasciitis and Achilles tendinopathy."
        if gait_type == "running" and camera_side == "side":
            right_ankle_text_info = "Severe dorsiflexion deficits (<5°) elevate risks of plantar fasciitis and Achilles tendinopathy."
        if gait_type == "walking" and camera_side == "back":
            right_ankle_text_info = "Associated with instability, compensatory pelvic motion, and medial tibial stress syndrome."
        if gait_type == "running" and camera_side == "back":
            right_ankle_text_info = "Associated with instability, compensatory pelvic motion, and medial tibial stress syndrome."

    if ankle_good[0] <= ankle_left_rom_mean <= ankle_good[1]:
        left_ankle_text_summary = "GOOD"
        if gait_type == "walking" and camera_side == "side":
            left_ankle_text_info = "Good ankle motion facilitates smooth heel-to-toe transition and shock absorption."
        if gait_type == "running" and camera_side == "side":
            left_ankle_text_info = "Good ankle motion facilitates smooth heel-to-toe transition and shock absorption."
        if gait_type == "walking" and camera_side == "back":
            left_ankle_text_info = "Linked to stable foot placement and reduced step-width variability. Recommended Shoe: Motion Control. Features:  Rigid heel counters 4/5 stiffness score) and reinforced arches.  High torsional rigidity 4/5 score) to restrict transverse plane tibial rotation. Extended medial posts for severe overpronators. Best for: Runners with chronic overpronation, posterior tibial tendon dysfunction, or ACL injury  risks."

        if gait_type == "running" and camera_side == "back":
            left_ankle_text_info = "Linked to stable foot placement and reduced step-width variability. Recommended Shoe: Motion Control. Features:  Rigid heel counters 4/5 stiffness score) and reinforced arches.  High torsional rigidity 4/5 score) to restrict transverse plane tibial rotation. Extended medial posts for severe overpronators. Best for: Runners with chronic overpronation, posterior tibial tendon dysfunction, or ACL injury  risks."


    elif ankle_moderate[0] <= ankle_left_rom_mean <= ankle_moderate[1]:
        left_ankle_text_summary = "Moderate"
        if gait_type == "walking" and camera_side == "side":
            left_ankle_text_info = "Reduced ankle range of motion increases forefoot loading and compensatory knee motion."
        if gait_type == "running" and camera_side == "side":
            left_ankle_text_info = "Reduced ankle range of motion increases forefoot loading and compensatory knee motion."
        if gait_type == "walking" and camera_side == "back":
            left_ankle_text_info = "Moderately limits lateral balance control, reducing walking stablity in older adults."
        if gait_type == "running" and camera_side == "back":
            left_ankle_text_info = "Moderately limits lateral balance control, reducing walking stablity in older adults."

    elif ankle_bad[0] >= ankle_left_rom_mean or ankle_left_rom_mean >= ankle_bad[1]:
        left_ankle_text_summary = "BAD"
        if gait_type == "walking" and camera_side == "side":
            left_ankle_text_info = "Limited ankle range of motion elevates risks of plantar fasciitis and reduces lower-limb efficiency to push off during each stride."
        if gait_type == "running" and camera_side == "side":
            left_ankle_text_info = "Limited ankle range of motion elevates risks of plantar fasciitis and reduces lower-limb efficiency to push off during each stride."
        if gait_type == "walking" and camera_side == "back":
            left_ankle_text_info = "Associated with instability, compensatory pelvic motion, and medial tibial stress syndrome."
        if gait_type == "running" and camera_side == "back":
            left_ankle_text_info = "Associated with instability, compensatory pelvic motion, and medial tibial stress syndrome."

    # SPINE FEEDBACK ---
    if spine_good[0] <= spine_segment_rom_mean <= spine_good[1]:
        spine_text_summary = "GOOD"
        if gait_type == "walking" and camera_side == "side":
            spine_text_info = "Neutral alignment (±2.5° from vertical) helps maintain natural lumbar lordosis/thoracic kyphosis for optimal shock absorption and energy transfer."    
        if gait_type == "running" and camera_side == "side":
            spine_text_info = "Neutral alignment (±7.5° from vertical) helps maintain natural lumbar lordosis/thoracic kyphosis for optimal shock absorption and energy transfer."    
        if gait_type == "walking" and camera_side == "back":
            spine_text_info = "Minimal lateral deviation (<2.5° per side) which correlates with hip abductor strength and balanced step width."
        if gait_type == "running" and camera_side == "back":
            spine_text_info = "Minimal lateral deviation (<5° per side) which correlates with hip abductor strength and balanced step width."

    elif spine_moderate[0] <= spine_segment_rom_mean <= spine_moderate[1]:
        spine_text_summary = "Moderate"
        if gait_type == "walking" and camera_side == "side":
            spine_text_info = "Moderate forward lean (5-7°) or backward lean (3-5°) is associated with reduced hip extension or ankle mobility deficits, increasing lumbar spine compensatory flexion."
        if gait_type == "running" and camera_side == "side":
            spine_text_info = "Moderate forward lean or backward lean is associated with reduced hip extension or ankle mobility deficits, increasing lumbar spine compensatory flexion."
        if gait_type == "walking" and camera_side == "back":
            spine_text_info = "Moderate lateral lean (5-7° per side), often compensating for hip adduction or ankle inversion/eversion asymmetry"
        if gait_type == "running" and camera_side == "back":
            spine_text_info = "Moderate lateral lean (5-10° per side), often compensating for hip adduction or ankle inversion/eversion asymmetry"

    elif spine_bad[0] <= spine_segment_rom_mean <= spine_bad[1]:
        spine_text_summary = "BAD"
        if gait_type == "walking" and camera_side == "side":
            spine_text_info = '''Not enough trunk lean: Lack of forward lean (walking too upright) reduces forward propulsion, which limits ankle propulsion and increases risk of calf strain. " \
                           Too much trunk lean: Severe anterior/posterior tilt, altering pelvic orientation, is linked to hamstring strain (excessive forward lean) or facet joint compression (excessive backward lean).'''
        if gait_type == "running" and camera_side == "side":
            spine_text_info = "Not enough trunk lean: Lack of forward lean (running too upright) reduces forward propulsion, which limits ankle propulsion and increases risk of calf strain. Too much trunk lean: Severe anterior/posterior tilt, altering pelvic orientation, is linked to hamstring strain (excessive forward lean) or facet joint compression (excessive backward lean)."
        if gait_type == "walking" and camera_side == "back":
            spine_text_info = "Pronounced lateral bending (>10° per side), increases spinal disc shear forces and is associated with unilateral hip weakness or ankle instability."
        if gait_type == "running" and camera_side == "back":
            spine_text_info = "Pronounced lateral bending (>10° per side), increases spinal disc shear forces and is associated with unilateral hip weakness or ankle instability."

    text_info = {
        "left ankle": left_ankle_text_info if 'left_ankle_text_info' in locals() else "",
        "left knee": left_knee_text_info if 'left_knee_text_info' in locals() else "",
        "left hip": left_hip_text_info if 'left_hip_text_info' in locals() else "",
        "right ankle": right_ankle_text_info if 'right_ankle_text_info' in locals() else "",
        "right knee": right_knee_text_info if 'right_knee_text_info' in locals() else "",
        "right hip": right_hip_text_info if 'right_hip_text_info' in locals() else "",
        "spine": spine_text_info if 'spine_text_info' in locals() else "",

        "left ankle summary": left_ankle_text_summary if 'left_ankle_text_summary' in locals() else "",
        "left knee summary": left_knee_text_summary if 'left_knee_text_summary' in locals() else "",
        "left hip summary": left_hip_text_summary if 'left_hip_text_summary' in locals() else "",
        "right ankle summary": right_ankle_text_summary if 'right_ankle_text_summary' in locals() else "",
        "right knee summary": right_knee_text_summary if 'right_knee_text_summary' in locals() else "",
        "right hip summary": right_hip_text_summary if 'right_hip_text_summary' in locals() else "",
        "spine segment summary": spine_text_summary if 'spine_text_summary' in locals() else "",

    }

    with st.expander("Click here to see your spine segment angle data"):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtered_time, y=filtered_spine_segment_angles, mode='lines', name="Spine Segment Angles"))
        fig.add_trace(go.Scatter(x=[frame_time, frame_time], y=[min(filtered_spine_segment_angles), max(filtered_spine_segment_angles)], mode='lines', line=dict(color='red', dash='dash'), name='Selected Frame'))
        fig.update_layout(title=f"Spine Segment Angles", xaxis_title="Time (s)", yaxis_title="Angle (degrees)")
        st.plotly_chart(fig, key=f"spine_segment_expander_{video_index}_{camera_side}_{hash(video_path)}")

        # Assuming filtered_time and filtered_spine_segment_angles are lists or numpy arrays
        spine_data = {
            "Time (s)": filtered_time,
            "Spine Segment Angles (degrees)": filtered_spine_segment_angles
        }

        # Create a DataFrame
        spine_df = pd.DataFrame(spine_data)

        # Convert DataFrame to CSV
        spine_csv = spine_df.to_csv(index=False).encode('utf-8')

        # Add download csv button
        st.download_button(
        label="Download Spine Segment Angle Data",
        data=spine_csv,
        file_name="spine_segment_angles_{camera_side}.csv",
        mime="text/csv",
        key=f"spine_segment_angles_{video_index}_{camera_side}_{hash(video_path)}"
    )        
    filtered_left_hip_angles = np.array(left_hip_angles)[mask]
    filtered_right_hip_angles = np.array(right_hip_angles)[mask]

    with st.expander("Click here to see your hip angle data"):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtered_time, y=filtered_left_hip_angles, mode='lines', name="Left Hip"))
        fig.add_trace(go.Scatter(x=filtered_time, y=filtered_right_hip_angles, mode='lines', name="Right Hip"))
        fig.add_trace(go.Scatter(x=[frame_time, frame_time], y=[min(np.min(filtered_left_hip_angles), np.min(filtered_right_hip_angles)), max(np.max(filtered_left_hip_angles), np.max(filtered_left_hip_angles))], mode='lines', line=dict(color='red', dash='dash'), name='Selected Frame'))
        fig.update_layout(title=f"Hip Joint Angles", xaxis_title="Time (s)", yaxis_title="Angle (degrees)")
        st.plotly_chart(fig)

        # Convert DataFrame to CSV
        hip_csv = hip_df.to_csv(index=False).encode('utf-8')

        # Add download csv button
        st.download_button(
            label="Download Hip Angle Data",
            data=hip_csv,
            file_name="hip_angles_{camera_side}.csv",
            mime="text/csv",
            key=f"hip_angles_{video_index}_{camera_side}_{hash(video_path)}"
        )
        
    filtered_left_knee_angles = np.array(left_knee_angles)[mask]
    filtered_right_knee_angles = np.array(right_knee_angles)[mask]

    with st.expander("Click here to see your knee angle data"):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtered_time, y=filtered_left_knee_angles, mode='lines', name="Left Knee"))
        fig.add_trace(go.Scatter(x=filtered_time, y=filtered_right_knee_angles, mode='lines', name="Right Knee"))
        fig.add_trace(go.Scatter(x=[frame_time, frame_time], y=[min(np.min(filtered_left_knee_angles), np.min(filtered_right_knee_angles)), max(np.max(filtered_left_knee_angles), np.max(filtered_left_knee_angles))], mode='lines', line=dict(color='red', dash='dash'), name='Selected Frame'))
        fig.update_layout(title=f"Knee Joint Angles", xaxis_title="Time (s)", yaxis_title="Angle (degrees)")
        st.plotly_chart(fig)

        knee_data = {
            "Time (s)": filtered_time,
            "Left Knee Angle (degrees)": filtered_left_knee_angles,
            "Right Knee Angle (degrees)": filtered_right_knee_angles
        }

        # Create a DataFrame
        knee_df = pd.DataFrame(knee_data)

        # Convert DataFrame to CSV
        knee_csv = knee_df.to_csv(index=False).encode('utf-8')

        # Add download csv button
        st.download_button(
            label="Download Knee Angle Data",
            data=knee_csv,
            file_name="knee_angles_{camera_side}.csv",
            mime="text/csv",
            key=f"knee_angles_{video_index}_{camera_side}_{hash(video_path)}"
        )
   
    filtered_left_ankle_angles = np.array(left_ankle_angles)[mask]
    filtered_right_ankle_angles = np.array(right_ankle_angles)[mask]

    with st.expander("Click here to see your ankle angle data"):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtered_time, y=filtered_left_ankle_angles, mode='lines', name="Left Ankle"))
        fig.add_trace(go.Scatter(x=filtered_time, y=filtered_right_ankle_angles, mode='lines', name="Right Ankle"))
        fig.add_trace(go.Scatter(x=[frame_time, frame_time], y=[min(np.min(filtered_left_ankle_angles), np.min(filtered_right_ankle_angles)), max(np.max(filtered_left_ankle_angles), np.max(filtered_left_ankle_angles))], mode='lines', line=dict(color='red', dash='dash'), name='Selected Frame'))
        fig.update_layout(title=f"Ankle Joint Angles", xaxis_title="Time (s)", yaxis_title="Angle (degrees)")
        st.plotly_chart(fig)

        ankle_data = {
            "Time (s)": filtered_time,
            "Left Ankle Angle (degrees)": filtered_left_ankle_angles,
            "Right Ankle Angle (degrees)": filtered_right_ankle_angles
        }

        # Create a DataFrame
        ankle_df = pd.DataFrame(ankle_data)

        # Convert DataFrame to CSV
        ankle_csv = ankle_df.to_csv(index=False).encode('utf-8')

        # Add download csv button
        st.download_button(
            label="Download Ankle Angle Data",
            data=ankle_csv,
            file_name="ankle_angles_{camera_side}.csv",
            mime="text/csv",
            key=f"ankle_angles_{video_index}_{camera_side}_{hash(video_path)}"
        )     

    # Store data in DataFrame
    joint_angle_df = pd.DataFrame({
        "Time": filtered_time,
        "Spine": filtered_spine_segment_angles,
        "Left Hip": filtered_left_hip_angles, "Right Hip": filtered_right_hip_angles,
        "Left Knee": filtered_left_knee_angles, "Right Knee": filtered_right_knee_angles,
        "Left Ankle": filtered_left_ankle_angles, "Right Ankle": filtered_right_ankle_angles
    })

    # STRIDE CYCLE DETECTION
    # with st.expander("Stride Cycle Analysis"):

    #     strides = [f"Stride {i+1}" for i in range(min(len(peaks_left), len(mins_left), len(peaks_right), len(mins_right)))]
        
    #     # Plotly bar plot showing peaks and minima side by side with thinner bars
    #     fig = go.Figure()
    #     column_left = "Left Hip Angle (degrees)"
    #     column_right = "Right Hip Angle (degrees)"
    #     fig.add_trace(go.Bar(
    #         y=hip_df[column_left].iloc[peaks_left][:len(strides)],
    #         x=strides,
    #         name="Left Peak Flexion",
    #         marker_color='lightblue',
    #         width=0.2
    #     ))
        
    #     fig.add_trace(go.Bar(
    #         y=hip_df[column_right].iloc[peaks_right][:len(strides)],
    #         x=strides,
    #         name="Right Peak Flexion",
    #         marker_color='lightgreen',
    #         width=0.2
    #     ))

    #     fig.add_trace(go.Bar(
    #         y=hip_df[column_left].iloc[mins_left][:len(strides)],
    #         x=strides,
    #         name="Left Min Flexion",
    #         marker_color='blue',
    #         width=0.2
    #     ))
        
    #     fig.add_trace(go.Bar(
    #         y=hip_df[column_right].iloc[mins_right][:len(strides)],
    #         x=strides,
    #         name="Right Min Flexion",
    #         marker_color='green',
    #         width=0.2
    #     ))
        
    #     fig.update_layout(
    #         title="Joint Flexion Angles Per Stride",
    #         yaxis_title="Hip Angle (degrees)",
    #         barmode='group',  # Ensures bars are side by side
    #         xaxis=dict(tickmode='array', tickvals=list(range(len(strides))), ticktext=strides)
    #     )
        
    #     st.plotly_chart(fig, key=f"hip_expander_{video_index}_{camera_side}_{hash(video_path)}")

    #     strides = [f"Stride {i+1}" for i in range(min(len(peaks_left), len(mins_left), len(peaks_right), len(mins_right)))]

    #     # Plotly bar plot showing peaks and minima side by side with thinner bars
    #     fig = go.Figure()
    #     column_left = "Left Knee Angle (degrees)"
    #     column_right = "Right Knee Angle (degrees)"

    #     fig.add_trace(go.Bar(
    #         y=knee_df[column_left].iloc[peaks_left][:len(strides)],
    #         x=strides,
    #         name="Left Peak Flexion",
    #         marker_color='lightblue',
    #         width=0.2
    #     ))

    #     fig.add_trace(go.Bar(
    #         y=knee_df[column_right].iloc[peaks_right][:len(strides)],
    #         x=strides,
    #         name="Right Peak Flexion",
    #         marker_color='lightgreen',
    #         width=0.2
    #     ))

    #     fig.add_trace(go.Bar(
    #         y=knee_df[column_left].iloc[mins_left][:len(strides)],
    #         x=strides,
    #         name="Left Min Flexion",
    #         marker_color='blue',
    #         width=0.2
    #     ))

    #     fig.add_trace(go.Bar(
    #         y=knee_df[column_right].iloc[mins_right][:len(strides)],
    #         x=strides,
    #         name="Right Min Flexion",
    #         marker_color='green',
    #         width=0.2
    #     ))

    #     fig.update_layout(
    #         title="Joint Flexion Angles Per Stride",
    #         yaxis_title="Knee Angle (degrees)",
    #         barmode='group',  # Ensures bars are side by side
    #         xaxis=dict(tickmode='array', tickvals=list(range(len(strides))), ticktext=strides)
    #     )

    #     st.plotly_chart(fig, key=f"knee_expander_{video_index}_{camera_side}_{hash(video_path)}")

    #     # ANKLE CYCLES

    #     strides = [f"Stride {i+1}" for i in range(min(len(peaks_left), len(mins_left), len(peaks_right), len(mins_right)))]

    #     # Plotly bar plot showing peaks and minima side by side with thinner bars
    #     fig = go.Figure()
    #     column_left = "Left Ankle Angle (degrees)"
    #     column_right = "Right Ankle Angle (degrees)"

    #     fig.add_trace(go.Bar(
    #         y=ankle_df[column_left].iloc[peaks_left][:len(strides)],
    #         x=strides,
    #         name="Left Peak Flexion",
    #         marker_color='lightblue',
    #         width=0.2
    #     ))

    #     fig.add_trace(go.Bar(
    #         y=ankle_df[column_right].iloc[peaks_right][:len(strides)],
    #         x=strides,
    #         name="Right Peak Flexion",
    #         marker_color='lightgreen',
    #         width=0.2
    #     ))

    #     fig.add_trace(go.Bar(
    #         y=ankle_df[column_left].iloc[mins_left][:len(strides)],
    #         x=strides,
    #         name="Left Min Flexion",
    #         marker_color='blue',
    #         width=0.2
    #     ))

    #     fig.add_trace(go.Bar(
    #         y=ankle_df[column_right].iloc[mins_right][:len(strides)],
    #         x=strides,
    #         name="Right Min Flexion",
    #         marker_color='green',
    #         width=0.2
    #     ))

    #     fig.update_layout(
    #         title="Joint Flexion Angles Per Stride",
    #         yaxis_title="Ankle Angle (degrees)",
    #         barmode='group',  # Ensures bars are side by side
    #         xaxis=dict(tickmode='array', tickvals=list(range(len(strides))), ticktext=strides)
    #     )

    #     st.plotly_chart(fig, key=f"ankle_expander_{video_index}_{camera_side}_{hash(video_path)}")

    ### END CROP ###
  # show tables
    df = pd.DataFrame({'Time': filtered_time, 'Spine Segment Angles': filtered_spine_segment_angles, 'Left Joint Hip': filtered_left_hip_angles, 'Right Hip': filtered_right_hip_angles, 'Left Knee': filtered_left_knee_angles, 'Right Knee': filtered_right_knee_angles, 'Left Ankle': filtered_left_ankle_angles, 'Right Ankle': filtered_right_ankle_angles})
    st.write('### Joint Angles (°)')

    st.dataframe(df)

    st.write('### Range of Motion')
    # create dataframe of range of motion
    
    df_rom = pd.DataFrame({'Joint': ['Spine Segment', 'Left Hip', 'Right Hip', 'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle'], 
    'Min Angle (°)' : [np.min(filtered_spine_segment_angles), hip_left_mins_mean, hip_right_mins_mean, knee_left_mins_mean, knee_right_mins_mean, ankle_left_mins_mean, ankle_right_mins_mean],
    'Max Angle (°)' : [np.max(filtered_spine_segment_angles), hip_left_peaks_mean, hip_right_peaks_mean, knee_left_peaks_mean, knee_right_peaks_mean, ankle_left_peaks_mean, ankle_right_peaks_mean],
    'Range of Motion (°)': [np.ptp(filtered_spine_segment_angles), hip_left_peaks_mean - hip_left_mins_mean, hip_right_peaks_mean - hip_right_mins_mean, knee_left_peaks_mean - knee_left_mins_mean, knee_right_peaks_mean - knee_right_mins_mean, ankle_left_peaks_mean - ankle_left_mins_mean, ankle_right_peaks_mean - ankle_right_mins_mean]})
    
    # round df_rom to 1 decimal place
    df_rom = df_rom.round(1)
    st.dataframe(df_rom)

    # pca_checkbox = st.checkbox("Perform Principle Component Analysis", value=False, key=f"pca_{video_index}_{camera_side}")
    # if pca_checkbox:
    #     perform_pca(joint_angle_df, video_index)

    _, __, pose_image_path = process_first_frame_report(video_path, video_index)
    pdf_path = generate_pdf(pose_image_path, df_rom, spider_plot, asymmetry_bar_plot, text_info, camera_side, gait_type)
    # with open(pdf_path, "rb") as file:
    # st.download_button("Download Stride Sync Report", file, "Stride_Sync_Report.pdf", "application/pdf", key=f"pdf_report_{video_index}_{camera_side}_{hash(video_path)}")

    # email me my Stride Sync Report
    # email = st.text_input("Enter your email address to receive your Stride Sync Report",  key=f"text_input_email_{video_index}_{camera_side}_{hash(video_path)}")
    # if st.button("Email Stride Sync Report", key=f"email_pdf_{video_index}_{camera_side}_{hash(video_path)}"):
    #     send_email(email, pdf_path)

def send_email(to_email, attachment_path):

    if "EMAIL_ADDRESS" in st.secrets:
        sender_email = st.secrets["EMAIL_ADDRESS"]
        app_password = st.secrets["EMAIL_APP_PASSWORD"]
    else:
        load_dotenv()
        sender_email = os.getenv("EMAIL_ADDRESS")
        app_password = os.getenv("EMAIL_APP_PASSWORD")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(sender_email, app_password)     
        st.write("✅ Email sent!") 
    
    msg = EmailMessage()
    msg['Subject'] = "Your Stride Sync Report"
    msg['From'] = sender_email
    msg['To'] = to_email
    msg.set_content("Hi! Attached is your personalized gait report from Stride Sync. Feel free to reach out if you have any questions or would like to setup an appointment to discuss your results.")

    # Attach PDF
    with open(attachment_path, 'rb') as f:
        file_data = f.read()
        file_name = "Stride Sync Report " + str(datetime.now().strftime("%Y-%m-%d")) + ".pdf"
        msg.add_attachment(file_data, maintype='application', subtype='pdf', filename=file_name)

    # Send Email
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(sender_email, app_password)
        smtp.send_message(msg)


# TO DO:
# - Try to add article links like this: https://pmc.ncbi.nlm.nih.gov/articles/PMC3286897/
# - Neural Network to predict gait
# - Add more joints
# - Add more videos
# - Add more data sources (IMUs, wearables, heart rate)
# - Add more analysis
# - Add more visualizations
# - Add more interactivity
# - Add more features
# - Add more machine learning
# - Add more deep learning
# - Add more statistics
# - Add more physics (OpenSim)
# - Add more synthetic data
# - Add animations / rendering
# - Add step by step variation analysis

   

def main():
    # st.title("Biomechanics Analysis from Video")

    iphone_mockup = r"C:\Users\ano\Box\myBox\5G4PHealth App Code\5G4PHealth\photos\pickup pen mockup 2 iphone white background.gif"
    # play gif
    st.image(iphone_mockup, caption="5G4PHealth App", use_column_width=True)
   
    st.title("Try Example Videos")
    example_video = st.radio("Select an example video", 
            ["Select an option", "Pickup pen video", "Sit to stand video", "Single Leg Squat", "Depth Squat", "Timed Up and Go Test"],
            index=0)  
    if example_video == "Select an option":
        st.warning("Please select a valid option.")
    
    if example_video == "Pickup pen video":
        camera_side = "side"
        gait_type = "pickup pen"
        video_url = r"C:\Users\ano\Box\myBox\5G4PHealth App Code\5G4PHealth\photos\pickup pen 3 sec demo.mp4"
        # st.image(github_url + "photos/pickup pen no skeleton sharp.jpg", caption="Example Pickup Pen Video", width=155)
        st.video(video_url)
        # Video URL from GitHub
        for idx, video_file in enumerate([video_url]):
            output_txt_path = '/workspaces/PolarPlotter/results/joint_angles.txt'
            frame_number, frame_time, image_path = process_first_frame(video_file, video_index=idx)
            process_video(gait_type, camera_side, video_file, output_txt_path, frame_time, video_index=idx)

    if example_video == "Sit to stand video":
        camera_side = "side"
        gait_type = "pickup pen"
        video_url = r"C:\Users\ano\Box\myBox\5G4PHealth App Code\5G4PHealth\photos\sit2stand 1 rep.MOV"
        st.video(video_url)
        # Video URL from GitHub
        for idx, video_file in enumerate([video_url]):
            output_txt_path = '/workspaces/PolarPlotter/results/joint_angles.txt'
            frame_number, frame_time, image_path = process_first_frame(video_file, video_index=idx)
            process_video(gait_type, camera_side, video_file, output_txt_path, frame_time, video_index=idx)

    if example_video == "Single Leg Squat":
        camera_side = "side"
        gait_type = "pickup pen"
        video_url = r"C:\Users\ano\Box\myBox\5G4PHealth App Code\5G4PHealth\photos\single leg squat demo.MOV"
        st.video(video_url)
        # Video URL from GitHub
        for idx, video_file in enumerate([video_url]):
            output_txt_path = '/workspaces/PolarPlotter/results/joint_angles.txt'
            frame_number, frame_time, image_path = process_first_frame(video_file, video_index=idx)
            process_video(gait_type, camera_side, video_file, output_txt_path, frame_time, video_index=idx)

    
    # user_footwear = st.text_input("Enter your footwear", key="user_footwear") # maybe checkbox neutral, support, stability --> Opens up a catalogue at their stores...


    # File uploader for user to upload their own video
    video_files = st.file_uploader("Upload side walking video(s)", type=["mp4", "avi", "mov"], accept_multiple_files=True, key="side_walking")
    if video_files:
        camera_side = "side"
        gait_type = "walking"
        for idx, video_file in enumerate(video_files):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
                temp_video_file.write(video_file.read())
                temp_video_path = temp_video_file.name
                temp_video_file.close()
                output_txt_path = '/workspaces/PolarPlotter/results/joint_angles.txt'
                frame_number, frame_time, image_path = process_first_frame(temp_video_path, video_index=idx)
                process_video(gait_type, camera_side, temp_video_path, output_txt_path, frame_time, video_index=idx)

    # File uploader for user to upload their own video
    video_files = st.file_uploader("Upload back walking video(s)", type=["mp4", "avi", "mov"], accept_multiple_files=True, key="back_walking")
    if video_files:
        camera_side = "back"
        gait_type = "walking"
        for idx, video_file in enumerate(video_files):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
                temp_video_file.write(video_file.read())
                temp_video_path = temp_video_file.name
                temp_video_file.close()
                output_txt_path = '/workspaces/PolarPlotter/results/joint_angles.txt'
                frame_number, frame_time, image_path = process_first_frame(temp_video_path, video_index=idx)
                process_video(gait_type, camera_side, temp_video_path, output_txt_path, frame_time, video_index=idx)

    video_files = st.file_uploader("Upload side running video(s)", type=["mp4", "avi", "mov"], accept_multiple_files=True, key="side_running")
    if video_files:
        camera_side = "side"
        gait_type = "running"
        for idx, video_file_back in enumerate(video_files):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
                temp_video_file.write(video_file_back.read())
                temp_video_path = temp_video_file.name
                temp_video_file.close()
                output_txt_path = '/workspaces/PolarPlotter/results/joint_angles.txt'
                frame_number, frame_time, image_path = process_first_frame(temp_video_path, video_index=idx)
                process_video(gait_type, camera_side, temp_video_path, output_txt_path, frame_time, video_index=idx)

    # File uploader for back video(s)
    video_files = st.file_uploader("Upload back running video(s)", type=["mp4", "avi", "mov"], accept_multiple_files=True, key="back_running")
    if video_files:
        camera_side = "back"
        gait_type = "running"
        for idx, video_file_back in enumerate(video_files):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
                temp_video_file.write(video_file_back.read())
                temp_video_path = temp_video_file.name
                temp_video_file.close()
                output_txt_path = '/workspaces/PolarPlotter/results/joint_angles.txt'
                frame_number, frame_time, image_path = process_first_frame(temp_video_path, video_index=idx)
                process_video(gait_type, camera_side, temp_video_path, output_txt_path, frame_time, video_index=idx)

    st.title("Overview of 5G4PHealth")
    st.write("5G4PHealth is a mobile app that uses computer vision and machine learning to analyze your gait and provide personalized feedback. The app can be used to assess your gait, identify any issues, and provide recommendations for improvement.")
    st.write("5G4PHealth is designed to be used by anyone who wants to improve their gait, whether they are a runner, walker, or just looking to improve their overall health. The app is easy to use and provides real-time feedback on your gait.")
    st.write("5G4PHealth is currently in development and will be available for download in the near future. Stay tuned for updates!")
    st.write("For more information, please visit our website: [5G4PHealth](https://5g4phealth.com)")
    main_image = r"C:\Users\ano\Box\myBox\5G4PHealth App Code\5G4PHealth\photos\5G4PHealth overview.png"

    st.image(main_image, caption="Overview of 5G4PHealth", use_column_width=True)

if __name__ == "__main__":
    main()
