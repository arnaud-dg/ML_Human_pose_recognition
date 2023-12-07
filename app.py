import cv2
import numpy as np
import pandas as pd
import av
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import classes_functions
# from classes_functions import FaceLandmarks
from datetime import datetime
import requests

# Constant
min_detection_confidence=0.5 
min_tracking_confidence=0.5
# Initializing the mediapipe pose class
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_detection = mp.solutions.face_detection
mp_pose = mp.solutions.pose
model = mp_pose.Pose(min_detection_confidence, min_tracking_confidence)
fl = FaceLandmarks()
# WebRTC configuration
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
# Dataframe
df = pd.DataFrame(columns=['angle_arm_l', 'angle_arm_r', 'angle_leg_l', 'angle_leg_r', 'distance_shoulder', 'distance_hip'])

# Streamlit page configuration
st.set_page_config(
    page_title="Ergonomy Smart Assistant",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("Ergonomy Smart Assistant 🎥🦺")

# Sidebar
video_source = st.sidebar.radio("What is your video source", ["Webcam", "Video"])
blurring_mode = st.sidebar.radio("Would you like to activate the blurring mode", ["Yes", "No"], index=1)
min_detection_confidence = st.sidebar.slider("Detection Threshold :", 0.0, 1.0, 0.5, 0.1)
min_tracking_confidence = st.sidebar.slider("Tracking Threshold :", 0.0, 1.0, 0.5, 0.1)

# Function to get the raw URL of the video file from GitHub
def get_video_url(github_path):
    content_url = f"https://api.github.com/repos/{github_path}/contents/"
    repo_content = requests.get(content_url).json()
    video_urls = {item['name']: item['download_url'] for item in repo_content if item['name'].endswith('.mp4')}
    return video_urls

# Replace 'your-username/repo-name' with your GitHub username and repository name
video_urls = get_video_url('arnaud-dg/ML_Human_pose_recognition/contents')

# Class to process each frame of the video
class VideoProcessor():
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = classes_functions.process(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
tab1, tab2  = st.tabs(["Acquisition", "Report"])

with tab1:
    if video_source == "Webcam":
        st.header("Acquisition")
        webrtc_ctx = webrtc_streamer(
            key="example",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            # async_processing=True,
        )
    else:
        # Dropdown to select the video
        # selected_video = st.selectbox('Select a Video', options=list(video_urls.keys()))
        uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

        if uploaded_video is not None:
            # Lire la vidéo
            video_cap = cv2.VideoCapture(uploaded_video.name)

            # Lecture frame par frame
            stframe = st.empty()
            while video_cap.isOpened():
                ret, frame = video_cap.read()
                if not ret:
                    break

                    img = frame.to_ndarray(format="bgr24")
                    # print(img)
                    img = process(img)

                    # Afficher la frame traitée
                    stframe.image(img, channels="BGR", use_column_width=True)
    
with tab2:
    st.write("No report yet")
    st.dataframe(df)    
