import cv2
import numpy as np
import pandas as pd
import av
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import classes_functions
from classes_functions import FaceLandmarks
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
model = mp_pose.Pose() #min_detection_confidence, min_tracking_confidence)
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
        selected_video = st.selectbox('Select a Video', options=list(video_urls.keys()))

        # Process and display the selected video
        if selected_video:
            # Ouvrir la vidéo
            video_url = video_urls[selected_video]
            video_stream = cv2.VideoCapture(video_url)
    
            # Créer un espace pour afficher la frame traitée
            stframe = st.empty()
    
            # Lecture frame par frame
            while video_stream.isOpened():
                ret, frame = video_stream.read()
                if not ret:
                    break
    
                # Appliquer la fonction process() à la frame
                processed_frame = process(frame)
    
                # Afficher la frame traitée
                stframe.image(processed_frame, channels="BGR", use_column_width=True)
    
            # Relâcher la ressource vidéo
            video_stream.release()
        
with tab2:
    st.write("No report yet")
    st.dataframe(df)    
    st.write(df.shape)

# # Render curl counter
# # Setup status box
# cv2.rectangle(image, (0,0), (255,73), (245,117,16), -1)

# # Rep data
# cv2.putText(image, 'REPS', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
# cv2.putText(image, str(counter), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

# # Stage data
# cv2.putText(image, 'STAGE', (65,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
# cv2.putText(image, stage, (60,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)