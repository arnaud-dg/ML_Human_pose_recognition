import cv2
import numpy as np
import av
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_detection = mp.solutions.face_detection
mp_pose = mp.solutions.pose

min_detection_confidence=0.5 
min_tracking_confidence=0.5

model = mp_pose.Pose() #(min_detection_confidence, min_tracking_confidence)

def process(image):
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    print(results.pose_landmarks)
    if results.pose_landmarks == None:
        print('lapin')
    
    # Vérifier si des landmarks ont été détectés
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
    return cv2.flip(image, 1)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoProcessor():
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        print(img)
        # img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
        img = process(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Page
st.title("Ergonomy Smart Assistant")

video_source = st.sidebar.radio("What is your video source", ["Webcam", "Video"])
# min_detection_confidence = st.sidebar.slider("Detection Threshold :", min_value=0, max_value=1, value=0.5, step=0.1)
# min_tracking_confidence = st.sidebar.slider("Tracking Threshold :", min_value=0, max_value=1, value=0.5, step=0.1)
blurring_mode = st.sidebar.radio("Would you like to activate the blurring mode", ["Yes", "No"])

tab1, tab2  = st.tabs(["Acquisition", "Report"])

with tab1:
    st.header("Acquisition")
    webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        # async_processing=True,
    )

with tab2:
    st.write("No report yet")
