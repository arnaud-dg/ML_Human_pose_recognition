import cv2
import numpy as np
import av
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from facial_landmarks import FaceLandmarks

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_detection = mp.solutions.face_detection
mp_pose = mp.solutions.pose
fl = FaceLandmarks()

st.set_page_config(layout="wide")
min_detection_confidence=0.5 
min_tracking_confidence=0.5

model = mp_pose.Pose() #(min_detection_confidence, min_tracking_confidence)

def bluring_face(frame):
    frame = cv.resize(frame, None, fx = 0.5, fy = 0.5)
    frame_copy = frame.copy()
    height, width, _ = frame.shape

    # 1 Face landmark detection
    landmarks = fl.get_facial_landmarks(frame)
    convexhull = cv.convexHull(landmarks)

    # 2 Face blurring
    mask = np.zeros((height, width), np.uint8)
    cv.polylines(mask ,[convexhull], True, 255, 3)
    cv.fillConvexPoly(mask, convexhull, 255)

    # Extract the Face
    frame_copy = cv.blur(frame_copy, (27, 27))  
    face_extracted = cv.bitwise_and(frame_copy, frame_copy, mask = mask)
    blureed_frame = cv.GaussianBlur(face_extracted,(27, 27), 0)

    # Extract background
    background_mask = cv.bitwise_not(mask)
    background = cv.bitwise_and(frame, frame, mask= background_mask)

    # Final result
    result = cv.add(background, blureed_frame)

    # cv.imshow('Result', result)
    # cv.imshow('Frame', frame)

    return result

def process(image):
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Vérifier si des landmarks ont été détectés
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), #2,138,15
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

    if blurring_mode == "Yes":
        image = bluring_face(image) 

    return cv2.flip(image, 1)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoProcessor():
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        print(img)
        img = process(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Page
st.title("Ergonomy Smart Assistant")

video_source = st.sidebar.radio("What is your video source", ["Webcam", "Video"])
# min_detection_confidence = st.sidebar.slider("Detection Threshold :", min_value=0, max_value=1, value=0.5, step=0.1)
# min_tracking_confidence = st.sidebar.slider("Tracking Threshold :", min_value=0, max_value=1, value=0.5, step=0.1)
blurring_mode = st.sidebar.radio("Would you like to activate the blurring mode", ["Yes", "No"], index=1)

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
