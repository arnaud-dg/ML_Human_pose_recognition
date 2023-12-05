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

model = mp_pose.Pose()

# def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
#     image = frame.to_ndarray(format="bgr24")
#     # disable landmarks
#     # image.flags.writeable = False
#     # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     # Run inference
#     results = mp_pose.process(image)
#     # enable landmarks landmarks
#     # image.flags.writeable = True
#     # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#     print(results.pose_landmarks)
    
#     # Vérifier si des landmarks ont été détectés
#     if results.pose_landmarks:
#         mp_drawing.draw_landmarks(
#             image,
#             results.pose_landmarks,
#             mp_pose.POSE_CONNECTIONS,
#             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
#             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
#         )
#     return av.VideoFrame.from_ndarray(image, format="bgr24")

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
        
webrtc_ctx = webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    # video_frame_callback=video_frame_callback,
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    # async_processing=True,
)
