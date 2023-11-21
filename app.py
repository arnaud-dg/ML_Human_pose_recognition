from ultralytics import YOLO
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import numpy as np
from PIL import Image
import av
import cv2
from pytube import YouTube
import settings

# Setting page layout
st.set_page_config(
    page_title="Webcam ergonomy detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Pre-trained ML Model
try:
    model = YOLO('yolov8n-pose.pt')
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

def play_webcam(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLO object detection model.

    Returns:
        None

    Raises:
        None
    """
    st.sidebar.title("Webcam Object Detection")

    webrtc_streamer(
        key="example"
        # video_transformer_factory=lambda: MyVideoTransformer(conf, model),
        # rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        # media_stream_constraints={"video": True, "audio": False},
    )

st.title("Webcam Object Detection")
play_webcam(confidence, model)
