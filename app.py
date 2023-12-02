# Importation des packages
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
import streamlit as st
from PIL import Image
import numpy as np

# Basic configuration and title for the Streamlit app
st.set_page_config(page_title="Ergonomy Detection Bot", page_icon="🤖", layout="wide")
st.title("Ergonomy Detection Bot")
st.markdown('Ouvrez votre webcam et cliquez sur le bouton Start pour commencer l\'acquisition.')

class MyVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = YOLO('yolov8n-pose.pt')

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        return self.process_image(image)

    def process_image(self, image):
        input = np.asarray(Image.fromarray(image).resize((720, int(720 * image.shape[0] / image.shape[1]))))
        results = self.model.predict(input, conf=0.4)
        return results[0].plot()

# Stream webcam with YOLO model
webrtc_streamer(key="example", video_processor_factory=MyVideoTransformer, 
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": True, "audio": False})
