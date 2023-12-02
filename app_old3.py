# Importation des packages
from pathlib import Path
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import VideoProcessorBase, RTCConfiguration, WebRtcMode, webrtc_streamer, VideoTransformerBase
import numpy as np
from PIL import Image
import av
import cv2
import settings

# Paramètres et fonctions de settings.py et helper.py
WEBCAM_PATH = 0

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Ergonomy Detection Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre et description
st.title("Ergonomy Detection Bot")
st.markdown('Ouvrez votre webcam et cliquez sur le bouton Start pour commencer l acquisition.')

class MyVideoTransformer(VideoTransformerBase):
    def __init__(self, conf, model):
        self.conf = conf
        self.model = model

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        processed_image = self._display_detected_frames(image)
        st.image(processed_image, caption='Detected Video', channels="BGR", use_column_width=True)

    def _display_detected_frames(self, image):
        orig_h, orig_w = image.shape[0:2]
        width = 720  # Set the desired width for processing

        # cv2.resize used in a forked thread may cause memory leaks
        input = np.asarray(Image.fromarray(image).resize((width, int(width * orig_h / orig_w))))

        if self.model is not None:
            # Perform object detection using YOLO model
            results = self.model.predict(input, conf=self.conf)
            # result_keypoint = results.keypoints.xyn.cpu().numpy()[0]
            print(result_keypoint)

            # Plot the detected objects on the video frame
            res_plotted = results[0].plot()
            return res_plotted

        return input

# def process(image):
#     # image.flags.writeable = False
#     print("test")
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = model(image)
#     result_keypoint = results.keypoints.xyn.cpu().numpy()[0]
#     print(results)
#     # Draw the hand annotations on the image.
#     # image.flags.writeable = True
#     # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     # print(results)
#     image_draw = results.plot(boxes=False)
#     image_draw = cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB)
#     return cv2.flip(image, 1)

# class VideoProcessor:
#     def recv(self, frame):
#         image = frame.to_ndarray(format="bgr24")
#         processed_image = process(image)
#         st.image(processed_image, caption='Detected Video', channels="BGR", use_column_width=True)

# Configuration de la webcam
def play_webcam(conf, model):
    webrtc_streamer(
        key="example",
        # video_processor_factory=VideoProcessor,
        video_processor_factory=lambda: MyVideoTransformer(conf, model),
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

# Lancement de la webcam
confidence = 0.4  # Définir la valeur de confiance
model = YOLO('yolov8n-pose.pt')  # Charger le modèle
play_webcam(confidence, model)
