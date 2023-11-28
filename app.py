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
# ... contenu de helper.py ...

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
            res = self.model.predict(input, conf=self.conf)
            print(res)
            str_result = ",".join(res)

            # Plot the detected objects on the video frame
            res_plotted = res[0].plot()
            return res_plotted

        return input


# Configuration de la webcam
def play_webcam(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLO object detection model.

    Returns:
        None

    Raises:
        None
    """
    webrtc_streamer(
        key="example",
        video_transformer_factory=lambda: MyVideoTransformer(conf, model),
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

# Lancement de la webcam
confidence = 0.4  # Définir la valeur de confiance
model = YOLO('yolov8n-pose.pt')  # Charger le modèle
play_webcam(confidence, model)

st.write(str_result)
