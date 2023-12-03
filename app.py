# Importation des packages
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
import streamlit as st
from PIL import Image
import numpy as np

# Basic configuration and title for the Streamlit app
st.set_page_config(page_title="Ergonomy Detection Bot", page_icon="🤖", layout="wide")
st.title("Ergonomy Detection Bot")

results = []

# Définir une fonction de mise en cache pour stocker les résultats
@st.cache(allow_output_mutation=True)
def get_cached_data():
    return pd.DataFrame(columns=['Resultat'])

# Fonction pour mettre à jour le dataframe
def update_data(new_data):
    df = get_cached_data()
    df = df.append(new_data, ignore_index=True)
    return df

class MyVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = YOLO('yolov8n-pose.pt')

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        processed_image = self.process_image(image)
        st.image(processed_image, caption='Detected Video', channels="BGR", use_column_width=True)

    def process_image(self, image):
        input = np.asarray(Image.fromarray(image).resize((720, int(720 * image.shape[0] / image.shape[1]))))
        results = self.model.predict(input, conf=0.4)
        result_keypoint = results[0].keypoints.xyn.cpu().numpy()[0]
        # Mettre à jour le dataframe avec les nouveaux résultats
        new_data = {'Resultat': result_keypoint}
        update_data(new_data)
        return results[0].plot()

tab1, tab2 = st.tabs(["Acquisition", "Report"])

with tab1:
    st.markdown('Ouvrez votre webcam et cliquez sur le bouton Start pour commencer l\'acquisition.')
    # Stream webcam with YOLO model
    webrtc_streamer(key="example", video_processor_factory=MyVideoTransformer, rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}, media_stream_constraints={"video": True, "audio": False})

with tab2:
    st.markdown('Rapport des résultats de prédiction :')
    # Afficher le dataframe
    df = get_cached_data()
    if not df.empty:
        st.dataframe(df)
    else:
        st.markdown('Sorry, you haven\'t acquired anything !')
