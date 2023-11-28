# Importation des packages
from pathlib import Path
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
# ... autres importations ...

# Paramètres et fonctions de settings.py et helper.py
# ... contenu de settings.py ...
# ... contenu de helper.py ...

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Détection d'objets avec Webcam",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre et description
st.title("Détection d'objets avec Webcam")
st.markdown('Ouvrez votre webcam et cliquez sur le bouton "Détecter les objets" pour commencer.')

# Configuration de la webcam
def play_webcam(conf, model):
    # ... code de la fonction play_webcam de helper.py ...

# Lancement de la webcam
confidence = 0.4  # Définir la valeur de confiance
model = load_model(Path('chemin/vers/le/modèle'))  # Charger le modèle
play_webcam(confidence, model)
