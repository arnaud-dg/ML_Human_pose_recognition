import cv2
import numpy as np
import pandas as pd
import av
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import classes_functions as cf
from datetime import datetime
import requests

# Constant
min_detection_confidence=0.5 
min_tracking_confidence=0.5
# WebRTC configuration
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
# Dataframe
# Avant la définition de vos fonctions, initialisez une variable de session pour le DataFrame
# if 'df' not in st.session_state:
#     st.session_state.df = pd.DataFrame(columns=['angle_arm_l', 'angle_arm_r', 'angle_leg_l', 'angle_leg_r', 'distance_shoulder', 'distance_hip'])
df = pd.DataFrame(columns=['angle_arm_l', 'angle_arm_r', 'angle_leg_l', 'angle_leg_r', 'distance_shoulder', 'distance_hip'])

# Initializing the mediapipe pose class + Load the model
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_detection = mp.solutions.face_detection
mp_pose = mp.solutions.pose
model = mp_pose.Pose() #min_detection_confidence, min_tracking_confidence)
fl = cf.FaceLandmarks()

# Load the model (only executed once!)
# NOTE: Don't set ttl or max_entries in this case
# @st.cache
# def load_model():
# 	  return torch.load("path/to/model.pt")
# model = load_model()

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
# def get_video_url(github_path):
#     content_url = f"https://api.github.com/repos/{github_path}/contents/"
#     repo_content = requests.get(content_url).json()
#     video_urls = {item['name']: item['download_url'] for item in repo_content if item['name'].endswith('.mp4')}
#     return video_urls

# Replace 'your-username/repo-name' with your GitHub username and repository name
# video_urls = get_video_url('arnaud-dg/ML_Human_pose_recognition/contents')
def calculate_angle(a,b,c):
    """
    Calculate angle between three points
    inputs : a, b, c coordinates
    return : angle in degrees
    """

    a = np.array(a) # Point 1
    b = np.array(b) # Point 2
    c = np.array(c) # Point 3
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360.0 - angle
        
    return angle

def distance(a,b):
    """
    Calculate euclidian distance between two points
    inputs : a, b coordinates
    return : distance in pixels
    """
    a = np.array(a) # Point 1
    b = np.array(b) # Point 2
    distance_ab = np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)
  
    return distance_ab

def angle_extraction(landmarks):
    """
    Extract angle from landmarks
    inputs : landmarks
    return : list of angles
    """
    # Left arm
    shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    angle_arm_l = calculate_angle(shoulder_l, elbow_l, hip_l)
    angle_arm_l = np.round(angle_arm_l,1)
    # Right arm
    shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    angle_arm_r = calculate_angle(shoulder_r, elbow_r, hip_r)
    angle_arm_r = np.round(angle_arm_r,1)
    # Left leg
    ankle_l = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    knee_l = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    angle_leg_l = calculate_angle(ankle_l, knee_l, hip_l)
    angle_leg_l = np.round(angle_leg_l,1)
    # Right leg
    ankle_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    knee_r = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    angle_leg_r = calculate_angle(ankle_r, knee_r, hip_r)
    angle_leg_r = np.round(angle_leg_r,1)
    # Ratio back
    shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    distance_shoulder = distance(shoulder_l, shoulder_r)
    distance_hip = distance(hip_l, hip_r)

    list_angle = [angle_arm_l, angle_arm_r, angle_leg_l, angle_leg_r, distance_shoulder, distance_hip]
    return list_angle

def process_hpr(image):
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if blurring_mode == "Yes":
        image = bluring_face(image) 

    # Vérifier si des landmarks ont été détectés
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), #2,138,15
            mp_drawing.DrawingSpec(color=(0,128,0), thickness=2, circle_radius=2)
        )

    landmarks = results.pose_landmarks.landmark
    list_angle = angle_extraction(landmarks) # [angle_arm_l, angle_arm_r, angle_leg_l, angle_leg_r, distance_shoulder, distance_hip]

    print(list_angle)
    # Affiche à l'écran un message
    # Setup status box
    # message = "Good posture"
    # cv2.rectangle(image, (0,0), (255,73), (245,117,16), -1)
    # if angle_arm_l >= 80 or angle_arm_r >= 80 :
    #     max_value = max(angle_arm_l, angle_arm_r)
    #     message = "Warning ! Dangerous shoulder angle detected: " + str(max_value) + "°)"
    #     cv2.rectangle(image, (0,0), (255,73), (245,117,16), -1)
    # elif angle_leg_l >= 70 or angle_leg_r >= 70 :
    #     max_value = max(angle_leg_l, angle_leg_r)
    #     message = "Warning ! Low posture detected: " + str(max_value) + "°)"
    # cv2.putText(image, message, (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

    # current_time = datetime.now()
    # df.loc[current_time] = list_angle
    # st.session_state.df.loc[current_time] = list_angle 

    return cv2.flip(image, 1)

def resize_image(image):
    scale_percent = 60 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    return resized

# Class to process each frame of the video
class VideoProcessor():
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = resize_image(img)
        img = process_hpr(img)
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
        # selected_video = st.selectbox('Select a Video', options=list(video_urls.keys()))
        st.write("No video yet")
        # Process and display the selected video
        # if selected_video:
        #     # Ouvrir la vidéo
        #     video_url = video_urls[selected_video]
        #     video_stream = cv2.VideoCapture(video_url)
    
        #     # Créer un espace pour afficher la frame traitée
        #     stframe = st.empty()
    
        #     # Lecture frame par frame
        #     while video_stream.isOpened():
        #         ret, frame = video_stream.read()
        #         if not ret:
        #             break
    
        #         # Appliquer la fonction process() à la frame
        #         processed_frame = process(frame)
    
        #         # Afficher la frame traitée
        #         stframe.image(processed_frame, channels="BGR", use_column_width=True)
    
        #     # Relâcher la ressource vidéo
        #     video_stream.release()
        
with tab2:
    st.write("No report yet")
    st.dataframe(df)
    # df_container = st.empty()  # Créer un conteneur vide pour le DataFrame  

# À l'extérieur des onglets, mettez à jour le conteneur avec le DataFrame actuel
# df_container.dataframe(st.session_state.df)

# # Render curl counter
# # Setup status box
# cv2.rectangle(image, (0,0), (255,73), (245,117,16), -1)

# # Rep data
# cv2.putText(image, 'REPS', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
# cv2.putText(image, str(counter), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

# # Stage data
# cv2.putText(image, 'STAGE', (65,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
# cv2.putText(image, stage, (60,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)