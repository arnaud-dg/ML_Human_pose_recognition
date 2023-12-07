import cv2
import numpy as np
import pandas as pd
import av
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from facial_landmarks import FaceLandmarks
from datetime import datetime
import requests

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_detection = mp.solutions.face_detection
mp_pose = mp.solutions.pose
fl = FaceLandmarks()

st.set_page_config(layout="wide")
min_detection_confidence=0.5 
min_tracking_confidence=0.5

# Création d'un DataFrame vide avec les colonnes spécifiées
columns = ['angle_arm_l', 'angle_arm_r', 'angle_leg_l', 'angle_leg_r', 'distance_shoulder', 'distance_hip']
df = pd.DataFrame(columns=columns)



model = mp_pose.Pose() #(min_detection_confidence, min_tracking_confidence)

# Function to get the raw URL of the video file from GitHub
def get_video_url(github_path):
    content_url = f"https://api.github.com/repos/{github_path}/contents/"
    repo_content = requests.get(content_url).json()
    video_urls = {item['name']: item['download_url'] for item in repo_content if item['name'].endswith('.mp4')}
    return video_urls

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360.0 - angle
        
    return angle

def distance(a,b):
    a = np.array(a) # First
    b = np.array(b) # Mid
    # Calculate euclidian distance between a[0], a[1] and b[0], b[1]
    distance_ab = np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2 )
  
    return distance_ab

def bluring_face(frame):
    # frame = cv2.resize(frame, None, fx = 0.5, fy = 0.5)
    frame_copy = frame.copy()
    height, width, _ = frame.shape

    # 1 Face landmark detection
    landmarks = fl.get_facial_landmarks(frame)
    if landmarks is not None:
        convexhull = cv2.convexHull(landmarks)

        # 2 Face blurring
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [convexhull], True, 255, 3)
        cv2.fillConvexPoly(mask, convexhull, 255)

        frame_copy = cv2.blur(frame_copy, (27, 27))
        face_extracted = cv2.bitwise_and(frame_copy, frame_copy, mask=mask)

        # Extract background
        background_mask = cv2.bitwise_not(mask)
        background = cv2.bitwise_and(frame, frame, mask=background_mask)

        # Final result
        result = cv2.add(background, face_extracted)
        return result
    else:
        return frame  # Retourne l'image originale si aucun visage n'est détecté

def angle_extraction(landmarks):
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
    soulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    soulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    distance_shoulder = distance(shoulder_l, shoulder_r)
    distance_hip = distance(hip_l, hip_r)

    list_angle = [angle_arm_l, angle_arm_r, angle_leg_l, angle_leg_r, distance_shoulder, distance_hip]
    return list_angle

def process(image):
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
            mp_drawing.DrawingSpec(color=(171,71,0), thickness=2, circle_radius=2), #2,138,15
            mp_drawing.DrawingSpec(color=(0,128,0), thickness=2, circle_radius=2)
        )

    landmarks = results.pose_landmarks.landmark
    list_angle = angle_extraction(landmarks)
    current_time = datetime.now()
    df.loc[current_time] = list_angle
    print(df[-1,:])

    return cv2.flip(image, 1)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoProcessor():
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # print(img)
        img = process(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Page
st.title("Ergonomy Smart Assistant")

# Replace 'your-username/repo-name' with your GitHub username and repository name
video_urls = get_video_url('arnaud-dg/ML_Human_pose_recognition/contents')

video_source = st.sidebar.radio("What is your video source", ["Webcam", "Video"])
min_detection_confidence = st.sidebar.slider("Detection Threshold :", 0.0, 1.0, 0.5, 0.1)
min_tracking_confidence = st.sidebar.slider("Tracking Threshold :", 0.0, 1.0, 0.5, 0.1)
blurring_mode = st.sidebar.radio("Would you like to activate the blurring mode", ["Yes", "No"], index=1)

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
        selected_video = st.selectbox('Select a Video', options=list(video_urls.keys()))

        # Process and display the selected video
        if selected_video:
            # Ouvrir la vidéo
            video_url = video_urls[selected_video]
            video_stream = cv2.VideoCapture(video_url)
    
            # Créer un espace pour afficher la frame traitée
            stframe = st.empty()
    
            # Lecture frame par frame
            while video_stream.isOpened():
                ret, frame = video_stream.read()
                if not ret:
                    break
    
                # Appliquer la fonction process() à la frame
                processed_frame = process(frame)
    
                # Afficher la frame traitée
                stframe.image(processed_frame, channels="BGR", use_column_width=True)
    
            # Relâcher la ressource vidéo
            video_stream.release()
        
with tab2:
    st.write("No report yet")
    st.dataframe(df)    
    st.write(df.shape)
