# https://pysource.com/blur-faces-in-real-time-with-opencv-mediapipe-and-python
import mediapipe as mp
import cv2
import numpy as np
from datetime import datetime
import requests
import cv2
import pandas as pd
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration


##########################
#         Classes        #
##########################
class FaceLandmarks:
    def __init__(self):
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh()

    def get_facial_landmarks(self, frame):
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(frame_rgb)

        facelandmarks = []
        for facial_landmarks in result.multi_face_landmarks:
            for i in range(0, 468):
                pt1 = facial_landmarks.landmark[i]
                x = int(pt1.x * width)
                y = int(pt1.y * height)
                facelandmarks.append([x, y])
        return np.array(facelandmarks, np.int32)

##########################
# Trigonometry functions #
##########################    
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
    soulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    soulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    distance_shoulder = distance(shoulder_l, shoulder_r)
    distance_hip = distance(hip_l, hip_r)

    list_angle = [angle_arm_l, angle_arm_r, angle_leg_l, angle_leg_r, distance_shoulder, distance_hip]
    return list_angle

##########################
#     Image Analysis     #
##########################    
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
    


# Render curl counter
# Setup status box
# cv2.rectangle(image, (0,0), (255,73), (245,117,16), -1)

# Rep data
# cv2.putText(image, 'REPS', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
# cv2.putText(image, str(counter), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

# Stage data
# cv2.putText(image, 'STAGE', (65,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
# cv2.putText(image, stage, (60,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)