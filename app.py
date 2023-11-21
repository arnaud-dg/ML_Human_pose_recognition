from ultralytics import YOLO
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import numpy as np
from PIL import Image
import av
import cv2
import settings

# Setting page layout
st.set_page_config(
    page_title="Webcam ergonomy detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("Webcam ergonomy detection using YOLOv8")

# Load Pre-trained ML Model
try:
    model = YOLO('yolov8n-pose.pt')
except Exception as ex:
    st.error(f"Unable to load model.")
    st.error(ex)

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    flipped = img[::-1,:,:]

    return av.VideoFrame.from_ndarray(flipped, format="bgr24")

webrtc_streamer(
    key="example", 
    video_frame_callback=video_frame_callback, 
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    )


# def display_tracker_options():
#     display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
#     is_display_tracker = True if display_tracker == 'Yes' else False
#     if is_display_tracker:
#         tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
#         return is_display_tracker, tracker_type
#     return is_display_tracker, None

# def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
#     """
#     Display the detected objects on a video frame using the YOLOv8 model.

#     Args:
#     - conf (float): Confidence threshold for object detection.
#     - model (YoloV8): A YOLOv8 object detection model.
#     - st_frame (Streamlit object): A Streamlit object to display the detected video.
#     - image (numpy array): A numpy array representing the video frame.
#     - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

#     Returns:
#     None
#     """

#     # Resize the image to a standard size
#     image = cv2.resize(image, (720, int(720*(9/16))))

#     # Display object tracking, if specified
#     if is_display_tracking:
#         res = model.track(image, conf=conf, persist=True, tracker=tracker)
#     else:
#         # Predict the objects in the image using the YOLOv8 model
#         res = model.predict(image, conf=conf)

#     # # Plot the detected objects on the video frame
#     res_plotted = res[0].plot()
#     st_frame.image(res_plotted,
#                    caption='Detected Video',
#                    channels="BGR",
#                    use_column_width=True
#                    )

# def play_webcam(conf, model):
#     """
#     Plays a webcam stream. Detects Objects in real-time using the YOLO object detection model.

#     Returns:
#         None

#     Raises:
#         None
#     """
#     webrtc_streamer(
#         key="example",
#         video_transformer_factory=lambda: MyVideoTransformer(conf, model),
#         # rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
#         media_stream_constraints={"video": True, "audio": False},
#     )

# class MyVideoTransformer(VideoTransformerBase):
#     def __init__(self, conf, model):
#         self.conf = conf
#         self.model = model

#     def recv(self, frame):
#         image = frame.to_ndarray(format="bgr24")
#         processed_image = self._display_detected_frames(image)
#         st.image(processed_image, caption='Detected Video', channels="BGR", use_column_width=True)

#     def _display_detected_frames(self, image):
#         orig_h, orig_w = image.shape[0:2]
#         width = 720  # Set the desired width for processing

#         # cv2.resize used in a forked thread may cause memory leaks
#         input = np.asarray(Image.fromarray(image).resize((width, int(width * orig_h / orig_w))))

#         if self.model is not None:
#             # Perform object detection using YOLO model
#             res = self.model.predict(input, conf=self.conf)

#             # Plot the detected objects on the video frame
#             res_plotted = res[0].plot()
#             return res_plotted

#         return input

# confidence = 50
# play_webcam(confidence, model)
