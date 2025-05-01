import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from drive_upload import upload_to_drive
import math
import time
import threading
import queue
import os
from flask import Flask, Response, jsonify
from twilio.rest import Client
from dotenv import load_dotenv
import tempfile

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, static_folder='static')

# Load the trained YOLOv8-OBB model
yolo_model = YOLO("best.pt")  # Update with your YOLOv8-OBB model path

# Load the Keras model
keras_model_path = "fall_detection_model.h5"  # Update with your Keras model path
if not os.path.exists(keras_model_path):
    print(f"Error: Keras model file {keras_model_path} not found")
    exit()
keras_model = tf.keras.models.load_model(keras_model_path)
print("Keras model loaded successfully")

# Define Keras label encoder (same as training: fall, non_fall, bending)
keras_label_encoder = LabelEncoder()
keras_label_encoder.fit(["fall", "non_fall", "bending"])

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Twilio config
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_WHATSAPP_FROM = os.getenv('TWILIO_WHATSAPP_FROM')
WHATSAPP_TO = os.getenv('WHATSAPP_TO')

if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM, WHATSAPP_TO]):
    raise ValueError("Missing Twilio credentials")

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Video setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Global variables
fall_detected_global = False
last_notification_time = 0
NOTIFICATION_COOLDOWN = 60
frame_queue = queue.Queue(maxsize=1)
recording = False
frames_to_record = []
RECORD_DURATION = 7
FPS = 30
last_fall_time = 0
FALL_DEBOUNCE_TIME = 2
buffer_size = 60
frames_buffer = []

# Function to calculate angle between two points
def calculate_angle(p1, p2):
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    radians = math.atan2(dy, dx)
    angle = abs(math.degrees(radians))
    return angle

# Function to extract keypoints for Keras model
def extract_keypoints_for_keras(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y])
        return np.array(keypoints)
    return None

# Fall detection with YOLOv8, MediaPipe, and Keras
def detect_fall(frame):
    try:
        # Step 1: YOLOv8-OBB detection
        results = yolo_model.predict(source=frame, task='obb', verbose=False)
        yolo_fall_detected = False
        processed_frame = frame.copy()

        # Initialize variables
        final_fall = False
        unnatural_posture = False
        keras_pred_class = "N/A"
        keras_confidence = 0.0

        # YOLOv8 processing
        for result in results:
            if result.obb is not None:
                obbs = result.obb
                for i in range(len(obbs.xyxyxyxy)):
                    confidence = obbs.conf[i].item()
                    class_id = int(obbs.cls[i])
                    class_name = result.names[class_id].lower()

                    poly_points = obbs.xyxyxyxy[i].cpu().numpy().reshape(-1, 2).astype(int)
                    aabb = obbs.xyxy[i].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = aabb

                    label = "Non-fall"
                    color = (0, 255, 0)

                    if class_name == "fall" and confidence >= 0.5:
                        yolo_fall_detected = True
                        label = "Fall (YOLO)"
                        color = (0, 0, 255)

                    cv2.polylines(processed_frame, [poly_points], isClosed=True, color=color, thickness=2)
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(processed_frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Step 2: Pose estimation if YOLO detects a fall
        if yolo_fall_detected:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(rgb_frame)

            if pose_results.pose_landmarks:
                lm = pose_results.pose_landmarks.landmark

                # Draw pose landmarks
                mp_drawing.draw_landmarks(
                    processed_frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

                # Step 3: Analyze pose geometry
                left_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]

                mid_shoulder = type(left_shoulder)(x=(left_shoulder.x + right_shoulder.x) / 2,
                                                 y=(left_shoulder.y + right_shoulder.y) / 2,
                                                 z=0, visibility=1.0)
                mid_hip = type(left_hip)(x=(left_hip.x + right_hip.x) / 2,
                                        y=(left_hip.y + right_hip.y) / 2,
                                        z=0, visibility=1.0)

                vertical_distance = abs(mid_shoulder.y - mid_hip.y)
                spine_angle = calculate_angle(mid_shoulder, mid_hip)

                if vertical_distance < 0.1 and (spine_angle < 45 or spine_angle > 135):
                    unnatural_posture = True
                    cv2.putText(processed_frame, "Unnatural posture", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Step 4: Keras model validation
                keypoints = extract_keypoints_for_keras(frame)
                if keypoints is not None:
                    keypoints = keypoints.reshape(1, -1)
                    keras_pred = keras_model.predict(keypoints, verbose=0)
                    keras_pred_class = keras_label_encoder.inverse_transform([np.argmax(keras_pred)])[0]
                    keras_confidence = np.max(keras_pred) * 100

                    cv2.putText(processed_frame, f"Keras: {keras_pred_class} ({keras_confidence:.2f}%)", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                    # Step 5: Fusion logic
                    if keras_pred_class == "fall" and unnatural_posture:
                        final_fall = True
                        cv2.putText(processed_frame, "Confirmed Fall", (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    elif keras_pred_class in ["non_fall", "bending"]:
                        cv2.putText(processed_frame, "False Alarm", (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return final_fall, processed_frame

    except Exception as e:
        print(f"Error in detect_fall: {e}")
        return False, frame

# Record fall video
def record_fall_video():
    global recording, frames_to_record
    recording = True
    frames_to_record = frames_buffer[-buffer_size:]  # Include pre-fall frames
    start_time = time.time()

    while recording and (time.time() - start_time) < RECORD_DURATION:
        success, frame = cap.read()
        if success:
            frames_to_record.append(frame.copy())
        time.sleep(1 / FPS)

    recording = False
    if frames_to_record:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(tempfile.gettempdir(), f"fall_{timestamp}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, FPS, (640, 480))

        for frame in frames_to_record:
            out.write(frame)
        out.release()

        try:
            return upload_to_drive(video_path)  # Assumes upload_to_drive is defined
        except Exception as e:
            print(f"Drive upload error: {e}")
            return None
    return None

def send_whatsapp_notification(timestamp, video_link=None):
# Send WhatsApp alert
    try:
        message_body = f"🚨 Fall Detected at {timestamp}!"
        if video_link:
            message_body += f"\n📹 Video: {video_link}"

        message = twilio_client.messages.create(
            body=message_body,
            from_=TWILIO_WHATSAPP_FROM,
            to=WHATSAPP_TO
        )
        print(f"Message sent: SID {message.sid}")
        return True
    except Exception as e:
        print(f"WhatsApp send error: {e}")
        return False

# Notify and upload
def record_and_send_notification():
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    video_link = record_fall_video()
    send_whatsapp_notification(timestamp, video_link)

# Inference thread
def inference_thread():
    global fall_detected_global, last_fall_time
    while True:
        frame = frame_queue.get()
        fall_detected, _ = detect_fall(frame)
        current_time = time.time()

        if fall_detected and (current_time - last_fall_time > FALL_DEBOUNCE_TIME):
            fall_detected_global = True
            last_fall_time = current_time
            threading.Thread(target=record_and_send_notification, daemon=True).start()
        else:
            fall_detected_global = False
        frame_queue.task_done()

# Frame generator for Flask
def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        frames_buffer.append(frame.copy())
        if len(frames_buffer) > buffer_size:
            frames_buffer.pop(0)

        _, processed_frame = detect_fall(frame.copy())

        if frame_queue.qsize() < 1:
            frame_queue.put(frame.copy())

        ret, buffer = cv2.imencode('.jpg', processed_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# Flask routes
@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/fall_status')
def fall_status():
    success, frame = cap.read()
    if success:
        fall_detected, _ = detect_fall(frame)
        if fall_detected:
            threading.Thread(target=record_and_send_notification, daemon=True).start()
        return jsonify({
            "status": "Fall Detected" if fall_detected else "Safe",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
    return jsonify({"error": "Camera read error"})

# Main execution
if __name__ == "__main__":
    print("Starting Flask app...")
    threading.Thread(target=inference_thread, daemon=True).start()
    try:
        app.run(host="0.0.0.0", port=5000, debug=False)
    finally:
        cap.release()
        pose.close()
        print("Program terminated")