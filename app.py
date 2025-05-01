from ultralytics import YOLO
import cv2
import time
import threading
import queue
import os
from flask import Flask, Response, jsonify
from twilio.rest import Client
from dotenv import load_dotenv
from drive_upload import upload_to_drive
import tempfile

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, static_folder='static')

# Load the trained YOLOv8-OBB model
model = YOLO("best.pt")  # Replace with your model path

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
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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
cooldown_time = 10  # seconds to wait before detecting another fall
last_detection_time = 0
is_recording = False
record_start_time = 0
frames_buffer = []
buffer_size = 60  # Store 60 frames (~2 sec) before fall

# Thread for real-time inference
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

# ⚠️ YOLOv8-OBB fall detection
def detect_fall(frame):
    try:
        results = model.predict(source=frame, task='obb', verbose=False)
        fall_detected = False

        for result in results:
            if result.obb is not None:
                obbs = result.obb

                for i in range(len(obbs.xyxyxyxy)):
                    confidence = obbs.conf[i].item()
                    class_id = int(obbs.cls[i])
                    class_name = result.names[class_id].lower()

                    poly_points = obbs.xyxyxyxy[i].cpu().numpy().reshape(-1, 2).astype(int)
                    aabb = obbs.xyxy[i].cpu().numpy().astype(int)

                    # Default label and color
                    label = "non-fall"
                    color = (0, 255, 0)  # Green

                    if class_name in ["fall", "bending"] and confidence >= 0.7:
                        label = class_name
                        color = (0, 0, 255)  # Red
                        if class_name == "fall":
                            fall_detected = True

                    # Draw OBB
                    cv2.polylines(frame, [poly_points], isClosed=True, color=color, thickness=2)

                    # Draw AABB
                    x1, y1, x2, y2 = aabb
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Put label
                    cv2.putText(
                        frame,
                        f"{label} ({confidence:.2f})",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )

        return fall_detected, frame

    except Exception as e:
        print("Error in detect_fall:", e)
        return False, frame


# Record fall video
def record_fall_video():
    global recording, frames_to_record
    recording = True
    frames_to_record = []
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
            return upload_to_drive(video_path)
        except Exception as e:
            print(f"Drive upload error: {e}")
            return None
    return None

# Send WhatsApp alert
def send_whatsapp_notification(timestamp, video_link=None):
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

# Frame generator for Flask route
def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Run detection and draw boxes
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

# Run app
if __name__ == "__main__":
    import numpy as np  # Required for drawing polygons
    print("Starting Flask app...")
    threading.Thread(target=inference_thread, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=True)
