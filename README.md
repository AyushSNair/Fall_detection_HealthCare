# 🧠 Fall Detection using YOLOv8-OBB + MediaPipe Pose

This project detects human falls in real-time by combining **YOLOv8 Oriented Bounding Boxes (OBB)** with **MediaPipe Pose Estimation**. It verifies fall predictions based on the **spine angle**, minimizing false alarms.

## 📌 Features

- ✅ Real-time fall detection using YOLOv8-OBB.
- ✅ Human posture verification using MediaPipe Pose.
- ✅ Spine angle calculation to confirm falls.
- ✅ Distinguishes between `Fall`, `False Alarm`, and `Non-fall`.
- ✅ Bounding boxes labeled accordingly on each detection.

## 📸 Sample Output

- `Fall`: Detected by YOLOv8 and confirmed by abnormal spine angle.
- `False Alarm`: YOLOv8 detects a fall, but spine angle is normal.
- `Non-fall`: Normal human activity.

## 🧠 How It Works

1. **YOLOv8-OBB** predicts bounding boxes and labels (e.g., "fall", "bending").
2. **MediaPipe Pose** detects key landmarks on the human body.
3. **Spine angle** is computed using shoulder and hip landmarks.
4. Final decision is:
   - `Fall` → YOLO label is "fall" **AND** spine angle is unnatural.
   - `False Alarm` → YOLO label is "fall" **BUT** posture is normal.
   - `Non-fall` → Any other prediction.

## 🧮 Spine Angle Logic

- Calculates the angle between the **mid-shoulder** and **mid-hip** line.
- If spine is nearly horizontal (`< 45° or > 135°`) and the vertical distance is low → **unnatural posture**.

## 🚀 Installation

```bash
git clone https://github.com/YOUR_USERNAME/fall-detection-yolov8-mediapipe.git
cd fall-detection-yolov8-mediapipe
pip install -r requirements.txt
