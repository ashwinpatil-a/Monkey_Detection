# Main Code to be run
from ultralytics import YOLO
import cv2
import pygame
import time
import os
import csv
from datetime import datetime

# === CONFIGURATION ===q
MODEL_PATH = "run2/detect/monkey_detector/weights/best.pt"
ALERT_SOUND = "siren.mp3"
CONF_THRESHOLD = 0.60           # Confidence threshold
ALERT_CLASSES = ['monkey']
ALERT_COOLDOWN = 20             # Seconds before another alert
N_REQUIRED_FRAMES = 5           # Frames needed to confirm monkey
CSV_FILE = "detailed_detections.csv"  # Log file

# === INIT ===
model = YOLO(MODEL_PATH)
device = '0' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
pygame.mixer.init()
pygame.mixer.music.load(ALERT_SOUND)

# Create CSV if it doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Class', 'Confidence'])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ ERROR: Webcam not accessible.")
    exit()

print("✅ Detection started (press Q to quit).")

# === MAIN LOOP ===
monkey_detected_frames = 0
last_alert_time = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame capture failed.")
        break

    # Inference
    results = model(frame, device=device, imgsz=640, conf=CONF_THRESHOLD)
    names = results[0].names
    boxes = results[0].boxes
    now = time.time()

    monkey_present = False

    for box in boxes:
        cls_id = int(box.cls[0])
        label = names[cls_id].lower()
        conf = float(box.conf[0])

        if label in ALERT_CLASSES and conf > CONF_THRESHOLD:
            monkey_present = True

            # === Log detection to CSV ===
            with open(CSV_FILE, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    label,
                    round(conf, 2)
                ])
            break  # log only one detection per frame

    if monkey_present:
        monkey_detected_frames += 1
        cv2.putText(frame, f"🛑 Monkey Detected! ({monkey_detected_frames}/{N_REQUIRED_FRAMES})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if monkey_detected_frames >= N_REQUIRED_FRAMES and (now - last_alert_time > ALERT_COOLDOWN):
            print("🐒 MONKEY CONFIRMED! Triggering alarm...")
            pygame.mixer.music.play()
            last_alert_time = now
            monkey_detected_frames = 0
    else:
        monkey_detected_frames = max(monkey_detected_frames - 1, 0)

    # Show frame
    annotated_frame = results[0].plot()
    cv2.imshow("🧠 Monkey Detector", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === CLEANUP ===
cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()
print("🛑 Detection stopped.")
