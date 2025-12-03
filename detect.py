# monkey_detect_live.py
from ultralytics import YOLO

def live_webcam_detection():
    model = YOLO("runs/detect/monkey_detector/weights/best.pt")

    model.predict(
        source=0,           # 0 = default webcam
        device="0",         # GPU
        show=True,          # Show live video
        conf=0.5,
        imgsz=640
    )

    print("\n📷 Live webcam detection started. Press 'q' or close window to exit.")

if __name__ == "__main__":
    live_webcam_detection()
