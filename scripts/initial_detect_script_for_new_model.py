from ultralytics import YOLO

model = YOLO("best.pt")
model.predict(source="dashcam.mp4", classes=[0], conf=0.25, imgsz=960, save=True)