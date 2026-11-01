import time

import cv2
from ultralytics import YOLO

model = YOLO("best.pt")

video_path = "dashcam.mp4"

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    "output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

start_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = model.predict(frame, conf=0.5, verbose=False)
    annotated_frame = result[0].plot()
    out.write(annotated_frame)

    frame_count += 1
    if frame_count % 30 == 0:
        elapsed = time.time() - start_time
        fps_actual = frame_count / elapsed
        print(f"Processed {frame_count} frames | {elapsed:.1f}s | {fps_actual:.1f} fps")

cap.release()
out.release()

total_time = time.time() - start_time
print(f"✓ Output saved: output.mp4")
print(
    f"Total time: {total_time:.1f}s | {frame_count} frames | {frame_count / total_time:.1f} fps"
)
