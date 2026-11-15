import bisect
import csv
import math
import re
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import gradio as gr
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "weights" / "best.pt"
NPT = timezone(timedelta(hours=5, minutes=45))
IMGSZ = 960
MAX_GAP = 1.5
MAX_DIST = 20.0
HEADER = ["id", "npt_time", "lat", "lon", "conf",
          "frame_start", "frame_end", "n_detections", "gmaps_url"]

_model = None


def get_model():
    global _model
    if _model is None:
        _model = YOLO(str(MODEL_PATH))
    return _model


def parse_gpx(path):
    pts = []
    for el in ET.parse(path).iter():
        if not el.tag.endswith("trkpt"):
            continue
        t = next((c.text for c in el if c.tag.endswith("time") and c.text), None)
        if t:
            dt = datetime.fromisoformat(t.strip().replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            pts.append((dt, float(el.get("lat")), float(el.get("lon"))))
    if not pts:
        raise gr.Error("No timed trackpoints in GPX.")
    return sorted(pts)


def video_start(video, first_fix):
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})[ _](\d{2})-(\d{2})-(\d{2})(?:\.(\d+))?",
                  video.name)
    if not m:
        return first_fix
    start = datetime(*map(int, m.groups()[:6]), tzinfo=NPT).astimezone(timezone.utc)
    return start + timedelta(seconds=float(f"0.{m.group(7)}") if m.group(7) else 0)


def position_at(gpx, times, t):
    i = bisect.bisect_right(times, t)
    if i == 0:
        return gpx[0][1:]
    if i == len(gpx):
        return gpx[-1][1:]
    (t0, la0, lo0), (t1, la1, lo1) = gpx[i - 1], gpx[i]
    span = (t1 - t0).total_seconds()
    f = (t - t0).total_seconds() / span if span else 0
    return la0 + (la1 - la0) * f, lo0 + (lo1 - lo0) * f


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6371000.0 * math.asin(math.sqrt(a))


def put_text(frame, text, y):
    cv2.putText(frame, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                (255, 255, 255), 2, cv2.LINE_AA)


def cluster(dets):
    events = []
    for d in dets:
        if events:
            last = events[-1][-1]
            if ((d["t"] - last["t"]).total_seconds() <= MAX_GAP
                    and haversine(last["lat"], last["lon"], d["lat"], d["lon"]) <= MAX_DIST):
                events[-1].append(d)
                continue
        events.append([d])
    return events


def process(video_file, gpx_file, conf, stride):
    if video_file is None or gpx_file is None:
        raise gr.Error("Upload both video and GPX.")
    video, stride = Path(video_file), int(stride)
    out_dir = ROOT / "outputs" / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video = out_dir / "annotated.mp4"

    gpx = parse_gpx(gpx_file)
    times = [p[0] for p in gpx]
    start = video_start(video, times[0])

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    results = get_model().predict(source=str(video), stream=True, verbose=False,
                                  conf=conf, imgsz=IMGSZ, vid_stride=stride,
                                  classes=[0])
    writer, dets = None, []
    for n, r in enumerate(results):
        frame_idx = n * stride
        t = start + timedelta(seconds=frame_idx / fps)
        lat, lon = position_at(gpx, times, t)
        count = len(r.boxes) if r.boxes is not None else 0

        frame = r.plot()
        put_text(frame, f"{lat:.6f}, {lon:.6f}", 34)
        put_text(frame, f"potholes in view: {count}", 64)
        if writer is None:
            h, w = r.orig_shape
            writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps / stride, (w, h))
        writer.write(frame)

        if count:
            for c in r.boxes.conf.tolist():
                dets.append({"frame": frame_idx, "t": t, "lat": lat, "lon": lon, "conf": c})

    if writer is None:
        raise gr.Error("Could not read video.")
    writer.release()

    tmp = out_video.with_suffix(".h264.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out_video),
                    "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(tmp)], check=True)
    tmp.replace(out_video)

    rows = []
    for i, group in enumerate(cluster(dets), 1):
        best = max(group, key=lambda d: d["conf"])
        url = f"https://www.google.com/maps/search/?api=1&query={best['lat']:.7f},{best['lon']:.7f}"
        rows.append([i, group[0]["t"].astimezone(NPT).strftime("%H:%M:%S"),
                     f"{best['lat']:.7f}", f"{best['lon']:.7f}", f"{best['conf']:.3f}",
                     group[0]["frame"], group[-1]["frame"], len(group), url])

    with open(out_dir / "potholes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)

    return str(out_video), rows

with gr.Blocks(title="Pothole Detector") as demo:
    gr.Markdown("Upload dashcam video and matching GPX file.")
    video_in = gr.File(label="Video")
    gpx_in = gr.File(label="GPX", file_types=[".gpx"])
    conf_in = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Confidence", info="Lower = more detections, more false alarms. Higher = fewer, stricter.")
    stride_in = gr.Slider(1, 10, value=1, step=1, label="Frame skip", info="1 = most accurate. Higher = faster, may miss potholes.")
    btn = gr.Button("Run")
    video_out = gr.Video(label="Annotated video")
    table_out = gr.Dataframe(headers=HEADER, label="Potholes")
    btn.click(process, [video_in, gpx_in, conf_in, stride_in], [video_out, table_out])

if __name__ == "__main__":
    demo.launch()
