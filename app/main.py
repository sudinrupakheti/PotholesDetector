import bisect
import csv
import math
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
HEADER = ["id", "npt_time", "video_time", "lat", "lon", "conf",
          "frame_start", "frame_end", "n_frames", "n_boxes", "gmaps_url"]

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


def fmt_video_time(sec):
    s = int(sec)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def cluster(frames):
    """frames: list of per-frame detection lists.
    A detection joins at most one event per frame, so two potholes
    in the same frame always become two events."""
    events, active = [], []
    for fdets in frames:
        if not fdets:
            continue
        t_now = fdets[0]["t"]
        active = [i for i in active
                  if (t_now - events[i][-1]["t"]).total_seconds() <= MAX_GAP]
        used = set()
        for d in fdets:
            best_i, best_s = None, None
            for i in active:
                if i in used:
                    continue
                last = events[i][-1]
                if d["lat"] is not None and last["lat"] is not None:
                    if haversine(last["lat"], last["lon"], d["lat"], d["lon"]) > MAX_DIST:
                        continue
                s = math.hypot(d["cx"] - last["cx"], d["cy"] - last["cy"])
                if best_s is None or s < best_s:
                    best_i, best_s = i, s
            if best_i is None:
                events.append([d])
                best_i = len(events) - 1
                active.append(best_i)
            else:
                events[best_i].append(d)
            used.add(best_i)
    return events


def process(video_file, gpx_file, conf, stride, offset, progress=gr.Progress()):
    if video_file is None:
        raise gr.Error("Please upload a video.")

    video, stride = Path(video_file), int(stride)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "outputs" / f"{video.stem}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video = out_dir / "annotated.mp4"

    has_gpx = gpx_file is not None

    if has_gpx:
        gpx = parse_gpx(gpx_file)
        times = [p[0] for p in gpx]
        start = times[0] + timedelta(seconds=float(offset or 0))
    else:
        gpx, times = None, []
        start = datetime.fromtimestamp(0, timezone.utc)  # relative clock only

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()

    results = get_model().predict(source=str(video), stream=True, verbose=False,
                                  conf=conf, imgsz=IMGSZ, vid_stride=stride,
                                  classes=[0])

    writer, frames = None, []
    for n, r in enumerate(results):
        frame_idx = n * stride
        t = start + timedelta(seconds=frame_idx / fps)
        count = len(r.boxes) if r.boxes is not None else 0
        frame = r.plot()

        if total:
            progress(min(frame_idx / total, 1.0), desc="Detecting")

        if has_gpx:
            lat, lon = position_at(gpx, times, t)
            put_text(frame, f"{lat:.6f}, {lon:.6f}", 34)
        else:
            lat, lon = None, None
            put_text(frame, "GPS: N/A", 34)

        put_text(frame, f"potholes in view: {count}", 64)

        if writer is None:
            h, w = r.orig_shape
            writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps / stride, (w, h))
        writer.write(frame)

        if count:
            confs = r.boxes.conf.tolist()
            centers = r.boxes.xywhn.tolist()
            frames.append([
                {"frame": frame_idx, "t": t, "lat": lat, "lon": lon, "conf": c,
                 "cx": xywh[0], "cy": xywh[1]}
                for c, xywh in zip(confs, centers)
            ])

    if writer is None:
        raise gr.Error("Could not read video.")
    writer.release()

    progress(0.97, desc="Encoding")
    tmp = out_video.with_suffix(".h264.mp4")
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out_video),
                        "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(tmp)], check=True)
        tmp.replace(out_video)
    except FileNotFoundError:
        gr.Warning("ffmpeg not found. Video may not play in browser.")
    except subprocess.CalledProcessError:
        gr.Warning("ffmpeg failed. Video may not play in browser.")
        tmp.unlink(missing_ok=True)

    rows = []
    for i, group in enumerate(cluster(frames), 1):
        best = max(group, key=lambda d: d["conf"])
        first = group[0]

        if best["lat"] is not None:
            url = f"https://www.google.com/maps/search/?api=1&query={best['lat']:.7f},{best['lon']:.7f}"
            lat_str = f"{best['lat']:.7f}"
            lon_str = f"{best['lon']:.7f}"
            npt_str = first["t"].astimezone(NPT).strftime("%H:%M:%S")
        else:
            url = lat_str = lon_str = npt_str = "N/A"

        rows.append([i, npt_str, fmt_video_time(first["frame"] / fps),
                     lat_str, lon_str, f"{best['conf']:.3f}",
                     first["frame"], group[-1]["frame"],
                     len({d["frame"] for d in group}), len(group), url])

    with open(out_dir / "potholes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)

    return str(out_video), rows


with gr.Blocks(title="Pothole Detector") as demo:
    gr.Markdown("Upload dashcam video. GPX file is optional (required for GPS mapping).")
    video_in = gr.File(label="Video")
    gpx_in = gr.File(label="GPX (Optional)", file_types=[".gpx"])
    offset_in = gr.Number(value=0, label="GPX offset (s)",
                          info="Seconds from first GPX point to video start. Negative = video starts earlier.")
    conf_in = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Confidence", info="Lower = more detections, more false alarms. Higher = fewer, stricter.")
    stride_in = gr.Slider(1, 10, value=1, step=1, label="Frame skip", info="1 = most accurate. Higher = faster, may miss potholes.")
    btn = gr.Button("Run")
    video_out = gr.Video(label="Annotated video")
    table_out = gr.Dataframe(headers=HEADER, label="Potholes")

    btn.click(process, [video_in, gpx_in, conf_in, stride_in, offset_in], [video_out, table_out])

if __name__ == "__main__":
    demo.launch()
