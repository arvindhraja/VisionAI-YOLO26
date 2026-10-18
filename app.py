from __future__ import annotations

import csv
import io
import threading
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import psutil
import streamlit as st
from PIL import Image
from ultralytics import YOLO

APP_TITLE = "VisionAI | YOLO26 Live Detection"
MODEL_PATH = Path("yolo26n.pt")
SNAPSHOT_DIR = Path("snapshots")
SNAPSHOT_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: #0b1020; color: #eef2ff; }
      [data-testid="stSidebar"] { background: #10182b; }
      .hero {
        padding: 1.15rem 1.4rem; border-radius: 18px;
        background: linear-gradient(120deg, #172554, #111827 70%);
        border: 1px solid #293654; margin-bottom: 1rem;
      }
      .hero h1 { margin: 0; font-size: 2rem; color: #f8fafc; }
      .hero p { margin: .35rem 0 0; color: #b7c4dd; }
      div[data-testid="stMetric"] {
        background: #111a2e; border: 1px solid #273550;
        padding: .75rem 1rem; border-radius: 14px;
      }
      .small-note { color: #9caecb; font-size: .88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero">
      <h1>🎯 {APP_TITLE}</h1>
      <p>Real-time object detection and tracking • Local webcam processing • YOLO26</p>
    </div>
    """,
    unsafe_allow_html=True,
)


class DetectionWorker:
    """Owns the webcam and YOLO model in a background thread."""

    def __init__(self, camera_index: int, confidence: float, model_path: str):
        self.camera_index = camera_index
        self.confidence = confidence
        self.model_path = model_path
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.running = False
        self.error: str | None = None
        self.latest_jpeg: bytes | None = None
        self.latest_detections: list[dict] = []
        self.latest_counts: dict[str, int] = {}
        self.fps = 0.0
        self.inference_ms = 0.0
        self.frame_count = 0
        self.started_at: float | None = None
        self.fps_times = deque(maxlen=30)
        self.model: YOLO | None = None

    def start(self):
        if self.running:
            return
        self.stop_event.clear()
        self.error = None
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)
        self.running = False

    def set_confidence(self, value: float):
        with self.lock:
            self.confidence = float(value)

    def snapshot(self):
        with self.lock:
            return {
                "jpeg": self.latest_jpeg,
                "detections": list(self.latest_detections),
                "counts": dict(self.latest_counts),
                "fps": self.fps,
                "inference_ms": self.inference_ms,
                "frame_count": self.frame_count,
                "running": self.running,
                "error": self.error,
            }

    def _run(self):
        cap = None
        try:
            self.model = YOLO(self.model_path)
            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0)
            if not cap.isOpened():
                raise RuntimeError(
                    f"Could not open webcam index {self.camera_index}. "
                    "Close apps using the camera or try another camera index."
                )
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.started_at = time.time()

            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue

                with self.lock:
                    conf = self.confidence

                t0 = time.perf_counter()
                result = self.model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=conf,
                    verbose=False,
                )[0]
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                annotated = result.plot()

                rows: list[dict] = []
                counts: Counter[str] = Counter()
                names = result.names or {}

                if result.boxes is not None:
                    for box in result.boxes:
                        cls_id = int(box.cls[0].item()) if box.cls is not None else -1
                        label = str(names.get(cls_id, f"class_{cls_id}"))
                        confidence = float(box.conf[0].item()) if box.conf is not None else 0.0
                        track_id = None
                        if box.id is not None:
                            try:
                                track_id = int(box.id[0].item())
                            except Exception:
                                track_id = None
                        xyxy = box.xyxy[0].tolist() if box.xyxy is not None else [0, 0, 0, 0]
                        x1, y1, x2, y2 = [int(v) for v in xyxy]
                        center_x = (x1 + x2) // 2
                        frame_width = frame.shape[1]
                        if center_x < frame_width / 3:
                            direction = "Left"
                        elif center_x > frame_width * 2 / 3:
                            direction = "Right"
                        else:
                            direction = "Center"

                        rows.append(
                            {
                                "Object": label,
                                "Track ID": track_id if track_id is not None else "—",
                                "Confidence": round(confidence, 3),
                                "Direction": direction,
                                "X1": x1,
                                "Y1": y1,
                                "X2": x2,
                                "Y2": y2,
                            }
                        )
                        counts[label] += 1

                ok_jpg, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                if not ok_jpg:
                    continue

                now = time.time()
                self.fps_times.append(now)
                fps = 0.0
                if len(self.fps_times) >= 2:
                    span = self.fps_times[-1] - self.fps_times[0]
                    if span > 0:
                        fps = (len(self.fps_times) - 1) / span

                with self.lock:
                    self.latest_jpeg = encoded.tobytes()
                    self.latest_detections = rows
                    self.latest_counts = dict(counts)
                    self.inference_ms = elapsed_ms
                    self.fps = fps
                    self.frame_count += 1

        except Exception as exc:
            with self.lock:
                self.error = f"{type(exc).__name__}: {exc}"
        finally:
            if cap is not None:
                cap.release()
            with self.lock:
                self.running = False


def get_worker() -> DetectionWorker | None:
    return st.session_state.get("worker")


with st.sidebar:
    st.header("⚙️ Detection settings")
    camera_index = st.number_input("Webcam index", min_value=0, max_value=10, value=0, step=1)
    confidence = st.slider("Confidence threshold", min_value=0.10, max_value=0.90, value=0.35, step=0.05)
    st.caption("Higher confidence reduces weaker detections.")
    st.divider()
    st.subheader("Model")
    st.code("yolo26n.pt", language="text")
    st.caption("The model downloads automatically on first run if it is not already present.")
    st.divider()
    st.markdown(
        '<p class="small-note">Privacy: frames are processed by this Python process. '
        'If you expose this app through a public tunnel, visitors may be able to access the app, '
        'so use authentication or share the URL only with trusted people.</p>',
        unsafe_allow_html=True,
    )

worker = get_worker()
if worker is not None:
    worker.set_confidence(confidence)

top1, top2, top3 = st.columns([1, 1, 2])
with top1:
    if st.button("▶ Start camera", type="primary", use_container_width=True, disabled=bool(worker and worker.running)):
        new_worker = DetectionWorker(int(camera_index), confidence, str(MODEL_PATH))
        st.session_state.worker = new_worker
        new_worker.start()
        st.rerun()
with top2:
    if st.button("■ Stop camera", use_container_width=True, disabled=not bool(worker and worker.running)):
        worker.stop()
        st.rerun()
with top3:
    st.markdown(
        '<p class="small-note">Keep this app running on the PC connected to your webcam. '
        'This version uses that PC camera, not the viewer’s camera.</p>',
        unsafe_allow_html=True,
    )

worker = get_worker()
if worker is None:
    st.info("Click **Start camera** to load YOLO26 and begin detection.")
    st.markdown("### Included features")
    st.write("Live bounding boxes · ByteTrack IDs · object inventory · object distribution chart · FPS and inference timing · CPU/RAM metrics · screenshot · CSV export")
else:
    @st.fragment(run_every=0.25)
    def live_dashboard():
        snap = worker.snapshot()
        if snap["error"]:
            st.error(snap["error"])
            st.caption("Check that the webcam is connected, close other camera apps, and verify the model/dependencies.")
        if snap["jpeg"] is not None:
            col_camera, col_inventory = st.columns([1.65, 1])
            with col_camera:
                st.subheader("📹 Live camera")
                st.image(snap["jpeg"], use_container_width=True)
            with col_inventory:
                st.subheader("📦 Object inventory")
                total = sum(snap["counts"].values())
                st.metric("Objects in current frame", total)
                if snap["counts"]:
                    counts_df = pd.DataFrame(
                        [{"Object": k, "Count": v} for k, v in sorted(snap["counts"].items(), key=lambda x: (-x[1], x[0]))]
                    )
                    st.dataframe(counts_df, use_container_width=True, hide_index=True, height=260)
                    st.bar_chart(counts_df.set_index("Object"), y="Count", height=220)
                else:
                    st.write("No objects detected in the current frame.")

            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Objects", total)
            m2.metric("Inference", f'{snap["inference_ms"]:.1f} ms')
            m3.metric("FPS", f'{snap["fps"]:.1f}')
            m4.metric("CPU", f"{cpu:.0f}%")
            m5.metric("RAM", f"{ram:.0f}%")

            st.subheader("🔎 Detection details")
            if snap["detections"]:
                details = pd.DataFrame(snap["detections"])
                st.dataframe(details, use_container_width=True, hide_index=True, height=260)
                csv_bytes = details.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇ Download current detections (CSV)",
                    data=csv_bytes,
                    file_name=f"visionai_detections_{datetime.now():%Y%m%d_%H%M%S}.csv",
                    mime="text/csv",
                )
            else:
                st.caption("Detection rows will appear here when objects are visible.")

            c1, c2 = st.columns(2)
            with c1:
                if st.button("📸 Save screenshot", use_container_width=True):
                    filename = SNAPSHOT_DIR / f"visionai_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                    filename.write_bytes(snap["jpeg"])
                    st.success(f"Saved to {filename.resolve()}")
            with c2:
                st.download_button(
                    "⬇ Download current frame",
                    data=snap["jpeg"],
                    file_name=f"visionai_frame_{datetime.now():%Y%m%d_%H%M%S}.jpg",
                    mime="image/jpeg",
                    use_container_width=True,
                )
        elif snap["running"]:
            st.info("Loading YOLO26 and opening the webcam…")
        else:
            st.info("Camera stopped. Click **Start camera** to run detection again.")

    live_dashboard()
