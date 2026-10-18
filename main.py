import csv
import time
import threading
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk
from ultralytics import YOLO
import psutil
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

APP_TITLE = "VISIONAI | Real-Time Object Intelligence"
MODEL_FILE = "yolo26n.pt"
SNAPSHOT_DIR = Path("snapshots")
SNAPSHOT_DIR.mkdir(exist_ok=True)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BG = "#0B1120"
PANEL = "#111B2E"
PANEL_2 = "#17243A"
TEXT = "#E6F0FF"
MUTED = "#91A4C2"
CYAN = "#36D6F5"
GREEN = "#38D996"
AMBER = "#FFCB70"
RED = "#FF6B7A"


class VisionAIApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1450x900")
        self.minsize(1100, 720)
        self.configure(fg_color=BG)

        self.model = None
        self.camera = None
        self.running = False
        self.worker = None
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.latest_annotated = None
        self.latest_detections = []
        self.current_conf = 0.35
        self.show_boxes = tk.BooleanVar(value=True)
        self.fps = 0.0
        self.inference_ms = 0.0
        self.total_frames = 0
        self.session_started = None
        self.history = deque(maxlen=60)
        self.event_rows = []
        self.last_ui_update = 0
        self.photo_ref = None
        self.selected_camera = tk.StringVar(value="0")

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(100, self.refresh_ui)

    def _build_ui(self):
        # Sidebar
        sidebar = ctk.CTkFrame(self, width=220, corner_radius=0, fg_color="#0E1728")
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        brand = ctk.CTkFrame(sidebar, fg_color="transparent")
        brand.pack(fill="x", padx=20, pady=(24, 30))
        ctk.CTkLabel(brand, text="◈", text_color=CYAN, font=ctk.CTkFont(size=32, weight="bold")).pack(side="left")
        brand_text = ctk.CTkFrame(brand, fg_color="transparent")
        brand_text.pack(side="left", padx=8)
        ctk.CTkLabel(brand_text, text="VISIONAI", text_color=TEXT, font=ctk.CTkFont(size=20, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(brand_text, text="OBJECT INTELLIGENCE", text_color=MUTED, font=ctk.CTkFont(size=9, weight="bold")).pack(anchor="w")

        ctk.CTkLabel(sidebar, text="WORKSPACE", text_color="#647895", font=ctk.CTkFont(size=10, weight="bold")).pack(anchor="w", padx=22, pady=(0, 8))
        self.nav_buttons = {}
        for label, glyph in [("Overview", "▦"), ("Live Detection", "◎"), ("Analytics", "▥"), ("Event History", "◷"), ("Settings", "⚙")]:
            btn = ctk.CTkButton(
                sidebar, text=f"  {glyph}   {label}", anchor="w", height=42,
                fg_color=PANEL_2 if label == "Overview" else "transparent",
                hover_color="#1C2D48", text_color=TEXT, corner_radius=10,
                font=ctk.CTkFont(size=13),
                command=lambda x=label: self.nav_notice(x)
            )
            btn.pack(fill="x", padx=12, pady=3)
            self.nav_buttons[label] = btn

        ctk.CTkFrame(sidebar, height=1, fg_color="#24334A").pack(fill="x", padx=18, pady=20)
        ctk.CTkLabel(sidebar, text="MODEL STATUS", text_color="#647895", font=ctk.CTkFont(size=10, weight="bold")).pack(anchor="w", padx=22)
        self.model_status = ctk.CTkLabel(sidebar, text="●  Loading model...", text_color=AMBER, font=ctk.CTkFont(size=12))
        self.model_status.pack(anchor="w", padx=22, pady=(8, 4))
        ctk.CTkLabel(sidebar, text="YOLO26n · ByteTrack", text_color=MUTED, font=ctk.CTkFont(size=10)).pack(anchor="w", padx=22)

        spacer = ctk.CTkFrame(sidebar, fg_color="transparent")
        spacer.pack(fill="both", expand=True)
        ctk.CTkLabel(sidebar, text="VISIONAI BUILD 1.0", text_color="#526681", font=ctk.CTkFont(size=10)).pack(anchor="w", padx=22, pady=(0, 6))
        ctk.CTkLabel(sidebar, text="Local processing • Webcam", text_color="#526681", font=ctk.CTkFont(size=10)).pack(anchor="w", padx=22, pady=(0, 20))

        # Main content
        main = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        main.pack(side="left", fill="both", expand=True)
        header = ctk.CTkFrame(main, fg_color="transparent")
        header.pack(fill="x", padx=26, pady=(22, 16))
        title_area = ctk.CTkFrame(header, fg_color="transparent")
        title_area.pack(side="left")
        ctk.CTkLabel(title_area, text="Real-Time Intelligence", text_color=TEXT, font=ctk.CTkFont(size=27, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(title_area, text="Monitor, track, and understand objects in your camera feed.", text_color=MUTED, font=ctk.CTkFont(size=12)).pack(anchor="w", pady=(4, 0))
        self.live_badge = ctk.CTkLabel(header, text="●  STANDBY", text_color=MUTED, fg_color=PANEL, corner_radius=18, padx=14, pady=8, font=ctk.CTkFont(size=11, weight="bold"))
        self.live_badge.pack(side="right", padx=(10, 0))

        # Metric cards
        metrics = ctk.CTkFrame(main, fg_color="transparent")
        metrics.pack(fill="x", padx=26, pady=(0, 15))
        self.metric_values = {}
        metric_specs = [
            ("Objects in frame", "0", "◈", CYAN),
            ("Inference speed", "0.0 ms", "⚡", GREEN),
            ("Processing FPS", "0.0", "◷", AMBER),
            ("CPU / RAM", "0% / 0%", "▤", "#B8A0FF"),
        ]
        for name, value, glyph, accent in metric_specs:
            card = ctk.CTkFrame(metrics, fg_color=PANEL, corner_radius=14, border_width=1, border_color="#1B2B43")
            card.pack(side="left", fill="x", expand=True, padx=(0, 10))
            top = ctk.CTkFrame(card, fg_color="transparent")
            top.pack(fill="x", padx=15, pady=(13, 4))
            ctk.CTkLabel(top, text=glyph, text_color=accent, font=ctk.CTkFont(size=18, weight="bold")).pack(side="left")
            ctk.CTkLabel(top, text=name, text_color=MUTED, font=ctk.CTkFont(size=11)).pack(side="left", padx=8)
            val = ctk.CTkLabel(card, text=value, text_color=TEXT, font=ctk.CTkFont(size=24, weight="bold"))
            val.pack(anchor="w", padx=15, pady=(0, 13))
            self.metric_values[name] = val

        # Body two columns
        body = ctk.CTkFrame(main, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=26, pady=(0, 24))
        left = ctk.CTkFrame(body, fg_color="transparent")
        left.pack(side="left", fill="both", expand=True, padx=(0, 12))
        right = ctk.CTkFrame(body, fg_color="transparent", width=365)
        right.pack(side="right", fill="both")
        right.pack_propagate(False)

        # Camera card
        camera_card = ctk.CTkFrame(left, fg_color=PANEL, corner_radius=14, border_width=1, border_color="#1B2B43")
        camera_card.pack(fill="both", expand=True)
        cam_head = ctk.CTkFrame(camera_card, fg_color="transparent")
        cam_head.pack(fill="x", padx=17, pady=(13, 8))
        ctk.CTkLabel(cam_head, text="LIVE CAMERA FEED", text_color=TEXT, font=ctk.CTkFont(size=12, weight="bold")).pack(side="left")
        self.camera_label = ctk.CTkLabel(camera_card, text="Camera is idle\n\nPress  START CAMERA  to begin detection", text_color=MUTED, fg_color="#080E19", corner_radius=10, font=ctk.CTkFont(size=15))
        self.camera_label.pack(fill="both", expand=True, padx=14, pady=(0, 12))

        controls = ctk.CTkFrame(camera_card, fg_color="transparent")
        controls.pack(fill="x", padx=14, pady=(0, 14))
        self.start_btn = ctk.CTkButton(controls, text="▶  Start Camera", height=38, corner_radius=9, fg_color="#087F9D", hover_color="#0A9CBD", command=self.start_camera)
        self.start_btn.pack(side="left", padx=(0, 8))
        self.stop_btn = ctk.CTkButton(controls, text="■  Stop", height=38, corner_radius=9, fg_color="#26364E", hover_color="#344967", command=self.stop_camera, state="disabled")
        self.stop_btn.pack(side="left", padx=(0, 8))
        ctk.CTkButton(controls, text="▣  Screenshot", height=38, corner_radius=9, fg_color="#26364E", hover_color="#344967", command=self.save_screenshot).pack(side="left", padx=(0, 8))
        ctk.CTkButton(controls, text="⇩  Export CSV", height=38, corner_radius=9, fg_color="#26364E", hover_color="#344967", command=self.export_csv).pack(side="left")

        settings = ctk.CTkFrame(camera_card, fg_color="transparent")
        settings.pack(fill="x", padx=17, pady=(0, 13))
        ctk.CTkLabel(settings, text="Confidence threshold", text_color=MUTED, font=ctk.CTkFont(size=11)).pack(side="left")
        self.conf_label = ctk.CTkLabel(settings, text="0.35", text_color=CYAN, font=ctk.CTkFont(size=11, weight="bold"))
        self.conf_label.pack(side="right")
        self.conf_slider = ctk.CTkSlider(settings, from_=0.10, to=0.90, number_of_steps=80, command=self.set_confidence, progress_color=CYAN, button_color=CYAN)
        self.conf_slider.set(0.35)
        self.conf_slider.pack(side="right", fill="x", expand=True, padx=12)

        # Right: object list
        object_card = ctk.CTkFrame(right, fg_color=PANEL, corner_radius=14, border_width=1, border_color="#1B2B43")
        object_card.pack(fill="both", expand=True, pady=(0, 12))
        object_head = ctk.CTkFrame(object_card, fg_color="transparent")
        object_head.pack(fill="x", padx=15, pady=(13, 6))
        ctk.CTkLabel(object_head, text="OBJECT INVENTORY", text_color=TEXT, font=ctk.CTkFont(size=12, weight="bold")).pack(side="left")
        self.object_count_badge = ctk.CTkLabel(object_head, text="0", text_color=CYAN, fg_color="#123044", corner_radius=10, padx=9, pady=3)
        self.object_count_badge.pack(side="right")
        ctk.CTkLabel(object_card, text="CLASS                     TRACK ID       CONFIDENCE", text_color="#647895", font=ctk.CTkFont(size=9, weight="bold")).pack(anchor="w", padx=15, pady=(5, 7))
        self.object_list = ctk.CTkScrollableFrame(object_card, fg_color="transparent", corner_radius=0)
        self.object_list.pack(fill="both", expand=True, padx=8, pady=(0, 10))
        self.empty_objects = ctk.CTkLabel(self.object_list, text="No objects detected yet.", text_color=MUTED, font=ctk.CTkFont(size=12))
        self.empty_objects.pack(anchor="center", pady=20)

        # Right: chart
        chart_card = ctk.CTkFrame(right, fg_color=PANEL, corner_radius=14, border_width=1, border_color="#1B2B43", height=250)
        chart_card.pack(fill="x")
        ctk.CTkLabel(chart_card, text="DETECTION DISTRIBUTION", text_color=TEXT, font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", padx=15, pady=(13, 0))
        self.figure = Figure(figsize=(4.2, 2.2), dpi=90, facecolor=PANEL)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(PANEL)
        self.ax.tick_params(colors=MUTED, labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color("#26364E")
        self.ax.set_title("Waiting for detections", color=MUTED, fontsize=9)
        self.figure.tight_layout(pad=1.8)
        self.chart_canvas = FigureCanvasTkAgg(self.figure, master=chart_card)
        self.chart_canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # Footer
        self.footer = ctk.CTkLabel(main, text="READY  •  Select Start Camera to begin  •  Processing is local to this computer", text_color="#647895", font=ctk.CTkFont(size=10), anchor="w")
        self.footer.pack(fill="x", padx=28, pady=(0, 10))

    def nav_notice(self, section):
        if section == "Settings":
            messagebox.showinfo("Settings", "Adjust confidence using the slider below the camera. More settings can be added in the next version.")
        elif section == "Event History":
            self.export_csv()
        elif section == "Analytics":
            messagebox.showinfo("Analytics", "The live detection distribution chart updates while the camera is running.")
        elif section == "Live Detection":
            self.start_camera()
        else:
            self.footer.configure(text="OVERVIEW  •  Live metrics and object inventory")

    def start_camera(self):
        if self.running:
            return
        try:
            self.model_status.configure(text="●  Loading YOLO26...", text_color=AMBER)
            self.update_idletasks()
            if self.model is None:
                self.model = YOLO(MODEL_FILE)
            cam_index = int(self.selected_camera.get())
            self.camera = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0)
            if not self.camera.isOpened():
                self.camera.release()
                self.camera = None
                messagebox.showerror("Camera error", f"Could not open camera index {cam_index}. Close other apps using the webcam and try again.")
                self.model_status.configure(text="●  Camera unavailable", text_color=RED)
                return
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
            self.running = True
            self.session_started = time.time()
            self.history.clear()
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            self.live_badge.configure(text="●  LIVE", text_color=GREEN)
            self.model_status.configure(text="●  Model ready", text_color=GREEN)
            self.footer.configure(text="LIVE  •  YOLO26n + ByteTrack  •  Press Stop to end session")
            self.worker = threading.Thread(target=self.detection_loop, daemon=True)
            self.worker.start()
        except Exception as exc:
            self.model_status.configure(text="●  Model error", text_color=RED)
            messagebox.showerror("Startup error", str(exc))

    def detection_loop(self):
        prev_time = time.perf_counter()
        while self.running and self.camera is not None:
            ok, frame = self.camera.read()
            if not ok:
                time.sleep(0.03)
                continue
            start = time.perf_counter()
            try:
                results = self.model.track(
                    frame, persist=True, tracker="bytetrack.yaml",
                    conf=float(self.current_conf), verbose=False
                )
                inference_ms = (time.perf_counter() - start) * 1000
                annotated = results[0].plot() if self.show_boxes.get() else frame.copy()
                detections = []
                boxes = results[0].boxes
                if boxes is not None:
                    names = results[0].names
                    ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
                    xyxy = boxes.xyxy.cpu().tolist()
                    confs = boxes.conf.cpu().tolist()
                    classes = boxes.cls.int().cpu().tolist()
                    for track_id, coords, conf, cls_id in zip(ids, xyxy, confs, classes):
                        x1, y1, x2, y2 = [int(v) for v in coords]
                        detections.append({
                            "class": names.get(cls_id, str(cls_id)),
                            "track_id": track_id,
                            "confidence": float(conf),
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "center_x": (x1 + x2) // 2,
                            "center_y": (y1 + y2) // 2,
                        })
                now = time.perf_counter()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now
                with self.frame_lock:
                    self.latest_frame = frame.copy()
                    self.latest_annotated = annotated.copy()
                    self.latest_detections = detections
                    self.inference_ms = inference_ms
                    self.fps = fps
                    self.total_frames += 1
                    self.history.append((time.time(), len(detections)))
                for d in detections:
                    self.event_rows.append({
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "class": d["class"],
                        "track_id": d["track_id"] if d["track_id"] is not None else "",
                        "confidence": round(d["confidence"], 4),
                        "x1": d["x1"], "y1": d["y1"], "x2": d["x2"], "y2": d["y2"],
                        "center_x": d["center_x"], "center_y": d["center_y"],
                    })
            except Exception as exc:
                with self.frame_lock:
                    self.latest_error = str(exc)
                time.sleep(0.2)

    def refresh_ui(self):
        try:
            if self.running:
                with self.frame_lock:
                    frame = self.latest_annotated.copy() if self.latest_annotated is not None else None
                    detections = list(self.latest_detections)
                    fps = self.fps
                    inference_ms = self.inference_ms
                if frame is not None:
                    self.display_frame(frame)
                cpu = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory().percent
                self.metric_values["Objects in frame"].configure(text=str(len(detections)))
                self.metric_values["Inference speed"].configure(text=f"{inference_ms:.1f} ms")
                self.metric_values["Processing FPS"].configure(text=f"{fps:.1f}")
                self.metric_values["CPU / RAM"].configure(text=f"{cpu:.0f}% / {ram:.0f}%")
                self.object_count_badge.configure(text=str(len(detections)))
                self.update_object_list(detections)
                self.update_chart(detections)
        except Exception as exc:
            self.footer.configure(text=f"UI update warning: {exc}")
        self.after(120, self.refresh_ui)

    def display_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        target_w = max(500, self.camera_label.winfo_width() - 24)
        target_h = max(300, self.camera_label.winfo_height() - 20)
        scale = min(target_w / w, target_h / h)
        size = (max(1, int(w * scale)), max(1, int(h * scale)))
        resized = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        image = Image.fromarray(resized)
        self.photo_ref = ImageTk.PhotoImage(image=image)
        self.camera_label.configure(image=self.photo_ref, text="")

    def update_object_list(self, detections):
        for child in self.object_list.winfo_children():
            child.destroy()
        if not detections:
            ctk.CTkLabel(self.object_list, text="No objects detected in this frame.", text_color=MUTED, font=ctk.CTkFont(size=12)).pack(anchor="center", pady=20)
            return
        for d in sorted(detections, key=lambda x: x["confidence"], reverse=True)[:40]:
            row = ctk.CTkFrame(self.object_list, fg_color=PANEL_2, corner_radius=8)
            row.pack(fill="x", padx=3, pady=3)
            ctk.CTkLabel(row, text=d["class"], text_color=TEXT, font=ctk.CTkFont(size=11, weight="bold"), anchor="w", width=115).pack(side="left", padx=(9, 4), pady=8)
            tid = f"#{d['track_id']}" if d["track_id"] is not None else "—"
            ctk.CTkLabel(row, text=tid, text_color=CYAN, font=ctk.CTkFont(size=10), width=65).pack(side="left")
            ctk.CTkLabel(row, text=f"{d['confidence']*100:.0f}%", text_color=GREEN if d["confidence"] >= 0.7 else AMBER, font=ctk.CTkFont(size=10, weight="bold"), width=55).pack(side="right", padx=7)

    def update_chart(self, detections):
        counts = Counter(d["class"] for d in detections)
        self.ax.clear()
        self.ax.set_facecolor(PANEL)
        self.ax.tick_params(colors=MUTED, labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color("#26364E")
        if counts:
            items = counts.most_common(7)
            labels = [x[0] for x in items]
            values = [x[1] for x in items]
            bars = self.ax.barh(labels[::-1], values[::-1], color=CYAN, height=0.58)
            self.ax.set_title("Objects in current frame", color=TEXT, fontsize=9, loc="left", pad=8)
            self.ax.tick_params(axis="y", colors=TEXT, labelsize=8)
            self.ax.tick_params(axis="x", colors=MUTED, labelsize=7)
            self.ax.grid(axis="x", color="#26364E", alpha=0.6, linewidth=0.6)
            self.ax.set_axisbelow(True)
            self.ax.spines["top"].set_visible(False)
            self.ax.spines["right"].set_visible(False)
            self.ax.bar_label(bars, padding=3, color=TEXT, fontsize=8)
            self.ax.set_xlim(0, max(values) + 1)
        else:
            self.ax.set_title("No detections in current frame", color=MUTED, fontsize=9)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
        self.figure.tight_layout(pad=1.6)
        self.chart_canvas.draw_idle()

    def set_confidence(self, value):
        self.current_conf = float(value)
        self.conf_label.configure(text=f"{self.current_conf:.2f}")

    def stop_camera(self):
        self.running = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.live_badge.configure(text="●  STANDBY", text_color=MUTED)
        self.model_status.configure(text="●  Model ready" if self.model else "●  Not loaded", text_color=GREEN if self.model else MUTED)
        self.footer.configure(text="STOPPED  •  Session data remains available for CSV export")
        self.camera_label.configure(image=None, text="Camera stopped\n\nPress  START CAMERA  to resume", text_color=MUTED)
        self.photo_ref = None

    def save_screenshot(self):
        with self.frame_lock:
            frame = self.latest_annotated.copy() if self.latest_annotated is not None else None
        if frame is None:
            messagebox.showinfo("Screenshot", "Start the camera first, then capture a screenshot.")
            return
        filename = SNAPSHOT_DIR / f"visionai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(str(filename), frame)
        messagebox.showinfo("Screenshot saved", f"Saved to:\n{filename.resolve()}")

    def export_csv(self):
        if not self.event_rows:
            messagebox.showinfo("Export CSV", "No detection events to export yet. Start the camera first.")
            return
        path = filedialog.asksaveasfilename(
            title="Export detection history", defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile=f"visionai_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.event_rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.event_rows)
        messagebox.showinfo("Export complete", f"Exported {len(self.event_rows)} detection records.")

    def on_close(self):
        self.stop_camera()
        self.destroy()


if __name__ == "__main__":
    # Keep OpenCV / torch logs from dominating the console where possible.
    app = VisionAIApp()
    app.mainloop()
