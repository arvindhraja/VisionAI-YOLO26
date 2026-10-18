from pathlib import Path

readme = r"""# 🎯 VisionAI-YOLO26

**Real-Time Object Detection and Tracking Dashboard using YOLO26, OpenCV, and Python**

VisionAI-YOLO26 is a computer vision project that detects and tracks objects in real time using a webcam. It combines YOLO26 object detection with ByteTrack tracking and provides dashboards for visualizing detections, object counts, and performance metrics.

## ✨ Features

- 🎥 Real-time webcam object detection
- 🎯 YOLO26 object detection model
- 🆔 Object tracking with ByteTrack
- 📦 Live object inventory and class counts
- 📊 Object distribution charts
- ⚡ Inference time and FPS monitoring
- 💻 CPU and RAM usage monitoring
- 🔍 Adjustable detection confidence threshold
- 📋 Detection details, tracking IDs, and object direction
- 📸 Screenshot capture and image download
- 📁 Export detection results as CSV
- 🌐 Streamlit browser-based dashboard
- 🖥️ CustomTkinter desktop dashboard

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Ultralytics YOLO26 | Object detection |
| OpenCV | Webcam and image processing |
| ByteTrack | Multi-object tracking |
| Streamlit | Browser-based interface |
| CustomTkinter | Desktop interface |
| Pandas | Detection data handling |
| Streamlit charts | Data visualization |
| psutil | CPU and memory monitoring |

## 📂 Project Structure

```text
VisionAI-YOLO26/
├── app.py              # Streamlit web dashboard
├── main.py             # CustomTkinter desktop dashboard
├── requirements.txt    # Python dependencies
├── .gitignore          # Files excluded from Git
└── README.md           # Project documentation
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/arvindhraja/VisionAI-YOLO26.git
cd VisionAI-YOLO26
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the browser dashboard

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

### 5. Run the desktop application

```bash
python main.py
```

The desktop version may need additional dependencies beyond `requirements.txt`, depending on the packages imported by `main.py`.

## ⚙️ Requirements

- Python version compatible with the installed dependencies
- A webcam for live detection
- Internet access for initial dependency installation and model download
- Windows, Linux, or macOS environment supported by the dependencies

A compatible NVIDIA GPU can improve inference performance, but CPU execution may also be possible at lower speeds.

## 🔎 How It Works

1. Capture frames from a webcam.
2. Process each frame using the YOLO26 model.
3. Track detected objects across frames with ByteTrack.
4. Draw bounding boxes and display detection information.
5. Calculate object counts, inference time, and FPS.
6. Visualize detection statistics in the dashboard.
7. Export detection details for further analysis.

## 🔒 Privacy and Deployment

The local webcam version processes video on the machine running the application. A cloud-hosted deployment does not automatically have access to a webcam connected to your personal computer.

For local processing with remote access, use an appropriately secured remote-access setup and keep the local application running. Do not expose an unauthenticated camera dashboard to the public internet.

## 🎯 Project Goals

- Demonstrate real-time computer vision.
- Explore object detection and multi-object tracking.
- Monitor model performance.
- Present detection results through an interactive dashboard.
- Build a foundation for future computer vision applications.

## 🔮 Future Improvements

- Object trajectory visualization
- Detection history and analytics
- Configurable camera selection
- Custom-trained object detection models
- Improved distance estimation
- Secure remote dashboard access
- Detection event logging and reporting

## 👨‍💻 Author

**Arvindh Raja**

- GitHub: [@arvindhraja](https://github.com/arvindhraja)
- Repository: [VisionAI-YOLO26](https://github.com/arvindhraja/VisionAI-YOLO26)

---

*Built with Python and computer vision technology.*
"""
path = Path("/mnt/data/README.md")
path.write_text(readme, encoding="utf-8")
print(f"Created README.md at {path}")
