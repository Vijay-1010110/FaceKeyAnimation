# FaceKey Animation Studio — Cloud-Ready Facial Motion AI Dataset & Training System

A high-performance pipeline for acquiring clean, timestamped, temporally consistent facial motion capture datasets and training neural speech-to-animation models.

Designed for **both Local PCs** (Windows 11 low-resource execution) and **Cloud Environments** (**Google Colab Free Tier with 16GB NVIDIA T4 GPUs** + **Google Drive automated chunked storage**).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vijay-1010110/FaceKeyAnimation/blob/main/notebooks/1_Colab_Data_Collector.ipynb)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/Vijay-1010110/FaceKeyAnimation)

---

## ⚡ Cloud & Local Processing Modes

- ☁️ **Google Colab Cloud Collection & Training**:
  - Run heavy YouTube extraction and PyTorch T4 GPU training entirely in the cloud without using local PC disk space or bandwidth!
  - Stores compressed `.tar.gz` chunks and checkpoints directly in **Google Drive** (`/MyDrive/FaceKeyDataset/`).
  - **Auto-resumes seamlessly** if Colab disconnects or reaches runtime limits.
  - Read the complete [Google Drive & Colab Guide](GOOGLE_DRIVE_AND_COLAB_GUIDE.md).
- 🚀 **Silent 480p Batch YouTube Queue**:
  - Ingests links continuously from `sessions/youtubeURLtoProcess.txt`.
  - Add links anytime while running; auto-resumes from save points if closed.
- 🎯 **Phase 1 Training Milestone (100 Hours Target)**:
  - Tracks accumulated clean animation hours toward 100 Hours (~48 GB estimated).
  - Scales comfortably to 1,000 Hours (~480 GB).

---

## 🚀 Key Architectural Principles

1. **Video is an observation source, not the dataset**:
   - Frames are observed ephemerally in RAM.
   - 478 3D landmarks, 52 ARKit-compatible blendshapes, head pose Euler angles, and quality scores are extracted.
   - Raw image frames are discarded immediately after feature extraction. No intermediate PNGs, JPEGs, or temporary video files are written to disk.
2. **Master Session Timeline**:
   - Master reference timestamp $t \ge 0.000$s aligns audio energy, facial landmarks, head pose, and quality metrics across all streams.
3. **Multi-Stage Facial Perception**:
   - **Perception**: MediaPipe Tasks `FaceLandmarker` with CPU XNNPACK SIMD acceleration (~19ms latency, ~50 FPS), leaving GPU VRAM free.
   - **Tracking**: Spatial-temporal Hungarian centroid/IoU matching assigning persistent `face_id` across frames with Primary Face selection.
   - **Quality Scoring**: Multi-metric evaluation (Laplacian variance sharpness, face area ratio, pose limit penalties, velocity jitter) with enter/exit hysteresis to eliminate state flickering.
   - **Speaker Attribution**: Audio VAD (RMS energy + zero-crossing rate) correlated with visual mouth dynamics ($\Delta \text{jawOpen}/\Delta t$) to distinguish `SPEAKER` vs `LISTENER` vs off-screen `NARRATION/VOICEOVER`.
   - **Eligibility Classification**: Multi-level hierarchy (Levels 0-4) separating raw detection from usable animation and speaker-paired datasets.
   - **Temporal Filtering**: Adaptive One Euro Filter removing measurement jitter while preserving fast intentional movements (blinks, consonant mouth bursts).
   - **Data Integrity**: Raw observed values and clean reconstructed values are stored in separate layers with strict provenance tracking (`is_observed` vs `is_reconstructed`).
4. **Structured Binary Storage**:
   - Completed sessions are stored in `sessions/session_<ID>/` containing:
     - `metadata.json`: Hardware telemetry, duration, total frames, face IDs, eligibility counts.
     - `face_motion.npz`: Compressed columnar arrays (timestamps, face_ids, raw & clean landmarks, 52 blendshapes, head pose, quality scores, roles).
     - `audio_features.json`: Timestamped VAD speech activity and RMS energy.

---

## 🛠️ System Requirements & Environment

- **OS**: Windows 10/11
- **Python**: 3.10+
- **Hardware Profile**: Tested on NVIDIA GeForce GTX 1650 Mobile (4GB VRAM), 8GB RAM
- **Dependencies**: `mediapipe`, `opencv-python`, `PySide6`, `numpy`, `av` (PyAV), `mss`, `psutil`, `sounddevice`, `pyyaml`

---

## 🖥️ Usage Guide

### 1. Launching GUI Studio
```bash
python main.py
```

#### Features:
- **Mode A: Watcher (Passive Background Observer)**:
  - Arm background observer on a screen region or window.
  - Automatically triggers recording with a 1.5–2.0 second in-memory pre-roll buffer when a valid face is detected.
- **Mode B: Manual Recording Session**:
  - Direct **● START RECORDING**, **PAUSE**, and **■ STOP SESSION** controls.
  - Select any screen region interactively via the neon ROI overlay or attach to an open application window.
  - Record system audio or microphone with live VAD and speaker attribution.
- **Mode C: Uploaded Video Processing**:
  - Select an offline video file (`.mp4`, `.mkv`, `.avi`, `.webm`).
  - PyAV decodes video frames directly into RAM, extracting facial motion and audio PTS without creating intermediate files.
- **Diagnostic Visualizer (Toggleable Debug Mode)**:
  - Live preview with face bounding boxes, 2D landmark mesh, head orientation axes, speaker probability, and quality status.
- **System Telemetry HUD**:
  - Real-time CPU %, RAM (MB), GPU %, VRAM (MB), Capture FPS, Inference FPS, Latency (ms), and frame drop counts.

---

### 2. Dataset Inspector & Viewer
Open any recorded session to scrub the timeline, inspect 3D face mesh wireframes, and view blendshape curves:
```bash
python main.py --cli --inspect sessions/session_20260922_163318_vid_97aa
```

#### Viewer Capabilities:
- **Scrubber**: Frame-by-frame stepping or continuous playback (~30 FPS).
- **Face Canvas**: Superimposed Raw Observed (orange) vs Clean Filtered (neon green) landmarks to visually inspect One Euro jitter reduction.
- **3D Pose**: RGB orientation axes projected from the nose tip.
- **Live Blendshapes**: Animated meters for `jawOpen`, `mouthSmile`, `eyeBlink`, `browInnerUp`, etc.
- **Provenance Badges**: Displays `Role` (Speaker/Listener), `Eligibility Level` (Level 0 to 4), and Laplacian sharpness score.
- **Export**: One-click export to newline-delimited JSON (`.jsonl`) for ML training pipelines.

---

### 3. Command-Line Workflows (CLI Mode)

#### Run a Manual Screen Recording Session
```bash
python main.py --cli --mode manual --duration 15 --roi 100,100,640,480
```

#### Process an Offline Video File
```bash
python main.py --cli --mode video --video path/to/video.mp4
```

#### Export Filtered Animation Data to JSONL
```bash
python main.py --cli --export sessions/session_<ID> --output dataset.jsonl --min-level 3
```
*Levels*:
- `0`: All detected faces
- `1`: Stable tracked faces
- `2`: Geometrically usable (size, blur, pose)
- `3`: Animation quality (low jitter, high quality)
- `4`: Speaker paired (animation quality + active confirmed speaker)

---

## 🧪 Verification & Unit Tests

Run the test suite covering One Euro filtering, multi-face tracking, quality scoring, speaker attribution, and storage round-trip:
```bash
python -m unittest discover -s tests
```
