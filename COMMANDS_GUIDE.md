# FaceKey MoCap Studio — Command & Execution Guide
**Simple Command Reference for Capturing, Background Recording, Phone Streaming, and AI Training**

---

## 🚀 Quick-Start: One-Click Batch Runners

If you don't want to type commands, double-click any of these `.bat` files in the folder:

| File | What It Does | When to Use |
| :--- | :--- | :--- |
| **`RUN_LIVE_YOUTUBE.bat`** | Launches the live interactive GUI window at the top-right of your screen. | When actively watching YouTube and you want the visual overlay, teeth meters, and buttons on screen. |
| **`START_BACKGROUND_RECORDING.bat`** | Interactive menu to launch silent background recording (Window, Screen, or Phone). | When you want to work, browse other tabs, or write code without any window popping up. |
| **`STOP_STUDIO_PEACEFULLY.bat`** | Sends a clean save signal (`stop.flag`) that safely flushes and saves all frames. | Anytime you want to stop recording safely without losing data. |

---

## 💻 Command Line Interface (CLI) Guide

All scripts run via Python (`D:\python_3.11\python.exe` or `python` if in PATH):

### 1. Interactive YouTube Live Studio (Default Screen Overlay)
Captures your desktop screen and displays an interactive, razor-sharp 1280x720 HD telemetry window (Teeth, Muscles, Head Pose, Eye Gaze):

```powershell
# Default full screen capture (1280x720 HD preview with razor-sharp text)
python test_face_speaker_tool.py --mode screen --roi 0,0,1920,1080

# Crop to a specific screen rectangle: x, y, width, height
python test_face_speaker_tool.py --mode screen --roi 100,100,1280,720
```
> **Minimization Support**: You can freely click the minimize `[_]` button on the top-right of the window to tuck it into your Windows taskbar. Recording and tracking will continue running smoothly at full speed in the background!
> **Crop Tool**: Inside the preview window, you can click **`[✂ CROP ROI]`** (or press `R`) to drag a box around your YouTube player.

---

### 2. Background Tab & Window Recording (With Dynamic Focus Switching)
Captures directly from a specific browser tab or application window (e.g., YouTube, Brave, Chrome) **even when occluded or behind other windows**:

```powershell
# Launch with interactive 1280x720 HD monitor window targeting YouTube/Brave
python test_face_speaker_tool.py --mode window --window "YouTube"

# Or launch targeting Brave
python test_face_speaker_tool.py --mode window --window "Brave"
```

> 🎯 **How to Switch to Your Desired Tab (Zero Dropdowns / Zero Menus)**:
> * **1-Click Button**: Click **`[🎯 SWITCH TAB]`** inside the monitor window to instantly cycle directly to the next open browser window/tab!
> * **Direct Hotkey**: Press **`Tab`** or **`F`** on your keyboard inside the window to flip between open windows immediately.
> * **Global Hotkey (`F9`)**: While browsing or watching YouTube anywhere on Windows, simply tap **`F9`** on your keyboard — FaceKey will instantly lock onto that active window without you having to touch the monitor window!
> 
> 💡 **Crucial Browser Tip (Brave / Chrome / Edge)**:
> In all Chromium browsers, if you have multiple tabs inside the *exact same* window, Windows only renders the currently active tab.  
> **To let FaceKey record YouTube in the background while you work or chat on another tab:**
> Simply **drag the YouTube tab out into its own separate window**! Both windows will remain active in the OS, and FaceKey will capture YouTube seamlessly behind your work!

---

### 3. Silent Headless Screen Recording
Records your desktop without displaying any preview window, leaving your screen completely clean:

```powershell
python test_face_speaker_tool.py --mode screen --roi 0,0,1920,1080 --headless
```

---

### 4. Direct Silent YouTube & Video Ingestion (100% Muted, 0 dB Sound)
If you have a YouTube video or file to train on, you **do not even need to open a browser or hear any sound**!
Python decodes the audio and video directly in memory with **zero sound sent to your speakers**:

```powershell
# Ingest YouTube video silently in RAM (0.0 dB audio to speakers, zero screen clutter)
python scripts/process_silent_video.py --url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"

# Process a local video file in 100% silence
python scripts/process_silent_video.py --file "C:\path\to\my_video.mp4"
```
*Benefits*: 100% silence for you, zero microphone background noise, faster than real-time speed.

---

### 5. Live Browser Tab Internal Muting (Windows Volume Mixer)
If you are playing YouTube in Brave or Chrome and want to **mute it for your ears** while FaceKey captures the audio internally:

1. **Check available audio devices**:
   ```powershell
   python test_face_speaker_tool.py --list-audio-devices
   ```
2. **In Windows 11**:
   - Right-click the **Speaker icon** on your Windows taskbar $\rightarrow$ select **Volume mixer** (or press `Win + I` $\rightarrow$ System $\rightarrow$ Sound $\rightarrow$ Volume mixer).
   - Scroll down to **Apps** $\rightarrow$ click on **Brave** (or Chrome).
   - Change the **Output device** dropdown from *"Default"* to **"AI Noise-cancelling Output (ASUS Utility)"** or a virtual cable.
3. **Launch FaceKey listening to that internal device**:
   ```powershell
   # Pass the device name or index (e.g. "ASUS" or index 2/8)
   python test_face_speaker_tool.py --mode window --window "Brave" --audio-device "ASUS" --headless
   ```
*Result*: Your physical speakers/headphones stay **100% silent**, while FaceKey records crystal-clear internal audio!

---

### 6. Vivo Y21 / Phone Camera Stream Recording
Streams video from your Vivo Y21 phone over Wi-Fi or USB (using DroidCam or IP Webcam). The phone uses hardware H.264 (<4% CPU) so it **remains ice-cold while you read**:

```powershell
# Connect to phone camera stream
python test_face_speaker_tool.py --mode stream --stream-url "http://192.168.1.15:8080/video"

# Phone stream in silent background mode (no PC preview window)
python test_face_speaker_tool.py --mode stream --stream-url "http://192.168.1.15:8080/video" --headless
```

---

### 7. Offline Simulation Test Mode
Runs the built-in 4-phase benchmark video (`tests/data/podcast_multi_face_test.mp4`) with synthesized audio to verify algorithm performance:

```powershell
python test_face_speaker_tool.py --mode sim
```

---

## 🛑 How to Stop Recording Peacefully

You can safely stop and save recording using **any** of these 4 methods:

| Method | How to Do It | Result |
| :--- | :--- | :--- |
| **Method 1: Window Close `[X]`** | Click standard Windows **`[X]`** title-bar button or on-screen **`[✕ CLOSE]`**. | Immediately flushes in-memory frames to `sessions/` and closes. |
| **Method 2: One-Click Script** | Double-click **`STOP_STUDIO_PEACEFULLY.bat`** (or run `python stop_studio.py`). | Drops `stop.flag`, loop immediately saves all frames to `sessions/` and exits. |
| **Method 3: Global Hotkey** | Press **`Shift + Esc`** on your keyboard from **any window**. | Instantly triggers peaceful save and shutdown from anywhere in Windows. |
| **Method 4: Terminal Command** | Type **`stop`** or **`q`** in the command prompt and hit `Enter`. | Cleanly finishes writing archive and terminates. |
| **Method 5: Keyboard Interrupt** | Press **`Ctrl + C`** in the command prompt. | Intercepted by signal handler to flush files before exiting. |

---

## 🧠 Dataset Normalization & AI Training Preparation

Once you have recorded one or more sessions, run the normalization pipeline to eliminate head tilt, camera distance, and facial asymmetry:

### 1. Run Preprocessing CLI
```powershell
# Scans all sessions in sessions/, applies Umeyama alignment, regional scaling, and symmetry
python scripts/preprocess_normalized_training_data.py
```

**Output Files**:
- `sessions/normalized_training_dataset.npz` (Master batch-ready training tensor)
- `sessions/dataset_statistics.json` (Invertible mean, std, min, max for model inference)

---

### 2. Loading Normalized Data in Python (PyTorch / TensorFlow)

```python
from src.storage.dataset_reader import NormalizedTrainingDataset

# Load the consolidated dataset
dataset = NormalizedTrainingDataset("sessions/normalized_training_dataset.npz")
print(f"Total training frames: {len(dataset)}")

# Get 85% train split or 15% validation split
train_data = dataset.get_split("train")
val_data   = dataset.get_split("val")

# Index individual samples
sample = dataset[0]
print(sample.keys())
# Available keys:
# - canonical_landmarks   : (478, 3) 3D mesh centered at origin
# - expression_deltas     : (478, 3) pure speech muscle displacements
# - symmetric_landmarks   : (478, 3) bilateral symmetrized speech target
# - dental_features       : [upper_teeth, lower_teeth, inter_gap, state_id]
# - blendshapes           : (52,) ARKit parameter activations
# - pose_deltas           : [pitch, yaw, roll] relative to rest
# - audio_energy          : float RMS energy
```

---

## ⌨️ Summary of Keyboard Controls (When Preview is Visible)

| Key | Action |
| :---: | :--- |
| **`Spacebar`** | Toggle Pause / Play (drops CPU usage to < 1% during pause) |
| **`S`** | Flush & Save in-memory frames to permanent `sessions/` folder |
| **`R`** | Open interactive crop tool to drag rectangle around YouTube player |
| **`1`** | Switch live to Simulation mode |
| **`2`** | Switch live to Screen Capture mode |
| **`Q` / `Esc`** | Clean peaceful shutdown (auto-saves any pending frames) |
