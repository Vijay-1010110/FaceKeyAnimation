# 📖 FaceKey Studio — Complete Cloud Setup Guide (Google Colab & Kaggle)

> **Zero Local Disk Bloat (< 5 GB) & Free Multi-GPU Acceleration**: This complete step-by-step tutorial walks you through setting up and running **Google Colab** and **Kaggle** concurrently, connecting both to your **5 TB Google Drive**, and collecting/training your dataset without redundant downloads or lost progress.

---

## 📑 Table of Contents
1. [Google Drive Architecture (5 TB Plan)](#1-google-drive-architecture-5-tb-plan)
2. [Part A: Complete Google Colab Setup (Step-by-Step)](#part-a-complete-google-colab-setup-step-by-step)
   - [A1. Open Notebook from GitHub](#a1-open-notebook-from-github)
   - [A2. Enable Free Hardware Acceleration (GPU / High-RAM)](#a2-enable-free-hardware-acceleration-gpu--high-ram)
   - [A3. Mount & Authorize Google Drive](#a3-mount--authorize-google-drive)
   - [A4. Run Data Collection on Colab](#a4-run-data-collection-on-colab)
   - [A5. Run Model Training on Colab](#a5-run-model-training-on-colab)
3. [Part B: Complete Kaggle Setup (Step-by-Step)](#part-b-complete-kaggle-setup-step-by-step)
   - [B1. Verify Phone & Enable Internet on Kaggle](#b1-verify-phone--enable-internet-on-kaggle)
   - [B2. Create a New Kaggle Notebook](#b2-create-a-new-kaggle-notebook)
   - [B3. Connect Google Drive to Kaggle](#b3-connect-google-drive-to-kaggle)
   - [B4. Run Parallel Ingestion alongside Colab](#b4-run-parallel-ingestion-alongside-colab)
4. [Part C: How to Add & Manage YouTube URLs](#part-c-how-to-add--manage-youtube-urls)
5. [Part D: Handling Disconnects, Timeouts & Auto-Resumes](#part-d-handling-disconnects-timeouts--auto-resumes)
6. [Part E: FAQ & Troubleshooting](#part-e-faq--troubleshooting)

---

## 1. Google Drive Architecture (5 TB Plan)

When you run Colab or Kaggle, the system automatically creates this clean directory structure inside your **5 TB Google Drive**:

```text
Google Drive/
└── MyDrive/
    └── FaceKeyDataset/
        ├── chunks/                          <- Stores compressed .tar.gz chunks (~250-500 MB each)
        │   ├── dataset_chunk_colab-worker-1_20261003_104850_0001.tar.gz
        │   └── dataset_chunk_kaggle-worker-1_20261003_105020_0002.tar.gz
        ├── checkpoints/                     <- Stores PyTorch model weights (checkpoint_latest.pt)
        ├── locks/                           <- Live atomic locks (<video_id>.lock.json) with heartbeats
        ├── data/                            <- Global shared completion registry
        │   └── shared_completed_sources.json
        ├── dataset_manifest.json            <- Master index of all chunks, hours, and frame counts
        ├── youtube_urls_colab.txt           <- Active queue for Colab worker
        ├── youtube_urls_kaggle.txt          <- Active queue for Kaggle worker
        └── youtubeURLtoProcess.txt          <- Master shared queue (all 351 curated videos)
```

---

## Part A: Complete Google Colab Setup (Step-by-Step)

### A1. Open Notebook from GitHub
1. Open your web browser and go to: **[https://colab.research.google.com/](https://colab.research.google.com/)**
2. In the popup window that appears, click the **GitHub** tab (orange GitHub icon).
   - If no popup appears, click **File $\rightarrow$ Open notebook** in the top menu, then select **GitHub**.
3. In the search box, paste your repository URL:
   ```text
   https://github.com/Vijay-1010110/FaceKeyAnimation.git
   ```
   and press **Enter** (or click the magnifying glass).
4. You will see the list of notebooks in the repository:
   - For collecting video data: Click **`notebooks/1_Colab_Data_Collector.ipynb`**.
   - For training the AI model: Click **`notebooks/2_Colab_Model_Trainer.ipynb`**.

---

### A2. Enable Free Hardware Acceleration (GPU / High-RAM)
- **For Data Collection**: CPU runtime is sufficient, or you can use the free T4 GPU.
- **For Model Training**: You **MUST** select the GPU:
  1. Click **Runtime** in the top menu $\rightarrow$ **Change runtime type**.
  2. Under **Hardware accelerator**, select **T4 GPU**.
  3. Click **Save**.

---

### A3. Mount & Authorize Google Drive
Run **Cell 1** in the notebook:
```python
from google.colab import drive
import os

drive.mount('/content/drive')
DRIVE_DIR = '/content/drive/MyDrive/FaceKeyDataset'
os.makedirs(f'{DRIVE_DIR}/chunks', exist_ok=True)
os.makedirs(f'{DRIVE_DIR}/checkpoints', exist_ok=True)
os.makedirs(f'{DRIVE_DIR}/locks', exist_ok=True)
os.makedirs(f'{DRIVE_DIR}/data', exist_ok=True)
print(f'[+] Google Drive mounted successfully at: {DRIVE_DIR}')
```
- A Google popup will appear asking: *"Permit this notebook to access your Google Drive files?"*
- Click **Connect to Google Drive**, select your Google Account (the one with the **5 TB Plan**), and click **Allow**.
- You will see: `[+] Google Drive mounted successfully at: /content/drive/MyDrive/FaceKeyDataset`.

---

### A4. Run Data Collection on Colab
1. Run **Cell 2**: Clones the latest code from GitHub and installs dependencies (`yt-dlp`, `mediapipe`, `opencv-python-headless`).
2. Run **Cell 3**: Previews your `youtube_urls_colab.txt` queue.
3. Run **Cell 4**: Starts the data collector:
   ```bash
   !python scripts/cloud_data_collector.py \
       --drive-dir "/content/drive/MyDrive/FaceKeyDataset" \
       --worker-id "colab-worker-1" \
       --urls-file "/content/drive/MyDrive/FaceKeyDataset/youtube_urls_colab.txt" \
       --chunk-size 5 \
       --quality 480p
   ```
- The script automatically claims URLs, writes atomic locks with live heartbeats, records facial motion in silent 480p, and packages sessions into `.tar.gz` chunks on Google Drive.
- Local scratch storage is wiped after each chunk so Colab disk stays $< 5\text{ GB}$.

---

### A5. Run Model Training on Colab
1. Open **`notebooks/2_Colab_Model_Trainer.ipynb`**.
2. Run **Cell 1**: Mounts Google Drive.
3. Run **Cell 2**: Fast-unpacks `.tar.gz` chunks from Drive into Colab's fast local NVMe SSD (`/content/training_data/`) in under 15 seconds.
4. Run **Cell 3**: Launches high-speed training with PyTorch Automatic Mixed Precision (FP16 Tensor Cores):
   ```bash
   !python scripts/train_speech_to_animation.py \
       --data-dir "/content/training_data" \
       --checkpoint-dir "/content/drive/MyDrive/FaceKeyDataset/checkpoints" \
       --batch-size 64 \
       --epochs 50 \
       --fp16 \
       --num-workers 4
   ```
- Weights (`checkpoint_latest.pt` and `checkpoint_best.pt`) save directly to Drive after every epoch.

---

## Part B: Complete Kaggle Setup (Step-by-Step)

Kaggle provides **30 hours of free GPU per week** (NVIDIA Tesla P100 16GB or 2x NVIDIA T4 16GB).

### B1. Verify Phone & Enable Internet on Kaggle
1. Sign in to **[https://www.kaggle.com/](https://www.kaggle.com/)**.
2. Go to your **Account Settings** $\rightarrow$ **Phone verification**. Verify your phone number (this is a one-time step that unlocks free GPUs and Internet access on Kaggle).

---

### B2. Create a New Kaggle Notebook
1. In the Kaggle top navigation, click **Create** $\rightarrow$ **New Notebook**.
2. In the right-hand **Notebook settings** sidebar:
   - **Accelerator**: Select **GPU P100** (or **GPU T4 x2**).
   - **Internet**: Toggle to **Internet On** (Required to download code and YouTube streams!).
   - **Language**: Python.

---

### B3. Connect Google Drive to Kaggle
There are two ways to connect your Google Drive to Kaggle:

#### Method 1: Kaggle's Built-in Google Drive Integration (Recommended)
1. In the right-hand panel of your Kaggle notebook, click **+ Add Input** (or **+ Add Data**).
2. Look for the **Google Drive** icon in the popup.
3. Authenticate with your Google account. Kaggle mounts your Google Drive under `/kaggle/input/google-drive` or `/kaggle/working/google_drive`.

#### Method 2: Direct Working Directory (Instant, Zero Setup)
If you don't connect Drive in the sidebar, the script automatically uses `/kaggle/working/FaceKeyDataset`. You can export the generated `.tar.gz` chunks directly from Kaggle outputs or upload them to Drive with a single command:
```python
# Install gdown / pydrive to sync chunks to Google Drive
!pip install -q pydrive2
```

---

### B4. Run Parallel Ingestion alongside Colab
Copy and run the code from [`notebooks/3_Kaggle_Data_Collector_and_Trainer.ipynb`](file:///d:/AIs/FaceKeyAnimation%20Process/notebooks/3_Kaggle_Data_Collector_and_Trainer.ipynb):

```python
# 1. Clone repo
!git clone https://github.com/Vijay-1010110/FaceKeyAnimation.git /kaggle/working/FaceKeyAnimation
%cd /kaggle/working/FaceKeyAnimation

# 2. Install dependencies
!pip install -q yt-dlp mediapipe opencv-python-headless numpy sounddevice

# 3. Launch parallel collector
!python scripts/cloud_data_collector.py \
    --drive-dir "/kaggle/working/FaceKeyDataset" \
    --worker-id "kaggle-worker-1" \
    --urls-file "sessions/youtube_urls_kaggle.txt" \
    --chunk-size 5 \
    --quality 480p
```

- Kaggle runs as `kaggle-worker-1` on the second half of the dataset (MIT OCW, Stanford Online, Closer to Truth, Lex Fridman, Numberphile).
- Colab runs as `colab-worker-1` on the first half (Big Think, Huberman Lab, TED, Google Talks, CS50).
- **Both run simultaneously, doubling your collection speed with zero duplicate work!**

---

## Part C: How to Add & Manage YouTube URLs

### Adding New Links Anytime
You do not need to restart your notebooks to add new YouTube videos!
1. Open Google Drive in your web browser: **`MyDrive/FaceKeyDataset/`**.
2. Double-click **`youtube_urls_colab.txt`** (for Colab) or **`youtube_urls_kaggle.txt`** (for Kaggle).
3. Paste any YouTube URLs at the bottom of the file (one URL per line):
   ```text
   https://www.youtube.com/watch?v=NEW_VIDEO_ID_1
   https://www.youtube.com/watch?v=NEW_VIDEO_ID_2
   ```
4. Save the file.
5. The running cloud workers re-read the file every few seconds and automatically start processing your newly added links!

---

## Part D: Handling Disconnects, Timeouts & Auto-Resumes

Free cloud VMs disconnect after long runs:
- **Google Colab**: 6–12 hours continuous execution.
- **Kaggle**: 12 hours per session.

### How FaceKey Protects Your Data:
1. **Save Points on Drive**: Every single completed video is written to `dataset_manifest.json` and `.tar.gz` chunks immediately on Drive.
2. **Resuming after Timeout**:
   - Simply open the notebook and click **Run All**.
   - The script inspects `shared_completed_sources.json` and skips all previously finished videos.
   - It continues from the next pending link in the queue.
3. **Stale Lock Recovery**:
   - If Colab or Kaggle disconnects in the middle of a video, the lock's heartbeat stops.
   - When you reconnect (or when the other worker finishes its current video), the system detects that the lock is $> 20$ minutes old, auto-reclaims it, and resumes.
4. **Training Resume**:
   - If model training is interrupted, running the training cell automatically detects `checkpoint_latest.pt` on Google Drive and restores model weights, optimizer state, and epoch count.

---

## Part E: FAQ & Troubleshooting

### Q1: Colab says `TransportError / Read timed out` for YouTube?
**Answer**: YouTube occasionally rate-limits specific IP ranges. FaceKey's cloud collector has built-in retry mechanisms (`retries=5`, `socket_timeout=30`). If a video fails, the script cleanly releases its lock and moves to the next video, then retries failed ones later.

### Q2: How can I keep Colab awake while I'm away?
**Answer**: Press `F12` in your browser to open Developer Tools, go to the **Console** tab, paste the following snippet, and press Enter:
```javascript
function keepAlive() {
    console.log("Keeping Colab Active...");
    document.querySelector("colab-connect-button")?.click();
}
setInterval(keepAlive, 60000); // Clicks every 60 seconds
```

### Q3: How do I know how much data has been collected?
**Answer**: In Google Drive, view `MyDrive/FaceKeyDataset/dataset_manifest.json`. It displays:
```json
{
  "total_chunks": 12,
  "total_sessions": 60,
  "total_frames": 432000,
  "total_duration_hours": 4.0,
  "total_size_mb": 1420.5
}
```
Or run `CHECK_STATUS.bat` on your local PC!
