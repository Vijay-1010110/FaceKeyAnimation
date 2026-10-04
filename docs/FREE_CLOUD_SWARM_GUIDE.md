# ⚡ FaceKey Swarm - Complete Guide to 100% Free Cloud Compute

> Accelerate your dataset collection to **100+ Hours** in days by utilizing all available free cloud compute providers simultaneously. All data automatically streams to your private Hugging Face dataset (`VijayTheOne/facekey-dataset-chunks`) and synchronizes into your 5 TB Google Drive.

---

## 📊 Summary of Free Cloud Providers

| Provider | Free Quota | Hardware | Setup Time | Best For |
| :--- | :---: | :---: | :---: | :---: |
| **GitHub Actions** | 2,000 mins/mo | 3x Parallel Ubuntu VMs | **10 seconds** (1 Click) | Unattended batch bursts |
| **Kaggle (CPU Mode)** | **Unlimited** | 4-Core CPU, 30 GB RAM | **30 seconds** | 12-hour continuous background runs |
| **GitHub Codespaces** | 60 hours/mo | 2 vCPU, 8 GB RAM | **30 seconds** | High-speed 1000 Mbps Azure VM |
| **Google Cloud Shell** | 50 hours/wk | 2-Core VM, 8 GB RAM | **1 minute** | Direct Google IP (zero bot challenges) |
| **Hugging Face Spaces** | **24/7 Free** | 2 vCPU, 16 GB RAM | **2 minutes** | Continuous 24/7 web dashboard worker |
| **Google Colab** | Free Daily T4/CPU | 2 vCPU / T4 GPU | Active | Active development & Google Drive sync |
| **Lightning.ai** | 22 Free Credits/mo | 4-Core CPU Studio | Active | Fast parallel 3-worker processing |

---

## 🚀 1. GitHub Actions Multi-Runner Swarm (1-Click Run)

You can launch 3 parallel runners simultaneously in GitHub's cloud without installing anything on your PC.

### How to Start:
1. Go to your repository on GitHub: [Vijay-1010110/FaceKeyAnimation](https://github.com/Vijay-1010110/FaceKeyAnimation)
2. Click the **Actions** tab at the top.
3. In the left sidebar, click **FaceKey Swarm Collector**.
4. Click the **Run workflow** dropdown button on the right.
5. (Optional) Select number of parallel runners (default: `3`) and paste your `HF_TOKEN`.
6. Click the green **Run workflow** button!

Three independent Ubuntu cloud instances will immediately boot, process videos from `sessions/youtube_urls_gha.txt`, and auto-upload `.tar.gz` chunks to your Hugging Face repository.

---

## 🦅 2. Kaggle CPU Background Swarm (Unlimited Free Hours)

Kaggle limits GPUs to 30 hrs/week, but **CPU mode has NO weekly quota limit**! Our collector runs at 150+ FPS on CPU.

### How to Start:
1. Open your Kaggle Notebook (e.g. `FaceKey-Collector` or duplicate it).
2. In the right-hand panel, set **Accelerator: None (CPU)**.
3. In Cell 1, paste or execute:
   ```python
   !python scripts/run_kaggle_cpu.py
   ```
4. Click **Save Version** (top right) -> Select **Save & Run All (Commit)** -> Click **Save**.
5. Close your browser! Kaggle will run in the cloud for up to 12 hours, collect data on all 4 CPU cores, stream chunks to Hugging Face, and shutdown cleanly.

---

## 💻 3. GitHub Codespaces (60 Free Hours/Month)

Every GitHub account gets 60 free hours every month.

### How to Start:
1. Go to your repository: [Vijay-1010110/FaceKeyAnimation](https://github.com/Vijay-1010110/FaceKeyAnimation)
2. Click the green **`<> Code`** button.
3. Switch to the **Codespaces** tab -> Click **Create codespace on main**.
4. A full VS Code browser terminal will open in ~15 seconds.
5. In the terminal, run:
   ```bash
   bash scripts/cloud_quickstart.sh
   ```
6. Paste your Hugging Face token when prompted (starts with `hf_`). It will run 2 parallel workers and stream chunks to your dataset!

---

## ☁️ 4. Google Cloud Shell (50 Free Hours/Week)

Google Cloud Shell provides a free interactive Linux environment with 8 GB RAM and Google Cloud IPs.

### How to Start:
1. Open [https://shell.cloud.google.com](https://shell.cloud.google.com) in your browser.
2. Run:
   ```bash
   git clone https://github.com/Vijay-1010110/FaceKeyAnimation.git
   cd FaceKeyAnimation
   bash scripts/cloud_quickstart.sh
   ```
3. Paste your Hugging Face token when prompted.

---

## 🤗 5. Hugging Face Spaces (24/7 Cloud Worker)

Hugging Face provides a free 2 vCPU, 16 GB RAM instance that runs indefinitely.

### How to Start:
1. Go to [https://huggingface.co/new-space](https://huggingface.co/new-space).
2. Space Name: `facekey-collector-swarm` (or any name).
3. Space SDK: **Gradio**.
4. Visibility: **Public** or **Private**.
5. Once created, go to **Settings** -> **Variables and secrets**:
   - Add Secret: `HF_TOKEN` = your write token (`hf_...`).
6. Upload the files from `spaces/hf_collector_space/` (`app.py`, `requirements.txt`, `README.md`).
7. Open the Space app URL: you will see a live dashboard with a **"🚀 Start Swarm Worker"** button that runs 24/7!

---

## 🔄 How to Pull All Cloud Data Back to Google Drive

Whenever you want all chunks from all cloud services merged into your 5 TB Google Drive:
In **Google Colab**, simply run:

```python
%cd /content/FaceKeyAnimation
!git pull origin main
!python scripts/sync_hf_to_drive.py --drive-dir "/content/drive/MyDrive/FaceKeyDataset"
```

This delta synchronizer checks all chunks on Hugging Face, skips any that are already in Drive, and downloads only new chunks at ~100 MB/s datacenter speed. Zero duplicate files, 100% verified.

---

## 📋 Queue Allocation & Isolation

### 🛡️ Core Proven Services (The 351 Original Videos - 100% Preserved)
| Service | Dedicated Queue File | Video Count |
| :--- | :--- | :---: |
| **Google Colab** | `sessions/youtube_urls_colab.txt` | **117 Videos** |
| **Kaggle (GPU & CPU)** | `sessions/youtube_urls_kaggle.txt` | **117 Videos** |
| **Lightning.ai Studios** | `sessions/youtube_urls_lightning.txt` | **117 Videos** |
| **Core Total** | `sessions/youtubeURLtoProcess.txt` | **351 Videos** |

### 🧪 Experimental Free Services (118 Brand-New Videos - 0% Overlap)
| Service | Dedicated Queue File | Video Count |
| :--- | :--- | :---: |
| **GitHub Actions (3 Runners)** | `sessions/youtube_urls_gha.txt` | **30 New Videos** |
| **GitHub Codespaces** | `sessions/youtube_urls_codespaces.txt` | **30 New Videos** |
| **Google Cloud Shell** | `sessions/youtube_urls_cloudshell.txt` | **29 New Videos** |
| **Hugging Face Spaces** | `sessions/youtube_urls_hfspace.txt` | **29 New Videos** |
| **Experimental Total** | `sessions/youtube_urls_experimental_master.txt` | **118 New Videos** |

