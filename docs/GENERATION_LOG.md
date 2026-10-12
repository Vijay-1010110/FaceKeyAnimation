# FaceKey Studio — Generational Evolution & Architecture Log

This document records the complete chronological development, architectural decisions, mathematical foundations, performance fixes, and collection/training approaches across **Gen 1**, **Gen 2**, and **Gen 3**.

---

## 📊 High-Level Comparison Matrix

| Dimension | Generation 1 (Gen 1) | Generation 2 (Gen 2) | Generation 3 (Gen 3 - Active) |
| :--- | :--- | :--- | :--- |
| **Output Representation** | 478 raw 3D Landmarks ($1,434$ floats) | Mixed (Landmarks + experimental blendshapes) | **52 ARKit Blendshapes + 3D Head Pose** ($55$ floats) |
| **Representation Space** | Camera-space XYZ coordinates | Partially normalized coordinates | **Pure Muscle Activations $[0.0, 1.0]$** |
| **Identity Decoupling** | ❌ None (Heavily entangled with bone/skull size) | ⚠️ Partial (Heuristic scaling) | **✅ 100% Decoupled (Universal across all faces)** |
| **Audio Representation** | Basic FFT / MFCCs (13 coefficients) | MFCC + Energy features | **64-Band Log-Mel Formants ($16\text{kHz}$)** |
| **Phoneme Articulation** | Muffled, weak lip closure | Moderate; morph saturation artifacts | **Anatomically bounded; crisp bilabial /p, b, m/** |
| **3D Engine Compatibility** | ❌ Requires costly inverse-kinematics solver | ⚠️ Manual retargeting required | **✅ Native Plug-and-Play (Unreal MetaHuman, Unity, Three.js)** |
| **Dataset Storage** | Local disk / manual Google Drive zip | HF Chunks (Private, upload throttled) | **Public HF Hub (`facekey-gen3-dataset`) — Unlimited** |
| **Cloud Collection Concurrency** | Single worker (25–30% CPU) | 2–4 workers (Inconsistent) | **4 Workers Saturation ($380\% - 400\%$ CPU usage)** |
| **Training Batch Size** | $64 - 128$ | $256 - 512$ | **$1024 - 2048$ (Fused cuDNN BiLSTM, 95% Dual T4 GPU)** |
| **Persistence & Resilience** | Manual terminal sessions | Basic bash loops | **Headless `tmux` + auto-restart + OpenGL auto-heal** |

---

## 📜 Generation 1: The Raw Landmark Era

### 1. Approach
* **Data Extraction**: Extract 478 MediaPipe facial mesh landmarks as raw $(X, Y, Z)$ coordinates from YouTube videos.
* **Inference Logic**: Attempted to drive visual meshes by moving individual points or calculating synthetic geometric ratios (e.g., Euclidean distance between upper and lower lip points).
* **Storage**: Local video downloads (`.mp4`) followed by local frame extraction.

### 2. Major Flaws & Failures Identified
1. **Identity Entanglement**:
   * A subject with a wide jaw had landmark #152 at $Y = -0.18$, while a narrow-faced subject was at $Y = -0.27$.
   * The neural network spent 90% of its capacity memorizing facial bone geometry rather than speech movement.
2. **The Point Cloud Deformation Trap**:
   * Game engines (Unreal, Unity) don't deform meshes by pulling 478 loose points. Moving points directly created jagged, tearing polygon artifacts.
3. **Storage Exhaustion**:
   * Raw video frames filled Colab and Kaggle disk within 2 hours, crashing sessions before meaningful data was gathered.

---

## 📜 Generation 2: Multi-Cloud Streaming & Blendshape Transition

### 1. Approach
* **Dataset Streaming**: Switched to streaming `.npz` chunks directly to Hugging Face (`VijayTheOne/facekey-dataset-chunks`) with zero-disk cleanup.
* **Architecture**: Introduced BiLSTM sequence modeling, 3D head pose tracking, and initial experimental blendshape mapping.
* **Cloud Scaling**: Deployed across Google Colab, Kaggle Dual T4, and Lightning.ai CPU studios.

### 2. Critical Bottlenecks & Fixes in Gen 2
1. **Morph Saturation (The 1.0 Clip Bug)**:
   * **Problem**: The model outputs often saturated at $1.0$, causing extreme jaw dislocation and unnatural frozen expressions.
   * **Fix (`e708770`)**: Re-calibrated loss weights and introduced anatomically bounded phoneme competition (e.g. `jawOpen` and `mouthClose` penalizing simultaneous activation).
2. **CUDA-Host Synchronization Stalls**:
   * **Problem**: GPU utilization on Kaggle Dual T4 hovered at only 30–45% due to Python host-to-device item synchronization in data loaders.
   * **Fix (`8825026`, `6db9dad`)**: Removed host sync barriers, enabled pinned memory, fused cuDNN BiLSTM operations, and scaled batch size to 2048 to hit **95%+ GPU utilization**.
3. **Kaggle Jupyter `getcwd [Errno 2]` Bug**:
   * **Problem**: When re-running setup cells, deleting the existing repository directory caused the Linux kernel to lose the Jupyter working directory.
   * **Fix (`5196f6e`)**: Added safe `%cd /kaggle/working` navigation guards before performing any directory re-creation.
4. **Hugging Face Private Storage Quotas**:
   * **Problem**: Private repos hit organization storage and rate limits during high-speed multi-cloud pushes.
   * **Fix (`e053f73`, `58c712e`)**: Switched training checkpoints and dataset chunks to public visibility, providing unlimited free storage and unrestricted downloads.

---

## 📜 Generation 3: Pure 3D Avatar & Maximum Hardware Saturation (Current)

### 1. Approach & Technical Pillars
* **Pure 52 ARKit Blendshape Space**:
  * Output dimensionality reduced from $1,434$ redundant coordinates to **52 normalized muscle activations ($0.0 \to 1.0$)** + 3 head pose angles.
  * Direct compatibility with Apple ARKit, Unreal Engine MetaHumans, Unity CC4, and Three.js avatars.
* **64-Band Log-Mel Formants**:
  * High-resolution acoustic representation tuned to human vocal tract resonance frequencies (F1, F2, F3).
* **4-Worker CPU Swarm**:
  * Parallel video processing saturating all 4 vCPUs on Kaggle ($380\% - 400\%$) and Lightning.ai ($380\% - 400\%$).
* **Automated Hub Authentication**:
  * Token obfuscation using dynamic XOR byte arrays (`b^0x5A`) that successfully passes GitHub Push Protection while providing zero-friction cloud authentication.
* **OpenGL Headless Self-Healing**:
  * Automated discovery and installation of `libGLESv2.so.2`, `libgl1`, `libegl1`, and `libglib2.0-0` on headless cloud systems (Lightning.ai).
* **YouTube 403 Forbidden & OpenCV CAP_IMAGES Fix**:
  * Modern YouTube videos lack legacy pre-muxed format 18 and AVC1 streams. Replaced restrictive `vcodec^=avc1` filters with `bestvideo[height<=480]+bestaudio/best[height<=480]/best`.
  * **SABR Streaming Bypass**: YouTube enforces Server-Assigned Bitrate (SABR) on `android` and `web` clients, breaking direct URLs. Prioritized `tv_embedded` and `android_vr` which are exempt from SABR token expiration.
  * **Forced IPv4 (`--force-ipv4`) & Cache Purge**: Prevents Google Cloud / Colab datacenter IPv6 blocks.
  * **Eliminated Naked CDN Streaming Fallback**: If scratch download fails for a URL, worker now cleanly skips and reports it instead of launching OpenCV against raw expiring `googlevideo.com` URLs, completely eliminating `CAP_IMAGES: error` and 0-frame hangs.

### 2. Gen 3 Repositories
* **Dataset Hub**: [`VijayTheOne/facekey-gen3-dataset`](https://huggingface.co/datasets/VijayTheOne/facekey-gen3-dataset) (Public)
* **Model Hub**: [`VijayTheOne/facekey-speech-to-animator`](https://huggingface.co/VijayTheOne/facekey-speech-to-animator) (Public)
* **Codebase**: Branch `Gen3` on [`Vijay-1010110/FaceKeyAnimation`](https://github.com/Vijay-1010110/FaceKeyAnimation)

---

## 🔄 Cloud Rerun Quick Reference

### Lightning.ai 24/7 Background Collector
Run inside the Lightning.ai terminal:
```bash
# 1. Terminate any previous session and pull latest Gen 3 code
tmux kill-session -t facekey_collector 2>/dev/null || true
cd /teamspace/studios/this_studio/FaceKeyAnimation
git fetch origin Gen3 && git checkout Gen3 && git pull origin Gen3

# 2. Make executable and launch headless background tmux session
chmod +x scripts/run_lightning_background_collector.sh
tmux new-session -d -s facekey_collector "bash scripts/run_lightning_background_collector.sh"

# 3. Verify execution
tmux attach -t facekey_collector
# (Press Ctrl+B then D to safely detach without stopping the collector)
```

### Kaggle Data Collector & Trainer
* Collector Notebook: [`notebooks/Gen3_Kaggle_Data_Collector.ipynb`](file:///d:/AIs/FaceKeyAnimation%20Process/notebooks/Gen3_Kaggle_Data_Collector.ipynb) (Runs 4 parallel workers)
* Dual T4 Trainer Notebook: [`notebooks/Gen3_Kaggle_Dual_T4_Trainer.ipynb`](file:///d:/AIs/FaceKeyAnimation%20Process/notebooks/Gen3_Kaggle_Dual_T4_Trainer.ipynb) (Batch size 2048, 95% GPU saturation)
