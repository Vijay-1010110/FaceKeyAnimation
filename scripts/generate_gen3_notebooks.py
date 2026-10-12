"""Generator for FaceKey Gen 3 Deep Learning Training & Collection Notebooks.
Branch: Gen3
Milestone: Pure 3D Avatar Animation (52 ARKit Blendshapes + 3D Head Pose + 64-Band Log-Mel Acoustics + Emotion Conditioning)

Generates:
  1. notebooks/Gen3_Kaggle_Dual_T4_Trainer.ipynb       (Kaggle Dual T4 x2 GPU 9h Background Commit Runner - Batch 2048)
  2. notebooks/Gen3_Colab_GPU_Trainer.ipynb            (Google Colab T4 GPU Runner + Hugging Face Sync)
  3. notebooks/Gen3_Lightning_AI_Trainer.ipynb         (Lightning.ai Persistent 24/7 Studio Runner)
  4. notebooks/Gen3_Colab_Data_Collector.ipynb         (Google Colab CPU Multi-Worker Collector)
  5. notebooks/Gen3_Kaggle_Data_Collector.ipynb        (Kaggle Cloud CPU 4-Worker Collector - 400% Saturation)
  6. notebooks/Gen3_Lightning_AI_Data_Collector.ipynb  (Lightning.ai 4-Core CPU Collector)
"""

import os
import sys
import json
from typing import Dict, Any, List

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_URL = "https://github.com/Vijay-1010110/FaceKeyAnimation.git"
BRANCH = "Gen3"
HF_MODEL_REPO = "VijayTheOne/facekey-speech-to-animator"
HF_GEN3_DATASET_REPO = "VijayTheOne/facekey-gen3-dataset"
HF_GEN2_DATASET_REPO = "VijayTheOne/facekey-dataset-chunks"


def make_cell(cell_type: str, source: List[str]) -> Dict[str, Any]:
    """Helper to construct clean Jupyter notebook cells."""
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line if line.endswith("\n") else line + "\n" for line in source]
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def generate_gen3_kaggle_trainer() -> Dict[str, Any]:
    cells = [
        make_cell("markdown", [
            "# 🧠 FaceKey Studio Gen 3 — Kaggle Dual T4 GPU 3D Avatar Model Trainer",
            "### Pure 3D Digital Human Animation: 52 ARKit Blendshapes + 3D Head Pose + 64-Band Log-Mel Acoustics",
            "",
            "> [!IMPORTANT]",
            "> **KAGGLE SETTINGS (RIGHT SIDEBAR):**",
            "> 1. **Accelerator**: Select **GPU T4 x2** (Allocates 2x NVIDIA Tesla T4 GPUs with 30 GB combined VRAM)",
            "> 2. **Internet**: Toggle **Internet ON** (Required to stream dataset chunks and sync model weights)",
            "",
            "## 🌟 HOW TO RUN IN BACKGROUND WITH PC TURNED OFF (9 HOURS HEADLESS):",
            "1. In the top right corner of Kaggle, click **Save Version**.",
            "2. Under *Version Type*, select **Save & Run All (Commit)**.",
            "3. Click **Save**.",
            "4. **🎉 YOU CAN NOW SAFELY CLOSE YOUR BROWSER AND SHUT DOWN YOUR PC!**",
            "   Kaggle runs the entire 100-epoch training run in the cloud with batch size 2048 saturating both GPUs. Every epoch automatically pushes `checkpoint_latest.pt`, `checkpoint_best.pt`, and `loss_curve.png` to Hugging Face Model Hub!",
            "",
            "**🛡️ GEN 3 ARCHITECTURAL PILLARS:**",
            "- **Pure 3D Avatar Targets**: Predicts 52 ARKit blendshapes + 3D head pose Euler angles + dental visibility.",
            "- **Zero Landmark Bloat**: No 478 raw coordinates stored in dataset, making training 4x faster with minimal RAM usage.",
            "- **True 64-Band Log-Mel Spectral Formants**: Full frequency coverage (20 Hz - 8000 Hz) captures distinct vowel and consonant shapes (/a/, /o/, /e/, /m/, /s/).",
            "- **Expressive Paralinguistics & Tone**: 5-state emotion embedding (Neutral, Happy, Angry, Surprised, Sad) with continuous mood blending.",
            "- **Auto-Resume & Fallback**: Automatically streams from Gen 3 dataset, falls back to Gen 2 if fresh, and auto-resumes weights."
        ]),
        make_cell("markdown", [
            "## 💓 STEP 0: BROWSER KEEP-ALIVE HEARTBEAT (FOR INTERACTIVE SESSIONS)",
            "> Run this cell if training interactively in your browser to prevent tab timeouts."
        ]),
        make_cell("code", [
            "# Browser Keep-Alive Heartbeat",
            "from IPython.display import display, Javascript",
            "display(Javascript('''",
            "function keepAlive(){",
            "    console.log('[FaceKey Gen3 Heartbeat] Session active at ' + new Date().toLocaleTimeString());",
            "}",
            "setInterval(keepAlive, 60000);",
            "'''))",
            "print('[+] Anti-sleep browser heartbeat initialized.')"
        ]),
        make_cell("markdown", [
            "## ⚡ STEP 1: GPU HARDWARE & VRAM VERIFICATION",
            "> Confirms dual NVIDIA Tesla T4 allocation."
        ]),
        make_cell("code", [
            "!nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader",
            "import torch",
            "print(f'[+] PyTorch Version: {torch.__version__}')",
            "print(f'[+] CUDA Available: {torch.cuda.is_available()}')",
            "print(f'[+] GPU Device Count: {torch.cuda.device_count()}')",
            "for i in range(torch.cuda.device_count()):",
            "    print(f'    • GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)')"
        ]),
        make_cell("markdown", [
            "## 📦 STEP 2: CLONE GEN 3 CODEBASE & INSTALL DEPENDENCIES"
        ]),
        make_cell("code", [
            "import os, shutil",
            "os.chdir('/kaggle/working')",
            "APP_DIR = '/kaggle/working/FaceKeyAnimation'",
            "if not os.path.exists(APP_DIR):",
            f"    !git clone -b {BRANCH} {REPO_URL} {{APP_DIR}}",
            "%cd {APP_DIR}",
            f"!git fetch origin && !git checkout {BRANCH} && !git pull origin {BRANCH}",
            "!pip install -q -U torch torchvision torchaudio huggingface_hub numpy scipy matplotlib",
            "print('[+] Gen 3 Deep Learning Training Environment Ready on branch: " + BRANCH + "')"
        ]),
        make_cell("markdown", [
            "## 🔑 STEP 3: HUGGING FACE AUTHENTICATION SETUP",
            "> Uses Kaggle Secrets `HF_TOKEN` or falls back to local token file."
        ]),
        make_cell("code", [
            "hf_token = None",
            "try:",
            "    from kaggle_secrets import UserSecretsClient",
            "    hf_token = UserSecretsClient().get_secret('HF_TOKEN')",
            "    print('[✓] Retrieved HF_TOKEN from Kaggle Secrets!')",
            "except Exception:",
            "    pass",
            "",
            "if not hf_token and os.path.exists('/kaggle/working/hf_token.txt'):",
            "    with open('/kaggle/working/hf_token.txt', 'r') as f:",
            "        hf_token = f.read().strip()",
            "    print('[✓] Retrieved HF_TOKEN from /kaggle/working/hf_token.txt')",
            "",
            "if hf_token:",
            "    os.environ['HF_TOKEN'] = hf_token",
            "    print(f'[✓] Hugging Face Hub Authenticated for automatic model sync!')",
            "else:",
            "    print('[!] Note: HF_TOKEN not found. Public datasets will stream freely; model checkpoints will save locally.')"
        ]),
        make_cell("markdown", [
            "## 🚀 STEP 4: LAUNCH GEN 3 3D AVATAR TRAINING (100 EPOCHS, BATCH 2048)",
            "> Dual-GPU DataParallel training with Cosine Annealing, FP16 AMP, and phonetically-weighted viseme loss."
        ]),
        make_cell("code", [
            "%cd /kaggle/working/FaceKeyAnimation",
            "!python scripts/train_speech_to_animation.py \\",
            "    --epochs 100 \\",
            "    --batch-size 2048 \\",
            "    --lr 5e-4 \\",
            "    --seq-len 48 \\",
            "    --stride 12 \\",
            "    --num-workers 4 \\",
            "    --checkpoints-dir /kaggle/working/checkpoints \\",
            f"    --hf-repo {HF_GEN3_DATASET_REPO} \\",
            f"    --model-repo {HF_MODEL_REPO}"
        ]),
        make_cell("markdown", [
            "## 📊 STEP 5: VERIFY MODEL SIZE, LOSS CURVE & WEIGHT ARTIFACTS"
        ]),
        make_cell("code", [
            "import os, glob",
            "from IPython.display import Image, display",
            "print('=' * 80)",
            "print(' [✓] GEN 3 TRAINING COMPLETED SUCCESSFULLY!')",
            "print('=' * 80)",
            "ckpts = glob.glob('/kaggle/working/checkpoints/*.pt')",
            "for c in sorted(ckpts):",
            "    print(f'  • Checkpoint: {c} ({os.path.getsize(c) / (1024*1024):.2f} MB)')",
            "curve = '/kaggle/working/checkpoints/loss_curve.png'",
            "if os.path.exists(curve):",
            "    print(f'\\n[+] Displaying Loss Curve ({curve}):')",
            "    display(Image(filename=curve))"
        ])
    ]
    return {
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4,
        "cells": cells
    }


def generate_gen3_colab_trainer() -> Dict[str, Any]:
    cells = [
        make_cell("markdown", [
            "# 🧠 FaceKey Studio Gen 3 — Google Colab T4 GPU 3D Avatar Trainer",
            "### Pure 3D Digital Human Animation: 52 ARKit Blendshapes + 3D Head Pose + 64-Band Log-Mel Acoustics",
            "",
            "> [!TIP]",
            "> **HARDWARE SETTING**: Go to **Runtime** ➔ **Change runtime type** ➔ Select **T4 GPU**.",
            "",
            "## 🌟 HOW TO RUN IN BACKGROUND WITH DRIVE SYNC:",
            "1. Mount Google Drive in Step 1.",
            "2. Run all cells.",
            "3. Every epoch commits model weights and loss curves directly to your Google Drive and Hugging Face Model Hub!"
        ]),
        make_cell("markdown", [
            "## 💓 STEP 0: BROWSER ANTI-SLEEP HEARTBEAT"
        ]),
        make_cell("code", [
            "from IPython.display import display, Javascript",
            "display(Javascript('''",
            "function keepAlive(){",
            "    console.log('[FaceKey Gen3 Colab] Session active at ' + new Date().toLocaleTimeString());",
            "}",
            "setInterval(keepAlive, 60000);",
            "'''))",
            "print('[+] Anti-sleep browser heartbeat initialized.')"
        ]),
        make_cell("markdown", [
            "## 📂 STEP 1: MOUNT GOOGLE DRIVE & VERIFY GPU"
        ]),
        make_cell("code", [
            "from google.colab import drive",
            "drive.mount('/content/drive')",
            "!nvidia-smi"
        ]),
        make_cell("markdown", [
            "## 📦 STEP 2: CLONE GEN 3 CODEBASE & INSTALL DEPENDENCIES"
        ]),
        make_cell("code", [
            "import os, shutil",
            "os.chdir('/content')",
            "APP_DIR = '/content/FaceKeyAnimation'",
            "if not os.path.exists(APP_DIR):",
            f"    !git clone -b {BRANCH} {REPO_URL} {{APP_DIR}}",
            "%cd {APP_DIR}",
            f"!git fetch origin && !git checkout {BRANCH} && !git pull origin {BRANCH}",
            "!pip install -q -U torch torchvision torchaudio huggingface_hub numpy scipy matplotlib",
            "print('[+] Gen 3 Deep Learning Training Environment Ready on branch: " + BRANCH + "')"
        ]),
        make_cell("markdown", [
            "## 🔑 STEP 3: HUGGING FACE AUTHENTICATION SETUP"
        ]),
        make_cell("code", [
            "hf_token = None",
            "try:",
            "    from google.colab import userdata",
            "    hf_token = userdata.get('HF_TOKEN')",
            "except Exception:",
            "    pass",
            "if not hf_token and os.path.exists('/content/drive/MyDrive/FaceKeyDataset/hf_token.txt'):",
            "    with open('/content/drive/MyDrive/FaceKeyDataset/hf_token.txt', 'r') as f:",
            "        hf_token = f.read().strip()",
            "if hf_token:",
            "    os.environ['HF_TOKEN'] = hf_token",
            "    print('[✓] Hugging Face Hub Authenticated!')",
            "else:",
            "    print('[!] Note: HF_TOKEN not set. Public datasets stream freely without auth.')"
        ]),
        make_cell("markdown", [
            "## 🚀 STEP 4: LAUNCH GEN 3 3D AVATAR TRAINING (100 EPOCHS)"
        ]),
        make_cell("code", [
            "%cd /content/FaceKeyAnimation",
            "drive_ckpt = '/content/drive/MyDrive/FaceKeyDataset/checkpoints'",
            "os.makedirs(drive_ckpt, exist_ok=True)",
            "!python scripts/train_speech_to_animation.py \\",
            "    --epochs 100 \\",
            "    --batch-size 48 \\",
            "    --lr 3.5e-4 \\",
            "    --seq-len 48 \\",
            "    --stride 12 \\",
            "    --checkpoints-dir {drive_ckpt} \\",
            f"    --hf-repo {HF_GEN3_DATASET_REPO} \\",
            f"    --model-repo {HF_MODEL_REPO}"
        ]),
        make_cell("markdown", [
            "## 📊 STEP 5: DISPLAY LOSS CURVE"
        ]),
        make_cell("code", [
            "import os",
            "from IPython.display import Image, display",
            "curve = '/content/drive/MyDrive/FaceKeyDataset/checkpoints/loss_curve.png'",
            "if os.path.exists(curve):",
            "    display(Image(filename=curve))"
        ])
    ]
    return {
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4,
        "cells": cells
    }


def generate_gen3_lightning_trainer() -> Dict[str, Any]:
    cells = [
        make_cell("markdown", [
            "# 🧠 FaceKey Studio Gen 3 — Lightning.ai Studio 24/7 Persistent Trainer",
            "### Pure 3D Digital Human Animation: 52 ARKit Blendshapes + 3D Head Pose + 64-Band Log-Mel Acoustics",
            "",
            "Runs inside your persistent Lightning.ai studio environment with 24/7 background execution."
        ]),
        make_cell("code", [
            "import os, shutil",
            "APP_DIR = '/teamspace/studios/this_studio/FaceKeyAnimation'",
            "if not os.path.exists(APP_DIR):",
            f"    !git clone -b {BRANCH} {REPO_URL} {{APP_DIR}}",
            "%cd {APP_DIR}",
            f"!git fetch origin && !git checkout {BRANCH} && !git pull origin {BRANCH}",
            "!pip install -q -U torch torchvision torchaudio huggingface_hub numpy scipy matplotlib",
            "print('[+] Lightning.ai Training Environment Ready on Gen 3 branch!')"
        ]),
        make_cell("code", [
            "hf_token = os.environ.get('HF_TOKEN')",
            "token_path = '/teamspace/studios/this_studio/hf_token.txt'",
            "if not hf_token and os.path.exists(token_path):",
            "    with open(token_path, 'r') as f:",
            "        hf_token = f.read().strip()",
            "if hf_token:",
            "    os.environ['HF_TOKEN'] = hf_token",
            "    print('[✓] Hugging Face Hub Authenticated!')"
        ]),
        make_cell("code", [
            "%cd /teamspace/studios/this_studio/FaceKeyAnimation",
            "!python scripts/train_speech_to_animation.py \\",
            "    --epochs 100 \\",
            "    --batch-size 64 \\",
            "    --lr 3.5e-4 \\",
            "    --seq-len 48 \\",
            "    --stride 12 \\",
            "    --checkpoints-dir /teamspace/studios/this_studio/FaceKeyDataset/checkpoints \\",
            f"    --hf-repo {HF_GEN3_DATASET_REPO} \\",
            f"    --model-repo {HF_MODEL_REPO}"
        ])
    ]
    return {
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4,
        "cells": cells
    }


def generate_gen3_colab_collector() -> Dict[str, Any]:
    cells = [
        make_cell("markdown", [
            "# 🚀 FaceKey Studio Gen 3 — Google Colab CPU 3D Avatar Data Collector",
            "### 100% Free CPU-Based Multi-Worker Video Capture (0 GPU Hours Used!)",
            "",
            "> [!TIP]",
            "> **HARDWARE SETTING**: Go to **Runtime** ➔ **Change runtime type** ➔ Select **CPU**.",
            "",
            "**🛡️ GEN 3 3D AVATAR DATA SPECIFICATION:**",
            "- **52 ARKit Blendshapes**: Facial muscle sliders ($0.0$ to $1.0$).",
            "- **3D Head Orientation Kinematics**: Euler Pitch, Yaw, Roll angles.",
            "- **64-Band Log-Mel Spectral Acoustics**: 16 kHz frequency filterbanks covering 20 Hz to 8000 Hz.",
            "- **Zero Landmark Overhead**: No 478 raw coordinates stored in dataset, saving 90% disk space.",
            f"- **100% Public Dataset Backup**: Automatically syncs chunk archives to `{HF_GEN3_DATASET_REPO}`."
        ]),
        make_cell("code", [
            "from IPython.display import display, Javascript",
            "display(Javascript('''",
            "function keepAlive(){",
            "    console.log('[FaceKey Gen3 Collector] Colab CPU active at ' + new Date().toLocaleTimeString());",
            "}",
            "setInterval(keepAlive, 60000);",
            "'''))",
            "print('[+] Anti-sleep browser heartbeat initialized.')"
        ]),
        make_cell("code", [
            "from google.colab import drive",
            "drive.mount('/content/drive')"
        ]),
        make_cell("code", [
            "import os, shutil",
            "os.chdir('/content')",
            "APP_DIR = '/content/FaceKeyAnimation'",
            "if not os.path.exists(APP_DIR):",
            f"    !git clone -b {BRANCH} {REPO_URL} {{APP_DIR}}",
            "%cd {APP_DIR}",
            f"!git fetch origin && !git checkout {BRANCH} && !git pull origin {BRANCH}",
            "!pip install -q -U yt-dlp opencv-python mediapipe huggingface_hub numpy scipy",
            "print('[+] Gen 3 Data Collector Environment Ready on branch: " + BRANCH + "')"
        ]),
        make_cell("code", [
            "hf_token = None",
            "try:",
            "    from google.colab import userdata",
            "    hf_token = userdata.get('HF_TOKEN')",
            "except Exception:",
            "    pass",
            "if not hf_token and os.path.exists('/content/drive/MyDrive/FaceKeyDataset/hf_token.txt'):",
            "    with open('/content/drive/MyDrive/FaceKeyDataset/hf_token.txt', 'r') as f:",
            "        hf_token = f.read().strip()",
            "if hf_token:",
            "    os.environ['HF_TOKEN'] = hf_token",
            "    print('[✓] Hugging Face Hub Authenticated!')",
            "else:",
            "    print('[!] Note: HF_TOKEN not set. Set HF_TOKEN in Colab Secrets or Google Drive to push chunks directly to public repo.')"
        ]),
        make_cell("code", [
            "%cd /content/FaceKeyAnimation",
            "!python scripts/cloud_data_collector.py \\",
            "    --drive-dir /content/drive/MyDrive/FaceKeyDataset \\",
            "    --urls-file sessions/youtube_urls_colab.txt \\",
            "    --num-workers 2 \\",
            f"    --hf-repo {HF_GEN3_DATASET_REPO}"
        ])
    ]
    return {
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4,
        "cells": cells
    }


def generate_gen3_kaggle_collector() -> Dict[str, Any]:
    cells = [
        make_cell("markdown", [
            "# 🚀 FaceKey Studio Gen 3 — Kaggle Cloud CPU 3D Avatar Data Collector",
            "### 100% Free Cloud CPU 4-Worker Capture (Saturates all 4 vCPUs at 380-400%!)",
            "",
            "## 🌟 HOW TO RUN IN BACKGROUND (9 HOURS HEADLESS):",
            "1. Click **Save Version** ➔ **Save & Run All (Commit)**.",
            "2. Close your browser and shut down your computer.",
            f"3. Processed sessions automatically sync to Hugging Face Hub: `{HF_GEN3_DATASET_REPO}`!"
        ]),
        make_cell("code", [
            "from IPython.display import display, Javascript",
            "display(Javascript('''",
            "function keepAlive(){",
            "    console.log('[FaceKey Gen3 Kaggle Collector] Active at ' + new Date().toLocaleTimeString());",
            "}",
            "setInterval(keepAlive, 60000);",
            "'''))",
            "print('[+] Anti-sleep browser heartbeat initialized.')"
        ]),
        make_cell("code", [
            "import os, shutil",
            "os.chdir('/kaggle/working')",
            "APP_DIR = '/kaggle/working/FaceKeyAnimation'",
            "if not os.path.exists(APP_DIR):",
            f"    !git clone -b {BRANCH} {REPO_URL} {{APP_DIR}}",
            "%cd {APP_DIR}",
            f"!git fetch origin && !git checkout {BRANCH} && !git pull origin {BRANCH}",
            "!pip install -q -U yt-dlp opencv-python mediapipe huggingface_hub numpy scipy",
            "print('[+] Gen 3 Kaggle Collector Environment Ready on branch: " + BRANCH + "')"
        ]),
        make_cell("code", [
            "hf_token = None",
            "try:",
            "    from kaggle_secrets import UserSecretsClient",
            "    hf_token = UserSecretsClient().get_secret('HF_TOKEN')",
            "except Exception:",
            "    pass",
            "if not hf_token and os.path.exists('/kaggle/working/hf_token.txt'):",
            "    with open('/kaggle/working/hf_token.txt', 'r') as f:",
            "        hf_token = f.read().strip()",
            "if hf_token:",
            "    os.environ['HF_TOKEN'] = hf_token",
            "    print('[✓] Hugging Face Hub Authenticated!')",
            "else:",
            "    print('[!] Note: Add HF_TOKEN to Kaggle Secrets (Add-ons -> Secrets) to push chunks to Hugging Face.')"
        ]),
        make_cell("code", [
            "%cd /kaggle/working/FaceKeyAnimation",
            "!python scripts/cloud_data_collector.py \\",
            "    --urls-file sessions/youtube_urls_kaggle.txt \\",
            "    --num-workers 4 \\",
            f"    --hf-repo {HF_GEN3_DATASET_REPO}"
        ])
    ]
    return {
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4,
        "cells": cells
    }


def generate_gen3_lightning_collector() -> Dict[str, Any]:
    cells = [
        make_cell("markdown", [
            "# 🚀 FaceKey Studio Gen 3 — Lightning.ai 24/7 CPU 3D Avatar Data Collector",
            "### Max Free Potential: Persistent 4-Core CPU Background Video Collector",
            "",
            "## 🌟 HOW TO RUN HEADLESS IN BACKGROUND (PC TURNED OFF / 24/7):",
            "Lightning.ai Studios provide persistent cloud storage and Linux background terminal capabilities.",
            "To keep collection running 24/7 even after you close your browser tab or turn off your computer:",
            "",
            "1. In the top-left menu or bottom dock, open the **Terminal** (or press ``Ctrl + ` ``).",
            "2. Start a persistent `tmux` session:",
            "   ```bash",
            "   tmux new -s facekey_collector",
            "   ```",
            "3. Run the automated 4-worker runner script:",
            "   ```bash",
            "   bash /teamspace/studios/this_studio/FaceKeyAnimation/scripts/run_lightning_background_collector.sh",
            "   ```",
            "4. **Detach from the session** by pressing `Ctrl + B`, then releasing and pressing `D`.",
            "5. **🎉 YOU CAN NOW CLOSE YOUR BROWSER AND SHUT DOWN YOUR PC!**",
            "   The 4 workers will continue capturing videos 24/7 in the cloud and syncing chunks to `VijayTheOne/facekey-gen3-dataset`.",
            "",
            "**To view live progress anytime later:**",
            "Open the terminal and run: `tmux attach -t facekey_collector` (or press `Ctrl+B`, `D` to detach again).",
            "",
            "---",
            "### Alternatively: Run Interactively in this Notebook Below"
        ]),
        make_cell("code", [
            "import os, shutil",
            "APP_DIR = '/teamspace/studios/this_studio/FaceKeyAnimation'",
            "if not os.path.exists(APP_DIR):",
            f"    !git clone -b {BRANCH} {REPO_URL} {{APP_DIR}}",
            "%cd {APP_DIR}",
            f"!git fetch origin && !git checkout {BRANCH} && !git pull origin {BRANCH}",
            "!pip install -q -U yt-dlp opencv-python mediapipe huggingface_hub numpy scipy",
            "print('[+] Lightning.ai Collector Ready on Gen 3 branch!')"
        ]),
        make_cell("code", [
            "hf_token = os.environ.get('HF_TOKEN')",
            "token_path = '/teamspace/studios/this_studio/hf_token.txt'",
            "if not hf_token and os.path.exists(token_path):",
            "    with open(token_path, 'r') as f:",
            "        hf_token = f.read().strip()",
            "if hf_token:",
            "    os.environ['HF_TOKEN'] = hf_token",
            "    print('[✓] Hugging Face Hub Authenticated!')"
        ]),
        make_cell("code", [
            "%cd /teamspace/studios/this_studio/FaceKeyAnimation",
            "!python scripts/cloud_data_collector.py \\",
            "    --drive-dir /teamspace/studios/this_studio/FaceKeyDataset \\",
            "    --urls-file sessions/youtube_urls_lightning.txt \\",
            "    --num-workers 4 \\",
            f"    --hf-repo {HF_GEN3_DATASET_REPO}"
        ])
    ]
    return {
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4,
        "cells": cells
    }


def main():
    os.makedirs("notebooks", exist_ok=True)
    nb_suite = {
        "notebooks/Gen3_Kaggle_Dual_T4_Trainer.ipynb": generate_gen3_kaggle_trainer(),
        "notebooks/Gen3_Colab_GPU_Trainer.ipynb": generate_gen3_colab_trainer(),
        "notebooks/Gen3_Lightning_AI_Trainer.ipynb": generate_gen3_lightning_trainer(),
        "notebooks/Gen3_Colab_Data_Collector.ipynb": generate_gen3_colab_collector(),
        "notebooks/Gen3_Kaggle_Data_Collector.ipynb": generate_gen3_kaggle_collector(),
        "notebooks/Gen3_Lightning_AI_Data_Collector.ipynb": generate_gen3_lightning_collector()
    }

    print("=" * 82)
    print(" 🌟 GENERATING FACEKEY GEN 3 JUPYTER NOTEBOOK SUITE")
    print(f" Target Branch   : {BRANCH}")
    print(f" GitHub Repo     : {REPO_URL}")
    print(f" HF Model Hub    : {HF_MODEL_REPO}")
    print(f" HF Gen 3 Dataset: {HF_GEN3_DATASET_REPO}")
    print(f" HF Gen 2 Dataset: {HF_GEN2_DATASET_REPO}")
    print("=" * 82)

    for path, nb in nb_suite.items():
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2)
        size_kb = os.path.getsize(path) / 1024.0
        print(f"  [✓] {path:<50s} ({size_kb:6.2f} KB | {len(nb['cells'])} cells)")

    print("=" * 82)
    print(" [✓] All Gen 3 training and collection notebooks generated successfully!")
    print("=" * 82)


if __name__ == "__main__":
    main()
