#!/usr/bin/env python3
"""
Kaggle CPU Swarm Collector (100% Free - Unlimited Hours)
========================================================
Runs on Kaggle's 4 CPU cores (Accelerator: None).
Utilizes zero GPU quota, runs up to 12 hours in background mode,
and continuously streams compressed dataset chunks directly to Hugging Face Hub.
"""

import os
import sys
import subprocess

def run():
    print("=" * 76)
    print(" 🦅 KAGGLE CPU SWARM COLLECTOR (4 CORES / ZERO GPU QUOTA CONSUMED)")
    print("=" * 76)

    # 1. Resolve HF Token
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        try:
            from kaggle_secrets import UserSecretsClient
            hf_token = UserSecretsClient().get_secret("HF_TOKEN")
            print("[+] Retrieved HF_TOKEN from Kaggle Secrets")
        except Exception:
            pass

    candidates = [
        "/kaggle/working/hf_token.txt",
        "hf_token.txt"
    ]
    if not hf_token:
        for c in candidates:
            if os.path.exists(c):
                with open(c, "r") as f:
                    tok = f.read().strip()
                    if tok.startswith("hf_"):
                        hf_token = tok
                        break

    if not hf_token:
        print("[!] Warning: HF_TOKEN not found in Kaggle Secrets or hf_token.txt.")
        print("[!] Please add 'HF_TOKEN' to Kaggle Notebook -> Add-ons -> Secrets.")

    # 2. Set optimized thread environment
    os.environ["OPENCV_FFMPEG_THREADS"] = "4"
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # 3. Ensure ffmpeg is installed
    subprocess.call(["apt-get", "update", "-qq"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.call(["apt-get", "install", "-y", "-qq", "ffmpeg"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 4. Install dependencies
    cmd_install = [
        sys.executable, "-m", "pip", "install", "-q", "-U",
        "numpy<2", "yt-dlp[default]", "mediapipe", "opencv-python-headless",
        "sounddevice", "mss", "av", "scipy", "huggingface_hub", "pysocks", "requests", "tqdm"
    ]
    subprocess.check_call(cmd_install)

    # 5. Launch Collector with 4 Workers
    worker_id = f"kaggle-cpu-{os.getpid()}"
    cmd_run = [
        sys.executable, "scripts/cloud_data_collector.py",
        "--worker-id", worker_id,
        "--urls-file", "sessions/youtube_urls_kaggle.txt",
        "--chunk-size", "1",
        "--quality", "480p",
        "--num-workers", "4",
        "--hf-repo", "VijayTheOne/facekey-dataset-chunks"
    ]
    if hf_token:
        cmd_run.extend(["--hf-token", hf_token])

    print(f"[*] Starting 4 Parallel Workers: {worker_id}...")
    sys.exit(subprocess.call(cmd_run))

if __name__ == "__main__":
    run()
