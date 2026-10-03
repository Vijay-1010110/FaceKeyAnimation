#!/usr/bin/env python3
"""Diagnostic Verification Tool for FaceKey Multi-Cloud Data Collector.
Inspects active worker processes, active video locks, raw session folders on disk,
chunk files, Hugging Face cloud sync status, and queue progress.

Usage:
    python scripts/verify_collector_status.py
"""

import os
import sys
import glob
import json
import time
import subprocess

def get_dir_size_mb(path: str) -> float:
    total = 0
    if not os.path.exists(path):
        return 0.0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except Exception:
                pass
    return total / (1024 * 1024)

print("=" * 80)
print(" [FACEKEY DIAGNOSTICS] REAL-TIME CLOUD COLLECTOR STATUS VERIFIER")
print("=" * 80)

script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 1. Check Running Processes
print("\n[1] RUNNING PROCESSES:")
try:
    ps_proc = subprocess.run(["ps", "-eo", "pid,user,%cpu,%mem,etime,cmd"], capture_output=True, text=True)
    lines = [l for l in ps_proc.stdout.splitlines() if any(k in l for k in ["cloud_data_collector", "test_face_speaker", "run_lightning", "yt-dlp", "warp-svc"])]
    if lines:
        for l in lines[:10]:
            print(f"  -> {l[:110]}")
    else:
        print("  [!] No active FaceKey collector processes detected.")
except Exception as pe:
    print(f"  [!] Process check: {pe}")

# 2. Check Active Locks
print("\n[2] CURRENT VIDEO LOCKS (Active Downloads & Inferences):")
lock_dirs = [
    "/teamspace/studios/this_studio/FaceKeyDataset/locks",
    "/kaggle/working/FaceKeyDataset/locks",
    "/content/drive/MyDrive/FaceKeyDataset/locks",
    os.path.join(script_dir, "FaceKeyDataset", "locks")
]
found_locks = 0
for ld in lock_dirs:
    if os.path.isdir(ld):
        lfiles = glob.glob(os.path.join(ld, "*.json"))
        for lf in lfiles:
            try:
                with open(lf, "r", encoding="utf-8") as f:
                    ldata = json.load(f)
                    worker = ldata.get("worker_id", "unknown")
                    title = ldata.get("title", "Unknown Title")
                    url = ldata.get("url", "unknown url")
                    age = time.time() - ldata.get("timestamp", time.time())
                    print(f"  -> Worker: {worker} | Video: '{title[:40]}' | Age: {age:.0f}s")
                    found_locks += 1
            except Exception:
                pass
if found_locks == 0:
    print("  [!] No active video locks currently held.")

# 3. Check Raw Session Folders on Disk
print("\n[3] RAW SESSION FOLDERS (Locally Extracted Face Data):")
sess_dirs = [
    os.path.join(script_dir, "sessions"),
    "/teamspace/studios/this_studio/FaceKeyAnimation/sessions",
    "/kaggle/working/FaceKeyAnimation/sessions"
]
total_sessions = 0
total_frames_count = 0
unpacked_list = []
for sd in sess_dirs:
    if os.path.isdir(sd):
        s_folders = sorted(glob.glob(os.path.join(sd, "session_*")))
        for sf in s_folders:
            meta_path = os.path.join(sf, "metadata.json")
            frames_count = 0
            dur_sec = 0.0
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as mf:
                        m = json.load(mf)
                        frames_count = m.get("total_video_frames", 0)
                        dur_sec = m.get("duration_seconds", 0.0)
                except Exception:
                    pass
            sz_mb = get_dir_size_mb(sf)
            total_sessions += 1
            total_frames_count += frames_count
            unpacked_list.append((sf, frames_count, dur_sec, sz_mb))

if unpacked_list:
    print(f"  [+] Found {total_sessions} raw session folder(s) ({total_frames_count:,} frames total):")
    for sf, fc, ds, sz in unpacked_list[:8]:
        print(f"      - {os.path.basename(sf)}: {fc:,} frames ({ds:.1f}s speech) - {sz:.1f} MB")
    if len(unpacked_list) > 8:
        print(f"      ... and {len(unpacked_list) - 8} more session folders.")
else:
    print("  [!] 0 raw session folders found on disk.")

# 4. Check Packaged Chunks on Disk
print("\n[4] PACKAGED .tar.gz CHUNKS (Disk Ready for Upload):")
chunk_dirs = [
    "/teamspace/studios/this_studio/FaceKeyDataset/chunks",
    "/kaggle/working/FaceKeyDataset/chunks",
    "/content/drive/MyDrive/FaceKeyDataset/chunks",
    os.path.join(script_dir, "FaceKeyDataset", "chunks")
]
found_chunks = []
for cd in chunk_dirs:
    if os.path.isdir(cd):
        cfiles = glob.glob(os.path.join(cd, "*.tar.gz"))
        found_chunks.extend(cfiles)

found_chunks = sorted(list(set(found_chunks)))
if found_chunks:
    print(f"  [+] Found {len(found_chunks)} local chunk file(s):")
    for cf in found_chunks:
        sz_mb = os.path.getsize(cf) / (1024 * 1024)
        print(f"      - {os.path.basename(cf)} ({sz_mb:.1f} MB)")
else:
    print("  [!] 0 local .tar.gz chunk files found.")

# 5. Check Hugging Face Remote Repository
print("\n[5] HUGGING FACE REMOTE REPOSITORY (VijayTheOne/facekey-dataset-chunks):")
hf_token = os.environ.get("HF_TOKEN")
for tp in [
    "/teamspace/studios/this_studio/hf_token.txt",
    "/teamspace/studios/this_studio/FaceKeyDataset/hf_token.txt",
    "hf_token.txt",
    os.path.join(script_dir, "hf_token.txt")
]:
    if not hf_token and os.path.isfile(tp) and os.path.getsize(tp) > 5:
        try:
            with open(tp, "r", encoding="utf-8") as tf:
                hf_token = tf.read().strip()
        except Exception:
            pass

if hf_token:
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        repo_files = api.list_repo_files(repo_id="VijayTheOne/facekey-dataset-chunks", repo_type="dataset")
        remote_chunks = [f for f in repo_files if f.startswith("chunks/") and f.endswith(".tar.gz")]
        print(f"  [+] Total chunks verified safe on Hugging Face: {len(remote_chunks)}")
        for rc in remote_chunks[-5:]:
            print(f"      - https://huggingface.co/datasets/VijayTheOne/facekey-dataset-chunks/blob/main/{rc}")
    except Exception as he:
        print(f"  [!] HF status check: {he}")
else:
    print("  [!] HF Token not found to inspect remote cloud repo.")

# 6. Queue & Completed Video Registry
print("\n[6] COMPLETED / FAILED REGISTRY:")
comp_files = [
    "/teamspace/studios/this_studio/FaceKeyDataset/completed_videos.json",
    "/kaggle/working/FaceKeyDataset/completed_videos.json",
    os.path.join(script_dir, "sessions", "stream_queue_cache.json")
]
for cf in comp_files:
    if os.path.isfile(cf):
        try:
            with open(cf, "r", encoding="utf-8") as f:
                cdata = json.load(f)
                if isinstance(cdata, dict) and "completed" in cdata:
                    c_list = cdata.get("completed", [])
                    f_list = cdata.get("failed", [])
                    print(f"  -> File: {os.path.basename(cf)} | Completed: {len(c_list)} | Failed: {len(f_list)}")
                    if c_list:
                        print("     Latest completed:")
                        for c_item in c_list[-3:]:
                            t = c_item.get("title", "Unknown")
                            frames = c_item.get("frame_count", 0)
                            print(f"       * '{t[:35]}' ({frames:,} frames)")
                    if f_list:
                        print("     Latest failed:")
                        for f_item in f_list[-3:]:
                            t = f_item.get("title", f_item.get("url", "Unknown"))
                            err = f_item.get("error", "Error")
                            print(f"       * '{t[:35]}' -> {err}")
        except Exception:
            pass

print("\n" + "=" * 80)
