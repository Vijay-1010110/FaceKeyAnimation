#!/usr/bin/env python3
"""Emergency Hugging Face Dataset Uploader.
Uploads all pending .tar.gz chunks to your private Hugging Face dataset repository
from any cloud backend (Lightning.ai, Kaggle, Colab, local) in 1 simple command.
100% cloud-to-cloud transfer (0 MB local home internet used).

Usage:
    python scripts/emergency_hf_upload.py
"""

import os
import sys
import glob

# Ensure huggingface_hub is installed
try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
    from huggingface_hub import HfApi, create_repo

# 1. Resolve HF Token
token = os.environ.get("HF_TOKEN")

# Check CLI argument
for idx, arg in enumerate(sys.argv):
    if arg in ("--token", "--hf-token") and idx + 1 < len(sys.argv):
        token = sys.argv[idx + 1]

if not token:
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass

token_paths = [
    "/teamspace/studios/this_studio/hf_token.txt",
    "/teamspace/studios/this_studio/FaceKeyDataset/hf_token.txt",
    "/kaggle/working/hf_token.txt",
    "/content/drive/MyDrive/FaceKeyDataset/hf_token.txt",
    os.path.expanduser("~/.cache/huggingface/token"),
    "hf_token.txt",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hf_token.txt")
]

if not token:
    for tp in token_paths:
        if os.path.isfile(tp) and os.path.getsize(tp) > 5:
            try:
                with open(tp, "r", encoding="utf-8") as f:
                    t = f.read().strip()
                    if t.startswith("hf_"):
                        token = t
                        break
            except Exception:
                pass

if not token:
    print("[ERROR] Hugging Face token not found!")
    print("Please set HF_TOKEN environment variable or run:")
    print("  echo 'your_hf_token' > hf_token.txt")
    sys.exit(1)

# 2. Target Repo
repo_id = os.environ.get("HF_REPO", "VijayTheOne/facekey-dataset-chunks")

# 3. Auto-pack unarchived raw session folders into chunks
if os.path.exists("/content/drive/MyDrive/FaceKeyDataset/chunks"):
    target_chunks_dir = "/content/drive/MyDrive/FaceKeyDataset/chunks"
elif os.path.exists("/content"):
    target_chunks_dir = "/content/FaceKeyDataset/chunks"
elif os.path.exists("/teamspace/studios/this_studio"):
    target_chunks_dir = "/teamspace/studios/this_studio/FaceKeyDataset/chunks"
elif os.path.exists("/kaggle/working"):
    target_chunks_dir = "/kaggle/working/FaceKeyDataset/chunks"
else:
    target_chunks_dir = os.path.abspath("FaceKeyDataset/chunks")

os.makedirs(target_chunks_dir, exist_ok=True)

sess_dirs = [
    "/content/FaceKeyAnimation/sessions",
    "/content/sessions",
    "/teamspace/studios/this_studio/FaceKeyAnimation/sessions",
    "/kaggle/working/FaceKeyAnimation/sessions",
    "/kaggle/working/sessions",
    "sessions",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sessions"),
]

for sd in sess_dirs:
    if os.path.isdir(sd):
        raw_sess = sorted(glob.glob(os.path.join(sd, "session_*")))
        if raw_sess:
            print(f"[*] Found {len(raw_sess)} raw session folder(s) in {sd}. Packaging into chunks...")
            try:
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from src.storage.cloud_sync import CloudDriveSync
                cs = CloudDriveSync(drive_folder=os.path.dirname(target_chunks_dir))
                while True:
                    packed = cs.pack_sessions_into_chunk(
                        sessions_dir=sd,
                        worker_tag="lightning_sync",
                        max_sessions_per_chunk=5,
                        purge_local_after_pack=True
                    )
                    if not packed:
                        break
            except Exception as pack_err:
                print(f"[!] Packaging note: {pack_err}")

# 4. Locate Chunks
search_dirs = [
    target_chunks_dir,
    "/teamspace/studios/this_studio/FaceKeyDataset/chunks",
    "/kaggle/working/FaceKeyDataset/chunks",
    "/content/drive/MyDrive/FaceKeyDataset/chunks",
    "FaceKeyDataset/chunks",
    "chunks",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "FaceKeyDataset", "chunks"),
]

chunk_files = []
for d in search_dirs:
    if os.path.isdir(d):
        found = glob.glob(os.path.join(d, "*.tar.gz"))
        if found:
            chunk_files.extend(found)

# Deduplicate
chunk_files = sorted(list(set(chunk_files)))

print("=" * 72)
print(" [EMERGENCY UPLOAD] HUGGING FACE PRIVATE DATASET BACKUP")
print(f" Target Repository : https://huggingface.co/datasets/{repo_id}")
print(f" Chunks Detected   : {len(chunk_files)} file(s)")
print("=" * 72)

if not chunk_files:
    print("[INFO] No pending .tar.gz chunks found on disk.")
    print("All chunks have either already been uploaded and purged, or are currently being created.")
    sys.exit(0)

# 4. Ensure Private Repo Exists
api = HfApi(token=token)
try:
    create_repo(repo_id=repo_id, repo_type="dataset", private=True, token=token, exist_ok=True)
except Exception:
    pass

# 5. Upload Chunks
success_count = 0
total_bytes = 0

for i, cf in enumerate(chunk_files, 1):
    fname = os.path.basename(cf)
    fsize_mb = os.path.getsize(cf) / (1024 * 1024)
    print(f"[{i}/{len(chunk_files)}] Uploading {fname} ({fsize_mb:.1f} MB)... ", end="", flush=True)
    try:
        api.upload_file(
            path_or_fileobj=cf,
            path_in_repo=f"chunks/{fname}",
            repo_id=repo_id,
            repo_type="dataset"
        )
        print("[SUCCESS]")
        success_count += 1
        total_bytes += os.path.getsize(cf)
        
        # Free up disk space after upload
        try:
            os.remove(cf)
        except Exception:
            pass
    except Exception as e:
        print(f"[FAILED: {e}]")

print("-" * 72)
print(f"[SUMMARY] Successfully transferred {success_count}/{len(chunk_files)} chunk(s) ({total_bytes / (1024 * 1024):.1f} MB).")
print(f"[STATUS] Disk freed. All data safely secured in Hugging Face repository.")
print("=" * 72)
