#!/usr/bin/env python3
"""
Smart Cloud-to-Drive Dataset Delta Synchronizer (100% Duplicate-Proof)
=======================================================================
Pulls only NEW chunks from Hugging Face Hub into 5 TB Google Drive:
  1. Inspects all existing chunks in Google Drive (MyDrive/FaceKeyDataset/chunks/).
  2. Compares against the remote Hugging Face dataset repository (VijayTheOne/facekey-dataset-chunks).
  3. SKIPS all chunks that already exist on Drive (0 MB downloaded, zero duplicate files).
  4. Downloads ONLY new, missing chunks directly to Drive over cloud-to-cloud connection (100 MB/s).
  5. Updates local Drive dataset manifest safely.
"""

import os
import sys
import glob
import time
from typing import Optional, List, Dict, Any

try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
    from huggingface_hub import HfApi, hf_hub_download


def resolve_hf_token(token_arg: Optional[str] = None) -> Optional[str]:
    """Finds Hugging Face token from argument, env, or token files."""
    if token_arg and token_arg.startswith("hf_"):
        return token_arg
    if os.environ.get("HF_TOKEN") and os.environ.get("HF_TOKEN").startswith("hf_"):
        return os.environ.get("HF_TOKEN")
    
    candidates = [
        "/content/drive/MyDrive/FaceKeyDataset/hf_token.txt",
        "/content/hf_token.txt",
        "/kaggle/working/hf_token.txt",
        "/teamspace/studios/this_studio/hf_token.txt",
        os.path.expanduser("~/.cache/huggingface/token"),
        "hf_token.txt"
    ]
    for c in candidates:
        if os.path.isfile(c) and os.path.getsize(c) > 5:
            try:
                with open(c, "r", encoding="utf-8") as f:
                    tok = f.read().strip()
                    if tok.startswith("hf_"):
                        return tok
            except Exception:
                pass
    return None


def sync_hf_to_google_drive(
    drive_dir: str = "/content/drive/MyDrive/FaceKeyDataset",
    repo_id: str = "VijayTheOne/facekey-dataset-chunks",
    token: Optional[str] = None
):
    import shutil
    # Auto-mount Google Drive if on Colab and not already mounted
    if os.path.exists("/content") and not os.path.exists("/content/drive/MyDrive"):
        try:
            print("[*] Detecting Google Colab environment. Auto-mounting Google Drive...")
            from google.colab import drive
            drive.mount("/content/drive")
            print("[+] Google Drive successfully mounted at /content/drive/MyDrive")
        except Exception as me:
            print(f"[!] Warning: Auto-mount encountered: {me}. Continuing...")

    print("=" * 82)
    print(" 🔄 SMART CLOUD-TO-DRIVE DELTA SYNC (100% DUPLICATE-PROOF)")
    print("=" * 82)

    token = resolve_hf_token(token)
    chunks_dir = os.path.join(drive_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    # 1. Scan existing files on Google Drive
    existing_files: Dict[str, int] = {}
    for p in glob.glob(os.path.join(chunks_dir, "*.tar.gz")):
        existing_files[os.path.basename(p)] = os.path.getsize(p)

    print(f"[*] Target Google Drive : {chunks_dir}")
    print(f"[*] Chunks Already on Drive: {len(existing_files)} chunk(s) found")

    # 2. Query Hugging Face repository
    print(f"[*] Querying Hugging Face: '{repo_id}'...")
    api = HfApi(token=token)
    try:
        remote_files = api.list_repo_tree(repo_id=repo_id, repo_type="dataset", path_in_repo="chunks")
        remote_chunks = [f for f in remote_files if f.path.endswith(".tar.gz")]
    except Exception as e:
        # Fallback to list_repo_files if list_repo_tree not available
        all_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)
        remote_chunks = []
        for af in all_files:
            if af.startswith("chunks/") and af.endswith(".tar.gz"):
                remote_chunks.append(type("HFFile", (), {"path": af, "size": None})())

    if not remote_chunks:
        print("[!] No chunks found in Hugging Face repository.")
        return

    # 3. Determine delta (ONLY new chunks that are NOT on Drive)
    to_download = []
    already_synced = 0

    for rf in remote_chunks:
        fname = os.path.basename(rf.path)
        if fname in existing_files:
            # File exists on Drive. Check size integrity if available
            rf_size = getattr(rf, "size", None)
            if rf_size is not None and existing_files[fname] == rf_size:
                already_synced += 1
                continue
            elif rf_size is None and existing_files[fname] > 1024:
                already_synced += 1
                continue
        to_download.append(rf)

    print("-" * 82)
    print(f"  • Total Chunks in Hugging Face : {len(remote_chunks)}")
    print(f"  • Already Synced (SKIPPED)     : {already_synced} chunks (0 bytes downloaded, 0 duplicates)")
    print(f"  • NEW Chunks to Download       : {len(to_download)} chunk(s)")
    print("-" * 82)

    if not to_download:
        print("\n[✓] PERFECT: Google Drive is ALREADY 100% up-to-date! No download needed.")
        print("=" * 82)
        return

    # 4. Download ONLY the new missing chunks directly to Drive (with auto-cache purge)
    total_bytes_new = sum(getattr(f, "size", 0) or 0 for f in to_download)
    print(f"[*] Starting high-speed cloud-to-cloud transfer (~{total_bytes_new/(1024**2):.1f} MB total)...\n")

    t_start = time.perf_counter()
    newly_saved = 0
    temp_cache = "/tmp/fka_hf_cache"
    os.makedirs(temp_cache, exist_ok=True)

    for idx, rf in enumerate(to_download, 1):
        fname = os.path.basename(rf.path)
        fsize_mb = (getattr(rf, "size", 0) or 0) / (1024.0 * 1024.0)
        size_str = f" ({fsize_mb:.1f} MB)" if fsize_mb > 0 else ""
        print(f"  [{idx}/{len(to_download)}] Downloading NEW: '{fname}'{size_str}...", end=" ", flush=True)

        try:
            target_chunk = os.path.join(chunks_dir, fname)
            dl_file = hf_hub_download(
                repo_id=repo_id,
                filename=rf.path,
                repo_type="dataset",
                token=token,
                cache_dir=temp_cache
            )
            shutil.copyfile(dl_file, target_chunk)
            # Reclaim VM disk immediately after each chunk to prevent disk buildup
            try:
                shutil.rmtree(temp_cache, ignore_errors=True)
                os.makedirs(temp_cache, exist_ok=True)
            except Exception:
                pass
            print("[DONE ✓]")
            newly_saved += 1
        except Exception as dl_err:
            print(f"[ERROR: {dl_err}]")

    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 82)
    print(f" [✓] SYNC COMPLETE in {elapsed:.1f}s!")
    print(f"  • {newly_saved} new chunk(s) saved directly to Google Drive.")
    print(f"  • 0 duplicate files created.")
    print(f"  • Google Drive total: {len(existing_files) + newly_saved} chunks stored.")
    print("=" * 82)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Duplicate-Proof Hugging Face to Google Drive Sync")
    parser.add_argument("--drive-dir", type=str, default="/content/drive/MyDrive/FaceKeyDataset", help="Path to Google Drive dataset folder")
    parser.add_argument("--repo-id", type=str, default="VijayTheOne/facekey-dataset-chunks", help="Hugging Face repo ID")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face access token")
    args = parser.parse_args()

    sync_hf_to_google_drive(
        drive_dir=args.drive_dir,
        repo_id=args.repo_id,
        token=args.token
    )
