"""Cloud-to-Cloud Dataset Uploader for Headless Cloud Backends (Kaggle -> Google Drive).
Enables transferring .tar.gz chunks directly from Kaggle to Google Drive without consuming
any local home internet bandwidth (0 MB local data usage).

Supports:
  1. Direct Google Drive API via Service Account JSON (recommended direct cloud push)
  2. HuggingFace Hub Private Dataset Bridge (zero GCP config, 100% cloud-to-cloud)
  3. Disk Space Monitoring & Safe Local Purging (< 20 GB Kaggle disk quota)
"""

import os
import sys
import glob
import time
import shutil
import argparse
from typing import Optional, List, Dict, Any


def get_disk_usage_summary(target_dir: str = "/kaggle/working") -> Dict[str, Any]:
    """Inspect disk usage of working directory and chunks folder."""
    if not os.path.exists(target_dir):
        target_dir = "."
    
    total, used, free = shutil.disk_usage(target_dir)
    chunks_dir = os.path.join(target_dir, "FaceKeyDataset", "chunks")
    chunks_size = 0
    chunks_count = 0
    
    if os.path.exists(chunks_dir):
        chunk_files = glob.glob(os.path.join(chunks_dir, "*.tar.gz"))
        chunks_count = len(chunk_files)
        chunks_size = sum(os.path.getsize(f) for f in chunk_files)
        
    return {
        "disk_total_gb": round(total / (1024**3), 2),
        "disk_used_gb": round(used / (1024**3), 2),
        "disk_free_gb": round(free / (1024**3), 2),
        "disk_used_pct": round((used / total) * 100, 1),
        "chunks_count": chunks_count,
        "chunks_size_mb": round(chunks_size / (1024**2), 2),
        "chunks_dir": chunks_dir
    }


def upload_to_gdrive_service_account(
    chunks_dir: str,
    service_account_json: str,
    folder_id: str,
    purge_after_upload: bool = True
) -> int:
    """Uploads .tar.gz chunks directly to Google Drive via a Service Account JSON.
    Zero home internet used (transfers purely cloud-to-cloud).
    """
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
    except ImportError:
        print("[!] Installing google-api-python-client and google-auth...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "google-api-python-client", "google-auth"])
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload

    if not os.path.exists(service_account_json):
        raise FileNotFoundError(f"Service account JSON not found at: {service_account_json}")

    print("=" * 75)
    print(" [CLOUD-TO-CLOUD] DIRECT GOOGLE DRIVE UPLOADER (0 MB LOCAL DATA)")
    print(f" Source Directory  : {chunks_dir}")
    print(f" Target Folder ID  : {folder_id}")
    print(f" Credentials Key   : {service_account_json}")
    print("=" * 75)

    creds = service_account.Credentials.from_service_account_file(
        service_account_json,
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    service = build("drive", "v3", credentials=creds)

    chunk_files = sorted(glob.glob(os.path.join(chunks_dir, "*.tar.gz")))
    if not chunk_files:
        print("[*] No .tar.gz chunks found to upload.")
        return 0

    print(f"[*] Found {len(chunk_files)} chunk(s) ready to transfer to Google Drive...")
    uploaded_count = 0
    total_freed_mb = 0.0

    for idx, cf in enumerate(chunk_files, 1):
        filename = os.path.basename(cf)
        filesize_mb = os.path.getsize(cf) / (1024 * 1024)
        print(f"[{idx}/{len(chunk_files)}] Uploading '{filename}' ({filesize_mb:.1f} MB)...", end=" ", flush=True)

        try:
            file_metadata = {
                "name": filename,
                "parents": [folder_id]
            }
            media = MediaFileUpload(cf, resumable=True)
            uploaded_file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields="id"
            ).execute()

            print(f"[DONE - File ID: {uploaded_file.get('id')}]")
            uploaded_count += 1

            if purge_after_upload:
                try:
                    os.remove(cf)
                    total_freed_mb += filesize_mb
                except Exception:
                    pass
        except Exception as e:
            print(f"[FAILED: {e}]")

    print("-" * 75)
    print(f"[+] Successfully transferred {uploaded_count}/{len(chunk_files)} chunk(s) directly to 5 TB Google Drive!")
    if purge_after_upload:
        print(f"[+] Freed {total_freed_mb:.1f} MB from Kaggle local disk. Usage reset safely.")
    print("=" * 75)
    return uploaded_count


def upload_to_hf_hub(
    chunks_dir: str,
    hf_token: str,
    repo_id: str,
    purge_after_upload: bool = True
) -> int:
    """Uploads .tar.gz chunks to a private HuggingFace Dataset repository.
    100% free, unlimited storage, and zero Google Cloud configuration needed.
    Can be pulled into Google Drive using 1-line in Colab without using local data!
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
        from huggingface_hub import HfApi, create_repo

    api = HfApi(token=hf_token)
    try:
        create_repo(repo_id=repo_id, repo_type="dataset", private=True, token=hf_token, exist_ok=True)
    except Exception:
        pass

    chunk_files = sorted(glob.glob(os.path.join(chunks_dir, "*.tar.gz")))
    if not chunk_files:
        print("[*] No .tar.gz chunks found to upload.")
        return 0

    print("=" * 75)
    print(" [CLOUD-TO-CLOUD] HUGGINGFACE HUB DATASET BRIDGE (0 MB LOCAL DATA)")
    print(f" Source Directory  : {chunks_dir}")
    print(f" Target Dataset    : https://huggingface.co/datasets/{repo_id} (Private)")
    print(f" Pending Chunks    : {len(chunk_files)}")
    print("=" * 75)

    uploaded_count = 0
    total_freed_mb = 0.0

    for idx, cf in enumerate(chunk_files, 1):
        filename = os.path.basename(cf)
        filesize_mb = os.path.getsize(cf) / (1024 * 1024)
        print(f"[{idx}/{len(chunk_files)}] Uploading '{filename}' ({filesize_mb:.1f} MB)...", end=" ", flush=True)

        try:
            api.upload_file(
                path_or_fileobj=cf,
                path_in_repo=f"chunks/{filename}",
                repo_id=repo_id,
                repo_type="dataset"
            )
            print("[DONE]")
            uploaded_count += 1
            if purge_after_upload:
                try:
                    os.remove(cf)
                    total_freed_mb += filesize_mb
                except Exception:
                    pass
        except Exception as e:
            print(f"[FAILED: {e}]")

    print("-" * 75)
    print(f"[+] Successfully backed up {uploaded_count}/{len(chunk_files)} chunk(s) to HuggingFace Private Dataset!")
    if purge_after_upload:
        print(f"[+] Freed {total_freed_mb:.1f} MB from Kaggle local disk space.")
    print("=" * 75)
    return uploaded_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cloud-to-Cloud Dataset Uploader (0 MB Local Data)")
    parser.add_argument("--chunks-dir", type=str, default="/kaggle/working/FaceKeyDataset/chunks")
    parser.add_argument("--method", type=str, choices=["gdrive", "hf", "status"], default="status")
    parser.add_argument("--service-account", type=str, help="Path to Google Service Account JSON")
    parser.add_argument("--folder-id", type=str, default="11VbtMpmNATsrBZxkPLpA2gPTtgaRFNlA", help="Target Google Drive Folder ID")
    parser.add_argument("--hf-token", type=str, help="HuggingFace Write Token")
    parser.add_argument("--hf-repo", type=str, help="HuggingFace Repo ID (e.g. username/facekey-chunks)")
    parser.add_argument("--no-purge", action="store_true", help="Keep local files after upload (default is to purge to save disk)")

    args = parser.parse_args()

    if args.method == "status":
        info = get_disk_usage_summary(os.path.dirname(os.path.dirname(args.chunks_dir)))
        print("\n" + "=" * 60)
        print(" [KAGGLE STORAGE MONITOR] DISK & CHUNKS SUMMARY")
        print("=" * 60)
        print(f" Total Disk Quota : {info['disk_total_gb']} GB")
        print(f" Used Disk Space  : {info['disk_used_gb']} GB ({info['disk_used_pct']}%)")
        print(f" Free Disk Space  : {info['disk_free_gb']} GB")
        print(f" Chunks on Disk   : {info['chunks_count']} file(s) ({info['chunks_size_mb']} MB)")
        print(f" Chunks Location  : {info['chunks_dir']}")
        print("=" * 60 + "\n")

    elif args.method == "gdrive":
        if not args.service_account or not args.folder_id:
            print("[!] Error: --service-account and --folder-id are required for gdrive upload.")
            sys.exit(1)
        upload_to_gdrive_service_account(
            chunks_dir=args.chunks_dir,
            service_account_json=args.service_account,
            folder_id=args.folder_id,
            purge_after_upload=not args.no_purge
        )

    elif args.method == "hf":
        if not args.hf_token or not args.hf_repo:
            print("[!] Error: --hf-token and --hf-repo are required for hf upload.")
            sys.exit(1)
        upload_to_hf_hub(
            chunks_dir=args.chunks_dir,
            hf_token=args.hf_token,
            repo_id=args.hf_repo,
            purge_after_upload=not args.no_purge
        )
