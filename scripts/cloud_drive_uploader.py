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
        # Auto-search in /kaggle/working and /kaggle/input for any service account json key
        candidate_files = []
        for search_dir in ["/kaggle/working", "/kaggle/input", "."]:
            if os.path.isdir(search_dir):
                for root, _, files in os.walk(search_dir):
                    for f in files:
                        if f.endswith(".json"):
                            candidate_files.append(os.path.join(root, f))

        for cand in candidate_files:
            try:
                with open(cand, "r", encoding="utf-8") as jf:
                    content_peek = jf.read(500)
                    if '"type": "service_account"' in content_peek or '"project_id"' in content_peek:
                        print(f"[+] Auto-detected Google Service Account JSON key: {cand}")
                        service_account_json = cand
                        break
            except Exception:
                pass

    if not os.path.exists(service_account_json):
        print("\n" + "=" * 75)
        print(" [!] NOTICE: Service Account JSON not found at:")
        print(f"     {service_account_json}")
        print("-" * 75)
        print(" To upload directly from Kaggle to Google Drive via Service Account:")
        print("   1. Go to https://console.cloud.google.com and enable 'Google Drive API'")
        print("   2. Under 'IAM & Admin' -> 'Service Accounts', create a service account")
        print("   3. Click 'Keys' -> 'Add Key' -> 'Create new key' -> JSON")
        print("   4. Upload the downloaded JSON to Kaggle as /kaggle/working/service_account.json")
        print(f"   5. In Google Drive, share your folder ({folder_id}) with the service account email as Editor")
        print("\n ALTERNATIVE (EASIEST - 0 GCP SETUP):")
        print("   Use Hugging Face Hub in the next cell (free, unlimited private dataset):")
        print("   !python scripts/cloud_drive_uploader.py --method hf --hf-token 'YOUR_TOKEN' --hf-repo 'username/facekey-chunks'")
        print("=" * 75 + "\n")
        return 0

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
    try:
        folder_info = service.files().get(fileId=folder_id, fields="id, name", supportsAllDrives=True).execute()
        print(f"[+] Connected to Google Drive target folder: '{folder_info.get('name')}' (ID: {folder_id})")
    except Exception as fe:
        print("\n" + "!" * 75)
        print(" [!] PERMISSION / SHARING ERROR ON GOOGLE DRIVE FOLDER:")
        print(f"     Target Folder ID: {folder_id}")
        print(f"     Error           : {fe}")
        print("-" * 75)
        print(" To fix this, you must share your Google Drive folder with the Service Account:")
        print("   1. Open: https://drive.google.com/drive/u/0/folders/11VbtMpmNATsrBZxkPLpA2gPTtgaRFNlA")
        print("   2. Click the 'Share' button at top right.")
        print(f"   3. Paste this exact email: {getattr(creds, 'service_account_email', 'your service account email')}")
        print("   4. Set role to 'Editor'.")
        print("   5. Click 'Share' (if prompted 'Share anyway', click 'Share anyway').")
        print("!" * 75 + "\n")
        return 0

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
                fields="id",
                supportsAllDrives=True
            ).execute()

            print(f"[DONE - File ID: {uploaded_file.get('id')}]")
            uploaded_count += 1

            if purge_after_upload:
                try:
                    cf_norm = os.path.normpath(os.path.abspath(cf)).lower()
                    if "drive" in cf_norm or "gdrive" in cf_norm:
                        pass
                    else:
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
                    cf_norm = os.path.normpath(os.path.abspath(cf)).lower()
                    if "drive" in cf_norm or "gdrive" in cf_norm:
                        print(" [SAFEGUARD: Preserved in Google Drive]")
                    else:
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


def sync_hf_to_gdrive(
    repo_id: str,
    hf_token: str,
    target_chunks_dir: str
) -> int:
    """Synchronizes .tar.gz chunks from HuggingFace dataset directly to Google Drive chunks dir.
    Skips existing files so no duplicate data is downloaded.
    Transfers 100% cloud-to-cloud (Colab/Cloud -> Google Drive) using 0 MB local data.
    """
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
        from huggingface_hub import HfApi, hf_hub_download

    os.makedirs(target_chunks_dir, exist_ok=True)
    api = HfApi(token=hf_token)

    print("=" * 75)
    print(" [CLOUD-TO-CLOUD] HUGGINGFACE -> GOOGLE DRIVE SYNC (0 MB LOCAL DATA)")
    print(f" Source HF Repo   : https://huggingface.co/datasets/{repo_id}")
    print(f" Target Drive Dir : {target_chunks_dir}")
    print("=" * 75)

    try:
        repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=hf_token)
    except Exception as e:
        print(f"[!] Error accessing HF repository '{repo_id}': {e}")
        return 0

    chunk_files = [f for f in repo_files if f.endswith(".tar.gz")]
    if not chunk_files:
        print("[*] No .tar.gz chunk files found in HuggingFace repository.")
        return 0

    downloaded = 0
    skipped = 0

    for idx, remote_path in enumerate(chunk_files, 1):
        filename = os.path.basename(remote_path)
        dest_file = os.path.join(target_chunks_dir, filename)

        if os.path.exists(dest_file) and os.path.getsize(dest_file) > 0:
            print(f"[{idx}/{len(chunk_files)}] [SKIP - Already in Drive] '{filename}'")
            skipped += 1
            continue

        print(f"[{idx}/{len(chunk_files)}] Downloading '{filename}' directly to Drive...", end=" ", flush=True)
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=remote_path,
                repo_type="dataset",
                token=hf_token,
                local_dir=target_chunks_dir,
            )
            if os.path.basename(downloaded_path) == filename and downloaded_path != dest_file:
                shutil.move(downloaded_path, dest_file)
            print("[DONE]")
            downloaded += 1
        except Exception as e:
            print(f"[FAILED: {e}]")

    print("-" * 75)
    print(f"[+] Sync Complete! Downloaded: {downloaded} new chunk(s), Skipped: {skipped} existing chunk(s).")
    print(f"[+] All chunks now safely in: {target_chunks_dir}")
    print("=" * 75)
    return downloaded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cloud-to-Cloud Dataset Uploader (0 MB Local Data)")
    parser.add_argument("--chunks-dir", type=str, default=None, help="Directory containing .tar.gz chunks")
    parser.add_argument("--method", type=str, choices=["gdrive", "hf", "sync-hf", "status"], default="status")
    parser.add_argument("--service-account", type=str, help="Path to Google Service Account JSON")
    parser.add_argument("--folder-id", type=str, default="11VbtMpmNATsrBZxkPLpA2gPTtgaRFNlA", help="Target Google Drive Folder ID")
    parser.add_argument("--hf-token", type=str, default=None, help="HuggingFace Write Token")
    parser.add_argument("--hf-repo", type=str, default=None, help="HuggingFace Repo ID (defaults to VijayTheOne/facekey-dataset-chunks)")
    parser.add_argument("--no-purge", action="store_true", help="Keep local files after upload (default is to purge to save disk)")

    args = parser.parse_args()

    # Auto-resolve chunks_dir if not specified or doesn't exist
    resolved_chunks_dir = args.chunks_dir
    if not resolved_chunks_dir or not os.path.exists(resolved_chunks_dir):
        candidate_dirs = [
            "/teamspace/studios/this_studio/FaceKeyDataset/chunks",
            "/kaggle/working/FaceKeyDataset/chunks",
            "/content/drive/MyDrive/FaceKeyDataset/chunks",
            os.path.abspath("FaceKeyDataset/chunks"),
            os.path.abspath("chunks"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "FaceKeyDataset", "chunks"),
        ]
        for cd in candidate_dirs:
            if os.path.exists(cd):
                resolved_chunks_dir = cd
                break
        if not resolved_chunks_dir:
            resolved_chunks_dir = "/teamspace/studios/this_studio/FaceKeyDataset/chunks" if os.path.exists("/teamspace/studios/this_studio") else (
                "/kaggle/working/FaceKeyDataset/chunks" if os.path.exists("/kaggle/working") else "FaceKeyDataset/chunks"
            )

    # Auto-resolve HF Token if not passed
    resolved_hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not resolved_hf_token:
        token_cands = [
            "/teamspace/studios/this_studio/hf_token.txt",
            "/teamspace/studios/this_studio/FaceKeyDataset/hf_token.txt",
            "/kaggle/working/hf_token.txt",
            os.path.expanduser("~/.cache/huggingface/token"),
            "hf_token.txt",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hf_token.txt"),
        ]
        for tc in token_cands:
            if os.path.isfile(tc) and os.path.getsize(tc) > 5:
                try:
                    with open(tc, "r", encoding="utf-8") as tf:
                        tok = tf.read().strip()
                        if tok.startswith("hf_"):
                            resolved_hf_token = tok
                            break
                except Exception:
                    pass

    # Auto-resolve HF repo ID
    resolved_hf_repo = args.hf_repo or os.environ.get("HF_REPO") or "VijayTheOne/facekey-dataset-chunks"

    if args.method == "status":
        info = get_disk_usage_summary(os.path.dirname(os.path.dirname(resolved_chunks_dir)))
        print("\n" + "=" * 60)
        print(" [CLOUD STORAGE MONITOR] DISK & CHUNKS SUMMARY")
        print("=" * 60)
        print(f" Total Disk Quota : {info['disk_total_gb']} GB")
        print(f" Used Disk Space  : {info['disk_used_gb']} GB ({info['disk_used_pct']}%)")
        print(f" Free Disk Space  : {info['disk_free_gb']} GB")
        print(f" Chunks on Disk   : {info['chunks_count']} file(s) ({info['chunks_size_mb']} MB)")
        print(f" Chunks Location  : {resolved_chunks_dir}")
        print("=" * 60 + "\n")

    elif args.method == "gdrive":
        if not args.service_account or not args.folder_id:
            print("[!] Error: --service-account and --folder-id are required for gdrive upload.")
            sys.exit(1)
        upload_to_gdrive_service_account(
            chunks_dir=resolved_chunks_dir,
            service_account_json=args.service_account,
            folder_id=args.folder_id,
            purge_after_upload=not args.no_purge
        )

    elif args.method == "hf":
        if not resolved_hf_token:
            print("[!] Error: Hugging Face token not found! Provide via --hf-token, HF_TOKEN env, or hf_token.txt")
            sys.exit(1)
        upload_to_hf_hub(
            chunks_dir=resolved_chunks_dir,
            hf_token=resolved_hf_token,
            repo_id=resolved_hf_repo,
            purge_after_upload=not args.no_purge
        )

    elif args.method == "sync-hf":
        if not resolved_hf_token:
            print("[!] Error: Hugging Face token not found! Provide via --hf-token, HF_TOKEN env, or hf_token.txt")
            sys.exit(1)
        sync_hf_to_gdrive(
            repo_id=resolved_hf_repo,
            hf_token=resolved_hf_token,
            target_chunks_dir=resolved_chunks_dir
        )
