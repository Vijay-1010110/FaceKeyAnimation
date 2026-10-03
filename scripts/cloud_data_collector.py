"""
Google Colab, Kaggle & Multi-Worker Cloud Batch Data Collector with Distributed Locking
========================================================================================
Supports concurrent parallel workers (Colab, Kaggle, Local PC) running at the same time:
  1. Mounts Google Drive (optimized for large 5 TB plans).
  2. Uses Distributed File Locks (Drive/locks/<video_id>.lock.json) to prevent duplicate downloads.
  3. Automatic Stale Lock Recovery: If a Colab/Kaggle session terminates unexpectedly or times out,
     abandoned locks are reclaimed without human intervention.
  4. Heartbeat monitoring keeps active locks alive during video processing.
  5. Packages sessions into worker-tagged .tar.gz chunks on Google Drive.
  6. Purges local temporary sessions to stay within disk limits (< 5 GB).
"""

import os
import sys
import time
import argparse
import subprocess
import shutil
from typing import Optional, Dict, Any

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.storage.cloud_sync import CloudDriveSync
from src.storage.cloud_coordinator import CloudCoordinator
from src.storage.stream_queue import StreamBatchQueue
from src.utils.notifier import notify_user


def run_cloud_collector(
    drive_dir: Optional[str] = None,
    worker_id: Optional[str] = None,
    urls_file: Optional[str] = None,
    chunk_size: int = 5,
    quality: str = "480p",
    purge_local: bool = True,
    stale_timeout_sec: int = 1200
):
    print("=" * 84)
    print(" [CLOUD] FACEKEY MULTI-WORKER CLOUD COLLECTOR & DISTRIBUTED LOCK MANAGER")
    print("=" * 84)

    script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cloud_sync = CloudDriveSync(project_root=script_dir, drive_folder=drive_dir)

    # 1. Mount Drive / verify storage
    cloud_sync.mount_google_drive()

    # 2. Initialize Distributed Coordinator
    coordinator = CloudCoordinator(
        drive_folder=cloud_sync.drive_folder,
        worker_id=worker_id,
        stale_timeout_sec=stale_timeout_sec
    )

    print(f"[*] Worker ID         : {coordinator.worker_id}")
    print(f"[*] Google Drive Root : {cloud_sync.drive_folder}")
    print(f"[*] Chunks Storage    : {cloud_sync.chunks_dir}")
    print(f"[*] Active Locks Dir  : {coordinator.locks_dir}")
    print(f"[*] Shared DB Path    : {coordinator.completed_db_path}")

    # 3. Clean any stale locks left by previously crashed sessions
    stale_cleaned = coordinator.clean_stale_locks()
    if stale_cleaned > 0:
        print(f"[+] Auto-reclaimed {stale_cleaned} stale lock(s) from terminated/expired sessions!")

    # 3b. Auto-purge any falsely completed entries from early crashes (< 15s)
    coordinator.purge_corrupted_completed(min_duration_sec=15.0)

    # 4. Resolve URL file
    if urls_file and os.path.exists(urls_file):
        target_urls_file = os.path.abspath(urls_file)
    elif urls_file and os.path.exists(os.path.join(cloud_sync.drive_folder, os.path.basename(urls_file))):
        target_urls_file = os.path.join(cloud_sync.drive_folder, os.path.basename(urls_file))
    elif cloud_sync.is_colab and os.path.exists(cloud_sync.colab_urls_path):
        target_urls_file = cloud_sync.colab_urls_path
    elif cloud_sync.is_kaggle and os.path.exists(cloud_sync.kaggle_urls_path):
        target_urls_file = cloud_sync.kaggle_urls_path
    elif os.path.exists(cloud_sync.shared_urls_path):
        target_urls_file = cloud_sync.shared_urls_path
    else:
        target_urls_file = os.path.join(script_dir, "sessions", "youtubeURLtoProcess.txt")

    print(f"[*] Active URLs Queue : {target_urls_file}")

    # Ephemeral local storage
    local_sessions_dir = os.path.join(script_dir, "sessions")
    os.makedirs(local_sessions_dir, exist_ok=True)

    # Initialize batch queue for tracking state
    queue = StreamBatchQueue(project_root=script_dir, urls_file=target_urls_file)
    queue.reset_interrupted_to_pending()
    queue.purge_corrupted_completed(min_duration_sec=15.0)
    new_added, completed, pending = queue.sync_from_file()

    shared_db = coordinator.get_completed_keys()
    total_globally_completed = shared_db.get("total_completed", len(shared_db.get("completed", {})))
    total_global_hours = shared_db.get("total_hours", 0.0)

    summary = queue.get_progress_summary()
    print("-" * 84)
    print(f"[*] Queue File Status  : {summary['queue_total']} Total | {summary['queue_completed']} Done | {summary['queue_pending']} Pending")
    print(f"[*] Global Multi-Cloud : {total_globally_completed} Videos Completed ({total_global_hours:.2f} hrs across all workers)")
    print(f"[*] Google Drive Plan  : 5 TB Capacity (Storage headroom is virtually unlimited)")
    print("=" * 84 + "\n")

    unpacked_count = 0
    python_exe = sys.executable

    while True:
        # Re-check for newly appended URLs
        new_u, _, _ = queue.sync_from_file()
        if new_u > 0:
            print(f"[+] Synced {new_u} new URL(s) dynamically added to queue file!")

        item = queue.get_next_pending()
        if not item:
            print("\n[+] All queued YouTube URLs in this file have been processed!")
            # Pack any remaining local sessions into Drive chunk
            cloud_sync.pack_sessions_into_chunk(
                sessions_dir=local_sessions_dir,
                worker_tag=coordinator.worker_id,
                purge_local_after_pack=purge_local
            )
            print(f"[*] Watching '{target_urls_file}' for new links... (Ctrl+C to stop)")
            time.sleep(8)
            continue

        key = item["key"]
        raw_url = item["raw_url"]
        item_title = item.get("title", "Pending Resolution")

        # -------------------------------------------------------------
        # DISTRIBUTED ATOMIC CLAIM: Check if another worker is doing it
        # -------------------------------------------------------------
        claimed, claim_reason, claim_key = coordinator.try_claim_url(raw_url, title=item_title)

        if not claimed:
            if claim_reason == "already_completed":
                print(f"[*] Video '{key}' is already completed in shared Drive registry. Skipping...")
                queue.mark_completed(key, session_id="shared_cloud", frame_count=0, duration_seconds=0, title=item_title)
            else:
                print(f"[-] Video '{key}' is {claim_reason}. Skipping to avoid duplicate work...")
                # Temporarily pass this item so we can look at the next one
                item["status"] = "locked_by_peer"
            time.sleep(0.5)
            continue

        print("-" * 75)
        print(f"[*] [WORKER {coordinator.worker_id}] CLAIMED Video: {raw_url}")

        # Extract 480p direct stream via yt-dlp
        try:
            import yt_dlp
            ydl_opts = {
                'format': f'best[height<={quality.replace("p","")}]/bestvideo[height<={quality.replace("p","")}]/best',
                'quiet': True,
                'no_warnings': True,
                'skip_download': True,
                'cachedir': False,
                'source_address': '0.0.0.0',
                'socket_timeout': 30,
                'retries': 5,
                'extractor_retries': 5
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(raw_url, download=False)
                stream_url = info.get('url')
                title = info.get('title', item_title)
                canonical_url = info.get('webpage_url') or item["canonical_url"]
        except Exception as e:
            err = f"yt-dlp extraction error: {e}"
            print(f"[!] {err}")
            coordinator.release_lock(key)
            queue.mark_failed(key, err)
            time.sleep(2)
            continue

        queue.mark_in_progress(key, title=title)
        print(f"[*] Connected: '{title}' ({quality})")
        print(f"[*] Processing silently on cloud worker with live heartbeat...")

        t_start = time.perf_counter()
        cmd = [
            python_exe, os.path.join(script_dir, "test_face_speaker_tool.py"),
            "--mode", "stream",
            "--stream-url", stream_url,
            "--quality", quality,
            "--headless",
            "--stream-title", title,
            "--canonical-url", canonical_url
        ]

        try:
            ret_code = subprocess.call(cmd)
            elapsed_sec = time.perf_counter() - t_start
        except KeyboardInterrupt:
            print(f"\n[*] Interrupted! Releasing lock for '{key}'...")
            coordinator.release_lock(key)
            break
        except Exception as e:
            print(f"[!] Subprocess error: {e}. Releasing lock...")
            coordinator.release_lock(key)
            queue.mark_failed(key, str(e))
            continue

        if ret_code != 0 or elapsed_sec < 5.0:
            print(f"[!] Process failed or exited too quickly (exit code {ret_code}, duration {elapsed_sec:.1f}s). Releasing lock and marking as failed!")
            coordinator.release_lock(key)
            queue.mark_failed(key, f"Process exited with code {ret_code} in {elapsed_sec:.1f}s")
            time.sleep(2)
            continue

        # Register completion in both global coordinator and local queue
        session_id = f"session_{key}"
        frame_count = int(elapsed_sec * 30)

        coordinator.mark_completed(
            key=key,
            canonical_url=canonical_url,
            title=title,
            session_id=session_id,
            frame_count=frame_count,
            duration_seconds=elapsed_sec
        )

        queue.mark_completed(
            key=key,
            session_id=session_id,
            frame_count=frame_count,
            duration_seconds=elapsed_sec,
            title=title
        )

        unpacked_count += 1
        print(f"[+] Finished '{title[:40]}' ({elapsed_sec/60.0:.1f} mins). Lock released and registered!")

        # Pack into chunk every `chunk_size` sessions to free local disk immediately!
        if unpacked_count >= chunk_size:
            print(f"\n[*] Reached batch threshold ({unpacked_count} sessions). Packaging chunk into 5 TB Google Drive...")
            cloud_sync.pack_sessions_into_chunk(
                sessions_dir=local_sessions_dir,
                worker_tag=coordinator.worker_id,
                purge_local_after_pack=purge_local
            )
            unpacked_count = 0

        time.sleep(1.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Worker Cloud Batch YouTube Collector & Distributed Lock Manager")
    parser.add_argument("--drive-dir", type=str, default=None, help="Google Drive storage directory path")
    parser.add_argument("--worker-id", type=str, default=None, help="Unique worker name (e.g. colab-1, kaggle-1)")
    parser.add_argument("--urls-file", type=str, default=None, help="Path to YouTube URLs queue text file")
    parser.add_argument("--chunk-size", type=int, default=5, help="Number of sessions per .tar.gz chunk on Drive")
    parser.add_argument("--quality", type=str, default="480p", help="Video resolution stream quality")
    parser.add_argument("--stale-timeout", type=int, default=1200, help="Heartbeat expiration in seconds (default 20 mins)")
    parser.add_argument("--no-purge", action="store_true", default=False, help="Do not delete local sessions after chunking")
    args = parser.parse_args()

    run_cloud_collector(
        drive_dir=args.drive_dir,
        worker_id=args.worker_id,
        urls_file=args.urls_file,
        chunk_size=args.chunk_size,
        quality=args.quality,
        purge_local=not args.no_purge,
        stale_timeout_sec=args.stale_timeout
    )
