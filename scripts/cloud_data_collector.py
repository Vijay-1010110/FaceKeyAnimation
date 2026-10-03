"""
Google Colab & Cloud Batch Data Collector with Auto-Chunking to Google Drive
=============================================================================
Runs on Google Colab or Kaggle cloud environments:
  1. Mounts Google Drive (/content/drive/MyDrive/FaceKeyDataset).
  2. Reads 'youtubeURLtoProcess.txt' from Drive (or local repository).
  3. Downloads and extracts facial motion + audio features in silent 480p.
  4. Automatically packages recorded sessions into .tar.gz chunks on Google Drive.
  5. Purges local temporary sessions to stay within disk limits (< 5 GB).
  6. Auto-resumes from Google Drive state if Colab disconnects or times out.
"""

import os
import sys
import time
import argparse
import subprocess
import shutil

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.storage.cloud_sync import CloudDriveSync
from src.storage.stream_queue import StreamBatchQueue
from src.utils.notifier import notify_user


def run_cloud_collector(
    drive_dir: Optional[str] = None,
    chunk_size: int = 5,
    quality: str = "480p",
    purge_local: bool = True
):
    print("=" * 82)
    print(" ☁️  FACEKEY CLOUD DATA COLLECTOR & GOOGLE DRIVE CHUNKER")
    print("=" * 82)
    
    script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cloud_sync = CloudDriveSync(project_root=script_dir, drive_folder=drive_dir)
    
    # 1. Mount Drive if on Colab
    cloud_sync.mount_google_drive()

    print(f"[*] Google Drive Root : {cloud_sync.drive_folder}")
    print(f"[*] Chunks Storage    : {cloud_sync.chunks_dir}")
    print(f"[*] Manifest File     : {cloud_sync.manifest_path}")

    # Ensure Drive queue file exists
    drive_urls_file = os.path.join(cloud_sync.drive_folder, "youtubeURLtoProcess.txt")
    local_urls_file = os.path.join(script_dir, "sessions", "youtubeURLtoProcess.txt")

    if not os.path.exists(drive_urls_file):
        if os.path.exists(local_urls_file):
            shutil.copy2(local_urls_file, drive_urls_file)
        else:
            with open(drive_urls_file, "w", encoding="utf-8") as f:
                f.write("# FaceKey Batch YouTube Stream URLs\n# Paste YouTube links here (one per line). New URLs are detected automatically!\n\n")

    # Local ephemeral session directory
    local_sessions_dir = os.path.join(script_dir, "sessions")
    os.makedirs(local_sessions_dir, exist_ok=True)

    # Initialize queue pointing to Drive file
    queue = StreamBatchQueue(project_root=script_dir, urls_file=drive_urls_file)
    queue.reset_interrupted_to_pending()
    new_added, completed, pending = queue.sync_from_file()

    summary = queue.get_progress_summary()
    print("-" * 82)
    print(f"[*] Queue Status   : {summary['queue_total']} Total | {summary['queue_completed']} Completed | {summary['queue_pending']} Pending")
    print(f"[*] Phase 1 Target : {summary['total_dataset_hours']:.2f} hrs / 100.0 hrs ({summary['phase1_progress_pct']:.1f}%)")
    print(f"[*] Est. 100h Size : ~{summary['estimated_gb_100h']:.1f} GB (Chunks stored safely on Google Drive)")
    print("=" * 82 + "\n")

    unpacked_count = 0
    python_exe = sys.executable

    while True:
        # Check for new URLs in Drive text file
        new_u, _, _ = queue.sync_from_file()
        if new_u > 0:
            print(f"[+] Synced {new_u} newly added URL(s) from Drive queue file!")

        item = queue.get_next_pending()
        if not item:
            print("\n[✓] All queued YouTube URLs have been processed!")
            # Pack any remaining local sessions into Drive chunk
            cloud_sync.pack_sessions_into_chunk(
                sessions_dir=local_sessions_dir,
                purge_local_after_pack=purge_local
            )
            print(f"[*] Watching '{drive_urls_file}' on Google Drive for new links... (Ctrl+C to stop)")
            time.sleep(5)
            continue

        key = item["key"]
        raw_url = item["raw_url"]
        print("-" * 75)
        print(f"[*] [COLAB INGEST] Video #{queue.state['stats']['total_completed'] + 1}: {raw_url}")

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
                title = info.get('title', 'YouTube Video')
                canonical_url = info.get('webpage_url') or item["canonical_url"]
        except Exception as e:
            err = f"yt-dlp extraction error: {e}"
            print(f"[!] {err}")
            queue.mark_failed(key, err)
            time.sleep(2)
            continue

        queue.mark_in_progress(key, title=title)
        print(f"[*] Connected: '{title}' ({quality})")
        print(f"[*] Processing silently on cloud compute...")

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
            subprocess.call(cmd)
            elapsed_sec = time.perf_counter() - t_start
        except KeyboardInterrupt:
            print("\n[*] Interrupted! Finalizing active session...")
            break

        queue.mark_completed(
            key=key,
            session_id=f"session_{key}",
            frame_count=int(elapsed_sec * 30),
            duration_seconds=elapsed_sec,
            title=title
        )

        unpacked_count += 1
        print(f"[✓] Completed '{title[:35]}' in {elapsed_sec/60.0:.1f} mins.")

        # Pack into chunk every `chunk_size` sessions to free Colab disk space immediately!
        if unpacked_count >= chunk_size:
            print(f"\n[*] Reached chunk batch limit ({unpacked_count} sessions). Packaging chunk to Google Drive...")
            cloud_sync.pack_sessions_into_chunk(
                sessions_dir=local_sessions_dir,
                purge_local_after_pack=purge_local
            )
            unpacked_count = 0

        time.sleep(1.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cloud Batch YouTube Collector & Google Drive Chunker")
    parser.add_argument("--drive-dir", type=str, default=None, help="Google Drive storage directory path")
    parser.add_argument("--chunk-size", type=int, default=5, help="Number of sessions per .tar.gz chunk on Drive")
    parser.add_argument("--quality", type=str, default="480p", help="Video resolution stream quality")
    parser.add_argument("--no-purge", action="store_true", default=False, help="Do not delete local sessions after chunking")
    args = parser.parse_args()

    run_cloud_collector(
        drive_dir=args.drive_dir,
        chunk_size=args.chunk_size,
        quality=args.quality,
        purge_local=not args.no_purge
    )
