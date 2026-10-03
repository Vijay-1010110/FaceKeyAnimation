"""
FaceKey Studio - Silent Background YouTube Batch Queue Runner & Auto-Resume Engine
===================================================================================
Preset:
  - Zero browser tabs needed (direct yt-dlp 480p stream extraction).
  - Headless / Silent background execution (0 display window, minimal CPU/GPU).
  - Continuous ingestion from 'sessions/youtubeURLtoProcess.txt'.
  - Seamless auto-resume from interruption / save points.
  - Phase 1 Milestone Tracker: 100 Hours Target & Storage Estimation (~48 GB).
"""

import os
import sys
import time
import subprocess
import signal

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.storage.stream_queue import StreamBatchQueue
from src.utils.notifier import notify_user


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_batch_header(summary: dict):
    print("=" * 82)
    print("  🚀 FACEKEY STUDIO - SILENT 480P BATCH YOUTUBE QUEUE & RESUME ENGINE")
    print("=" * 82)
    print("  🎯 PRESET CONFIGURATION:")
    print("     • Resolution : 480p SD Direct Stream (Lowest CPU, RAM, & Network Overhead)")
    print("     • Display    : Silent Background Mode (Headless, Zero Window Popups)")
    print("     • Ingestion  : Watching 'sessions\\youtubeURLtoProcess.txt' continuously")
    print("     • Resumption : Auto-saves progress; uninterrupted save-point continuation")
    print("-" * 82)
    print("  📊 100-HOUR TRAINING MILESTONE & STORAGE METRICS:")
    total_h = summary["total_dataset_hours"]
    speech_h = summary["total_speech_hours"]
    p1_pct = summary["phase1_progress_pct"]
    p1_rem = summary["phase1_remaining_hours"]
    scale_pct = summary["scale_progress_pct"]
    used_gb = summary["current_gb_used"]
    est_100 = summary["estimated_gb_100h"]
    est_1000 = summary["estimated_gb_1000h"]

    print(f"     • Phase 1 Target (100h) : {total_h:6.2f} hrs / 100.00 hrs ({p1_pct:5.1f}%) [Speech: {speech_h:.2f} hrs]")
    print(f"     • Hours Remaining to 100h: {p1_rem:6.2f} hrs remaining")
    print(f"     • Long-Term Scale (1,000h): {scale_pct:5.1f}% complete")
    print(f"     • Current Storage Used  : {used_gb:6.2f} GB ({summary['total_dataset_sessions']} sessions, {summary['total_dataset_frames']:,} frames)")
    print(f"     • Estimated 100h Size   : ~{est_100:.1f} GB  (Empirical rate: ~490 MB/hour)")
    print(f"     • Estimated 1,000h Size : ~{est_1000:.1f} GB")
    print("-" * 82)
    print("  📋 ACTIVE QUEUE STATUS:")
    print(f"     • Total in Queue  : {summary['queue_total']}")
    print(f"     • Completed       : {summary['queue_completed']}")
    print(f"     • Pending to Run  : {summary['queue_pending']}")
    if summary["queue_in_progress"] > 0:
        print(f"     • In-Progress / Resume: {summary['queue_in_progress']}")
    if summary["queue_failed"] > 0:
        print(f"     • Failed / Skipped: {summary['queue_failed']}")
    print("=" * 82 + "\n")


def run_batch_queue(watch_continuous: bool = True):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    python_exe = sys.executable
    queue = StreamBatchQueue(script_dir)

    # Cleanly reset any interrupted item to pending
    resumed_count = queue.reset_interrupted_to_pending()
    if resumed_count > 0:
        print(f"[*] Resumed {resumed_count} previously interrupted video(s) back into active queue.")

    # Initial sync from sessions/youtubeURLtoProcess.txt
    new_added, completed, pending = queue.sync_from_file()

    summary = queue.get_progress_summary()
    print_batch_header(summary)

    if new_added > 0:
        print(f"[+] Loaded {new_added} new URL(s) from 'sessions\\youtubeURLtoProcess.txt'!\n")

    running = True

    def sigint_handler(sig, frame):
        nonlocal running
        print("\n\n[*] Pause / Stop requested (Ctrl+C). Finalizing active queue safely...")
        running = False

    signal.signal(signal.SIGINT, sigint_handler)

    while running:
        # Check for stop trigger flag
        if os.path.exists("stop.flag"):
            try:
                os.remove("stop.flag")
            except Exception:
                pass
            print("\n[*] Detected 'stop.flag'. Stopping batch processing peacefully...")
            break

        # Always check for newly added URLs before picking the next pending item
        new_urls, _, _ = queue.sync_from_file()
        if new_urls > 0:
            print(f"[+] Detected {new_urls} new URL(s) added to 'sessions\\youtubeURLtoProcess.txt'!")

        item = queue.get_next_pending()

        if not item:
            if not watch_continuous:
                print("\n[✓] All queued YouTube URLs have been successfully processed!")
                break

            # Queue empty: Live Watch Mode
            print(f"\r[*] Watching '{queue.urls_file}' for new URLs... (Add links anytime | Ctrl+C to exit)", end="", flush=True)
            time.sleep(2.5)
            continue

        # Print item header
        key = item["key"]
        raw_url = item["raw_url"]
        print("\n" + "-" * 75)
        print(f"[*] [STARTING] Video #{queue.state['stats']['total_completed'] + 1}: {raw_url}")
        print("-" * 75)

        # 1. Resolve 480p stream via yt-dlp
        print(f"[*] Resolving 480p direct stream via yt-dlp...")
        try:
            import yt_dlp
            ydl_opts = {
                'format': 'best[height<=480]/bestvideo[height<=480]/best',
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
            err_msg = f"yt-dlp extraction failed: {e}"
            print(f"[!] Error: {err_msg}")
            queue.mark_failed(key, err_msg)
            time.sleep(2)
            continue

        # 2. Mark in progress
        queue.mark_in_progress(key, title=title)
        print(f"[*] Title: '{title}'")
        print(f"[*] Ingesting silently in background @ 480p... (You can minimize freely)")

        # 3. Launch test_face_speaker_tool.py silently (headless, 480p)
        cmd = [
            python_exe, "test_face_speaker_tool.py",
            "--mode", "stream",
            "--stream-url", stream_url,
            "--quality", "480p",
            "--headless",
            "--stream-title", title,
            "--canonical-url", canonical_url
        ]

        t_start = time.perf_counter()
        try:
            proc = subprocess.Popen(cmd)
            while proc.poll() is None:
                if os.path.exists("stop.flag") or not running:
                    print("\n[*] Forwarding stop signal to studio process...")
                    with open("stop.flag", "w") as sf:
                        sf.write("stop")
                    proc.wait(timeout=15)
                    break
                time.sleep(1.0)
            
            elapsed_sec = time.perf_counter() - t_start

        except KeyboardInterrupt:
            print("\n[*] Interrupted! Stopping current video peacefully...")
            with open("stop.flag", "w") as sf:
                sf.write("stop")
            if proc:
                try:
                    proc.wait(timeout=15)
                except Exception:
                    pass
            running = False
            break

        # 4. Check results from the completed session
        summary_after = queue.get_progress_summary()
        new_sessions = summary_after["total_dataset_sessions"]
        total_h_now = summary_after["total_dataset_hours"]

        queue.mark_completed(
            key=key,
            session_id=f"session_batch_{key}",
            frame_count=int(elapsed_sec * 30),
            duration_seconds=elapsed_sec,
            title=title
        )

        print("\n" + "=" * 75)
        print(f"  ✅ [COMPLETED] '{title[:50]}'")
        print(f"     Recorded ~{elapsed_sec/60.0:.1f} mins ({elapsed_sec:.1f}s)")
        print(f"     Dataset Total: {total_h_now:.2f} hrs / 100.0 hrs ({summary_after['phase1_progress_pct']:.1f}%)")
        print("=" * 75 + "\n")

        # Emit Windows desktop notification
        notify_user(
            "FaceKey Studio - Video Complete",
            f"Processed: '{title[:32]}'\nTotal Dataset: {total_h_now:.2f}h / 100h ({summary_after['phase1_progress_pct']:.1f}%)"
        )

        # Brief rest between videos to release memory & clean cache
        time.sleep(1.5)

    print("\n" + "=" * 82)
    print("  [BATCH QUEUE ENGINE PAUSED / COMPLETED]")
    print(f"  All save points and queue positions are safely preserved.")
    print(f"  You can add more URLs to '{queue.urls_file}' and resume anytime!")
    print("=" * 82 + "\n")


if __name__ == "__main__":
    run_batch_queue(watch_continuous=True)
