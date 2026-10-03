"""FaceKey Animation Studio - Master Interactive Console Launcher.
Provides an intuitive menu to launch any recording mode, inspect dataset, or peacefully stop.
"""

import os
import sys
import subprocess
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

def print_banner():
    print(r"""
================================================================================
        ⚡ FACEKEY ANIMATION STUDIO - MASTER CONTROL LAUNCHER ⚡
================================================================================
""")

def print_menu():
    print("  [1] 🎯 YouTube / Browser Tab Mode (HD Live Preview + 1-Click Tab Switch)")
    print("      -> Captures background YouTube/Brave tabs at 30 FPS.")
    print("      -> Includes [SWITCH TAB], [PAUSE], [STOP & SAVE], and F9 instant lock.")
    print("      -> Minimize freely with [_] to keep recording peacefully.\n")
    print("  [2] 🎬 Direct Single YouTube URL Mode (480p SD / 720p HD / 1080p FHD)")
    print("      -> Directly streams one YouTube video in 480p/720p with 0 browser tabs.\n")
    print("  [3] 🚀 Batch YouTube Queue Mode (Silent 480p Background Processing + Auto-Resume)")
    print("      -> Ingests 'sessions\\youtubeURLtoProcess.txt'; add links anytime continuously!")
    print("      -> 100% Silent 480p headless execution, auto-resume, & 100-Hour milestone.\n")
    print("  [4] 🖥️  Custom Screen ROI Box Mode")
    print("      -> Drag a visual rectangle over any player or region on your monitor.\n")
    print("  [5] 📱 Phone Camera Stream Mode (Vivo Y21 / DroidCam / IP Webcam)")
    print("      -> Connects to your phone camera over Wi-Fi without heating the phone.\n")
    print("  [6] 🔇 Silent Headless Background Mode (Zero GUI Window, Lowest CPU)")
    print("      -> Records speech & landmarks completely invisibly in background.\n")
    print("  [7] 🧪 4-Phase Diagnostic Simulation (Sample Video & Engine Test)")
    print("      -> Verifies audio gate, lip sync, teeth metrics, and 3D head pose.\n")
    print("  [8] 📊 Dataset Readiness & 100-Hour Training Milestone Inspector")
    print("      -> Scans all sessions, reports progress toward 100 Hours & storage estimation.\n")
    print("  [9] 🛑 Stop Any Running Studio Session Peacefully (Flushes to Disk)\n")
    print("  [10] 🧹 Clean System Temp, Caches & Reclaim Storage (Purge Junk)\n")
    print("  [11] 📋 View Streamed Videos Registry (History & Duplicates Inspector)")
    print("       -> Inspect all recorded YouTube URLs, sessions, titles, and durations.\n")
    print("  [0] ❌ Exit Launcher")
    print("=" * 80)

def main():
    python_exe = sys.executable
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    while True:
        clear_screen()
        print_banner()
        print_menu()
        
        try:
            choice = input("Enter choice [0-11] (or press Enter for [1]): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting FaceKey Launcher.")
            break

        if not choice:
            choice = "1"
        
        if choice == "0":
            print("\nExiting FaceKey Launcher. Goodbye!")
            time.sleep(1)
            break

        elif choice == "1":
            from src.core.capture import WindowCaptureSource
            wins = WindowCaptureSource.list_all_windows()
            eligible = [w for w in wins if not any(ig in w[1].lower() for ig in ("facekey", "drag box", "program manager", "realtek", "nvidia"))]

            print("\n" + "=" * 70)
            print("  CHOOSE APPLICATION OR WINDOW TO CAPTURE:")
            print("=" * 70)
            print("  💡 BROWSER BACKGROUND RECORDING TIP:")
            print("     To record YouTube in the background while browsing other sites:")
            print("     -> DRAG THE YOUTUBE TAB OUT into its own separate window!")
            print("     -> FaceKey will lock onto the YouTube window and record it")
            print("        in the background even while you browse in your main window!")
            print("-" * 70)
            if eligible:
                for idx, (h, t) in enumerate(eligible, 1):
                    tag = "🎬 [YOUTUBE/VIDEO]" if any(k in t.lower() for k in ("youtube", "video", "watch", "drishyam")) else "🌐 [APP/BROWSER]"
                    print(f"  [{idx}] {tag} {t[:55]}")
                print("  [G] 🖥️  Open Graphical Window Picker (Visual Click-to-Select)")
                print("  [0] 🖥️  Full Screen Desktop Mode")
                print("=" * 70)
                w_choice = input(f"Enter choice [1-{len(eligible)} / G / 0 / search word] (or press Enter for [1]): ").strip()
                if w_choice.lower() == "g":
                    subprocess.call([python_exe, os.path.join(script_dir, "src", "ui", "window_picker.py")])
                    flag_file = os.path.join(script_dir, "switch.flag")
                    chosen = ""
                    if os.path.exists(flag_file):
                        try:
                            with open(flag_file, "r", encoding="utf-8") as f:
                                chosen = f.read().strip()
                            os.remove(flag_file)
                        except Exception:
                            pass
                    if chosen == "screen":
                        cmd = [python_exe, "test_face_speaker_tool.py", "--mode", "screen"]
                    elif chosen:
                        cmd = [python_exe, "test_face_speaker_tool.py", "--mode", "window", "--window", chosen]
                    else:
                        cmd = [python_exe, "test_face_speaker_tool.py", "--mode", "window"]
                elif w_choice == "0":
                    cmd = [python_exe, "test_face_speaker_tool.py", "--mode", "screen"]
                else:
                    try:
                        sel_idx = int(w_choice) - 1 if w_choice.isdigit() else -1
                        if 0 <= sel_idx < len(eligible):
                            sel_h, sel_t = eligible[sel_idx]
                            print(f"\n[*] Starting capture on: [{sel_h}] '{sel_t}'...")
                            cmd = [python_exe, "test_face_speaker_tool.py", "--mode", "window", "--window", str(sel_h)]
                        elif w_choice:
                            # Search query text (e.g. 'brave', 'youtube', 'chrome')
                            cmd = [python_exe, "test_face_speaker_tool.py", "--mode", "window", "--window", w_choice]
                        else:
                            sel_h, sel_t = eligible[0]
                            cmd = [python_exe, "test_face_speaker_tool.py", "--mode", "window", "--window", str(sel_h)]
                    except Exception:
                        cmd = [python_exe, "test_face_speaker_tool.py", "--mode", "window"]
            else:
                cmd = [python_exe, "test_face_speaker_tool.py", "--mode", "window"]
            subprocess.call(cmd)
            input("\nPress Enter to return to launcher menu...")

        elif choice == "2" or choice.lower() == "y":
            print("\n" + "=" * 70)
            print("  🎬 DIRECT YOUTUBE STREAM MODE (Zero Browser Needed)")
            print("=" * 70)
            print("  Streams directly in 480p / 720p / 1080p without opening any browser.")
            print("  You can do anything else on your PC without losing frames!")
            print("-" * 70)
            yt_url = input("Enter YouTube Video URL: ").strip()
            if yt_url:
                from src.storage.stream_registry import StreamRegistry
                registry = StreamRegistry(script_dir)
                existing = registry.find_entry(yt_url)
                if existing:
                    print("\n" + "!" * 70)
                    print("  ⚠️  DUPLICATE DETECTED: THIS VIDEO HAS ALREADY BEEN RECORDED!")
                    print("!" * 70)
                    print(f"  Title           : {existing.get('title', 'Unknown')}")
                    print(f"  Canonical URL   : {existing.get('canonical_url', yt_url)}")
                    print(f"  Times Recorded  : {existing.get('recorded_count', 1)} session(s)")
                    print(f"  Total Duration  : {existing.get('total_duration_seconds', 0.0) / 60.0:.2f} mins ({existing.get('total_frames', 0):,} frames)")
                    print(f"  First Recorded  : {existing.get('first_recorded_at', 'Unknown')}")
                    print(f"  Last Recorded   : {existing.get('last_recorded_at', 'Unknown')}")
                    if existing.get('sessions'):
                        print(f"  Saved Sessions  : {', '.join(existing['sessions'][-3:])}")
                    print("-" * 70)
                    print("  💡 Tip: Re-recording the same video accumulates redundant frames.")
                    proceed = input("  Do you want to re-record this video anyway? [y/N]: ").strip().lower()
                    if proceed not in ("y", "yes"):
                        print("\n[*] Re-recording cancelled to protect dataset consistency.")
                        input("\nPress Enter to return to launcher menu...")
                        continue

                print("\n" + "-" * 70)
                print("  CHOOSE VIDEO STREAM RESOLUTION:")
                print("  [1] ⚡ 480p SD  (Low Bandwidth, Lowest CPU & RAM, Zero Lag) [Recommended]")
                print("  [2] 🎬 720p HD  (Balanced HD Clarity)")
                print("  [3] 🌟 1080p FHD (Maximum Resolution)")
                print("-" * 70)
                q_choice = input("Enter resolution [1-3] (or press Enter for [1] 480p): ").strip()
                if q_choice == "2":
                    quality_tag = "720p"
                    ydl_fmt = "best[height<=720]/bestvideo[height<=720]/best"
                elif q_choice == "3":
                    quality_tag = "1080p"
                    ydl_fmt = "best[height<=1080]/bestvideo[height<=1080]/best"
                else:
                    quality_tag = "480p"
                    ydl_fmt = "best[height<=480]/bestvideo[height<=480]/best"

                print("-" * 70)
                print("  CHOOSE DISPLAY / EXECUTION MODE:")
                print("  [1] 🖥️  Live HD Preview Window (Interactive Controls & Motion Overlay)")
                print("  [2] 🔇 Silent Background / Headless Mode (Zero GUI, Lowest CPU, Auto-Notifies on Complete)")
                print("-" * 70)
                bg_choice = input("Enter display mode [1-2] (or press Enter for [1]): ").strip()
                is_headless = (bg_choice == "2")

                print(f"\n[*] Resolving YouTube video stream in {quality_tag} via yt-dlp...")
                try:
                    import yt_dlp
                    ydl_opts = {
                        'format': ydl_fmt,
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
                        info = ydl.extract_info(yt_url, download=False)
                        stream_url = info.get('url')
                        title = info.get('title', 'YouTube Video')
                        res_w = info.get('width', '?')
                        res_h = info.get('height', '?')
                        canonical_url = existing.get('canonical_url', yt_url) if existing else (info.get('webpage_url') or yt_url)
                        print(f"[*] Successfully connected to: '{title}' ({res_w}x{res_h} @ {quality_tag})")
                        cmd = [
                            python_exe, "test_face_speaker_tool.py",
                            "--mode", "stream",
                            "--stream-url", stream_url,
                            "--quality", quality_tag,
                            "--stream-title", title,
                            "--canonical-url", canonical_url
                        ]
                        if is_headless:
                            cmd.append("--headless")
                            print("[*] Running in Silent Background Mode. You will receive a Windows notification when finished.")
                        subprocess.call(cmd)
                except Exception as e:
                    print(f"[!] Error resolving YouTube stream: {e}")
            else:
                print("[!] No URL entered.")
            input("\nPress Enter to return to launcher menu...")

        elif choice == "3" or choice.lower() in ("b", "batch"):
            print("\n" + "=" * 70)
            print("  🚀 LAUNCHING BATCH YOUTUBE QUEUE RUNNER (Silent 480p Background Mode)")
            print("=" * 70)
            print("  • Monitoring: 'sessions\\youtubeURLtoProcess.txt'")
            print("  • You can add/paste links into that file anytime - they auto-load!")
            print("  • Runs completely silently in background with auto-resume support.")
            print("-" * 70)
            subprocess.call([python_exe, "batch_stream_runner.py"])
            input("\nPress Enter to return to launcher menu...")

        elif choice == "4":
            print("\n[*] Starting Screen ROI Mode (Drag box on monitor)...")
            cmd = [python_exe, "test_face_speaker_tool.py", "--mode", "screen"]
            subprocess.call(cmd)
            input("\nPress Enter to return to launcher menu...")

        elif choice == "5":
            print("\n" + "=" * 65)
            print("  PHONE CAMERA STREAM SETUP (Vivo Y21 / DroidCam / IP Webcam)")
            print("=" * 65)
            default_url = "http://192.168.1.100:8080/video"
            entered_url = input(f"Enter Phone Stream URL [{default_url}]: ").strip()
            stream_url = entered_url if entered_url else default_url
            print(f"\n[*] Connecting to Phone Camera: {stream_url}...")
            cmd = [python_exe, "test_face_speaker_tool.py", "--mode", "stream", "--stream-url", stream_url]
            subprocess.call(cmd)
            input("\nPress Enter to return to launcher menu...")

        elif choice == "6":
            print("\n[*] Launching Silent Headless Background Mode...")
            print("    -> Window is hidden to save CPU/GPU resources.")
            print("    -> To stop peacefully anytime, run STOP_STUDIO_PEACEFULLY.bat\n")
            cmd = [python_exe, "test_face_speaker_tool.py", "--mode", "window", "--window", "Brave", "--headless"]
            subprocess.call(cmd)
            input("\nPress Enter to return to launcher menu...")

        elif choice == "7":
            print("\n[*] Starting 4-Phase Diagnostic Simulation Mode...")
            cmd = [python_exe, "test_face_speaker_tool.py", "--mode", "sim"]
            subprocess.call(cmd)
            input("\nPress Enter to return to launcher menu...")

        elif choice == "8" or choice.lower() in ("d", "data"):
            print("\n[*] Scanning recorded dataset sessions on disk...")
            try:
                from src.storage.dataset_tracker import DatasetReadinessTracker
                tracker = DatasetReadinessTracker("sessions")
                report = tracker.compute_cumulative_report()
                tot_hours = report.total_duration_seconds / 3600.0
                speech_hours = report.total_active_speech_sec / 3600.0
                used_gb = report.total_size_bytes / (1024.0 * 1024.0 * 1024.0)

                print("\n" + "=" * 75)
                print("  📊 DATASET ACCUMULATION & 100-HOUR TRAINING MILESTONE REPORT")
                print("=" * 75)
                print(f"  Total Valid Sessions    : {report.total_sessions_count}")
                print(f"  Total Video Frames      : {report.total_samples_frames:,}")
                print(f"  Total Clean Speech Time : {speech_hours:.2f} hrs ({report.total_active_speech_sec / 60.0:.1f} mins)")
                print(f"  Total Conversational    : {tot_hours:.2f} hrs ({report.total_duration_seconds / 60.0:.1f} mins)")
                print(f"  Current Disk Storage    : {used_gb:.2f} GB")
                print("-" * 75)
                print("  🎯 MILESTONES & STORAGE ESTIMATIONS:")
                print(f"  Phase 1 Target          : 100.00 Hours")
                print(f"  Phase 1 Progress        : {tot_hours:.2f} hrs / 100.00 hrs ({report.phase1_progress_pct:.2f}%)")
                print(f"  Hours Remaining to 100h : {max(0.0, 100.0 - tot_hours):.2f} hrs remaining")
                print(f"  Estimated Size for 100h : ~48 GB (~47.9 GB @ 490 MB/hr compressed rate)")
                print(f"  Scale Target (1,000h)   : ~480 GB (~479.1 GB @ 490 MB/hr compressed rate)")
                print(f"  Long-Term 1,000h Scale  : {report.scale_progress_pct:.2f}% complete")
                print("-" * 75)
                print(f"  Status Indicator        : {report.readiness_label}")
                print("=" * 75)
            except Exception as e:
                print(f"Error checking dataset: {e}")
            input("\nPress Enter to return to launcher menu...")

        elif choice == "9":
            print("\n[*] Signaling active FaceKey Studio to stop peacefully...")
            try:
                subprocess.call([python_exe, "stop_studio.py"])
            except Exception as e:
                print(f"Error stopping studio: {e}")
            time.sleep(2)
            input("\nPress Enter to return to launcher menu...")

        elif choice in ("10", "c"):
            print("\n" + "=" * 70)
            print("  🧹 CLEANING SYSTEM TEMP, PIP CACHES & JUNK FILES...")
            print("=" * 70)
            import shutil, glob
            # 1. Purge pip cache
            try:
                subprocess.call([python_exe, "-m", "pip", "cache", "purge"])
            except Exception:
                pass
            # 2. Purge pycache folders
            try:
                for root, dirs, files in os.walk(script_dir):
                    if "__pycache__" in dirs:
                        shutil.rmtree(os.path.join(root, "__pycache__"), ignore_errors=True)
            except Exception:
                pass
            # 3. Clean Windows user temp files safely
            cleaned_mb = 0.0
            temp_dir = os.environ.get("TEMP", "")
            if temp_dir and os.path.exists(temp_dir):
                for f in glob.glob(os.path.join(temp_dir, "*")):
                    try:
                        sz = os.path.getsize(f) if os.path.isfile(f) else 0
                        if os.path.isdir(f):
                            shutil.rmtree(f, ignore_errors=True)
                        else:
                            os.remove(f)
                        cleaned_mb += sz / (1024 * 1024)
                    except Exception:
                        pass
            print(f"[*] Reclaimed space from temporary files!")
            print("[*] Cleanup complete! Storage is clean and optimized.")
            input("\nPress Enter to return to launcher menu...")

        elif choice in ("11", "r"):
            print("\n" + "=" * 75)
            print("  📋 STREAMED VIDEOS & URL REGISTRY (Duplicate Protection & History)")
            print("=" * 75)
            try:
                from src.storage.stream_registry import StreamRegistry
                registry = StreamRegistry(script_dir)
                sources = registry.registry.get("sources", {})
                if not sources:
                    print("  No streamed video sessions recorded yet.")
                    print("  When you record videos in Option [2] or [3], they will appear here automatically.")
                else:
                    total_dur_min = sum(s.get("total_duration_seconds", 0.0) for s in sources.values()) / 60.0
                    total_frames = sum(s.get("total_frames", 0) for s in sources.values())
                    print(f"  Tracked Videos   : {len(sources)}")
                    print(f"  Total Frames     : {total_frames:,}")
                    print(f"  Total Duration   : {total_dur_min:.2f} minutes ({total_dur_min / 60.0:.2f} hours)")
                    print("-" * 75)
                    for idx, (k, s) in enumerate(sources.items(), 1):
                        print(f"  [{idx}] {s.get('title', 'Unknown Title')}")
                        print(f"      Canonical URL: {s.get('canonical_url', 'N/A')}")
                        print(f"      Recorded     : {s.get('recorded_count', 1)}x | {s.get('total_duration_seconds', 0.0)/60.0:.2f} mins ({s.get('total_frames', 0):,} frames)")
                        print(f"      Last Date    : {s.get('last_recorded_at', 'N/A')}")
                        if s.get("sessions"):
                            print(f"      Sessions     : {', '.join(s['sessions'][-3:])}")
                        print("")
                print("-" * 75)
                print("  💡 Full documentation auto-saved at: STREAMED_VIDEOS_REGISTRY.md")
                print("=" * 75)
            except Exception as e:
                print(f"Error inspecting stream registry: {e}")
            input("\nPress Enter to return to launcher menu...")

        else:
            print("\nInvalid choice. Please select 0 to 11.")
            time.sleep(1.5)

if __name__ == "__main__":
    main()
