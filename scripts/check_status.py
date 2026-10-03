"""
FaceKey Live Studio Status Checker
===================================
Instantly checks if FaceKey background recording is running,
what window/tab is being captured, and how much data has been recorded.
"""

import os
import sys
import json
import time

def is_pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        SYNCHRONIZE = 0x00100000
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, False, pid)
        if handle:
            exit_code = ctypes.c_ulong()
            kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            kernel32.CloseHandle(handle)
            # STILL_ACTIVE = 259
            return exit_code.value == 259
    return False

def main():
    print("=" * 72)
    print("             FACEKEY STUDIO -- REAL-TIME STATUS CHECK               ")
    print("=" * 72)

    status_file = "live_status.json"
    if not os.path.exists(status_file):
        print("\n  [STATE] FaceKey Studio is NOT currently running.")
        print("  -> To launch live YouTube capture: double-click RUN_LIVE_YOUTUBE.bat")
        print("  -> To launch background capture  : double-click START_BACKGROUND_RECORDING.bat\n")
        print("=" * 72)
        return

    try:
        with open(status_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"\n  [ERROR] Could not read status file: {e}\n")
        return

    pid = data.get("pid", 0)
    running = data.get("is_running", False) and is_pid_alive(pid)

    if not running:
        print("\n  [STATE] FaceKey Studio is currently STOPPED.")
        print(f"  -> Last recorded state : {data.get('status', 'Idle')}")
        print(f"  -> Last active time    : {data.get('last_updated', 'N/A')}")
        print("\n  Launch commands:")
        print("  - RUN_LIVE_YOUTUBE.bat            (Interactive 1280x720 HD Preview)")
        print("  - START_BACKGROUND_RECORDING.bat  (Silent background capture)\n")
        print("=" * 72)
        return

    mode = data.get("mode", "screen").upper()
    target = data.get("target", "Screen")
    state = data.get("conversational_state", "SILENCE")
    is_rec = data.get("is_recording", False)
    frames = data.get("session_frames", 0)
    speech_s = data.get("speech_seconds", 0.0)
    pause_s = data.get("pause_seconds", 0.0)
    fps = data.get("fps", 0.0)
    latency = data.get("latency_ms", 0.0)
    total_mins = data.get("total_dataset_mins", 0.0)
    readiness = data.get("readiness_label", "")
    faces = data.get("faces_detected", 0)
    updated = data.get("last_updated", "")

    if is_rec and state == "ACTIVE_SPEECH":
        badge = "[*] RECORDING: ACTIVE SYLLABLE SPEECH"
    elif is_rec and state == "CONVERSATIONAL_PAUSE":
        badge = "[*] RECORDING: CONVERSATIONAL PAUSE / IDLE"
    elif faces > 1:
        badge = "[!] PAUSED: MULTIPLE FACES DETECTED (Strict Single-Face Gate)"
    elif faces == 1:
        badge = "[-] WAITING: SILENCE / LISTENER DETECTED (Audio Gate Closed)"
    else:
        badge = "[-] SEARCHING: NO FACE DETECTED IN TARGET WINDOW"

    print(f"\n  PROCESS ID         : {pid} (RUNNING HEALTHY)")
    print(f"  CAPTURE MODE       : {mode}")
    print(f"  CAPTURED TAB/WINDOW: \"{target}\"")
    print(f"  RECORDING STATUS   : {badge}")
    print(f"  FACES IN FRAME     : {faces} face(s) verified")
    print(f"  CURRENT SESSION    : {frames} frames ({speech_s:.1f}s speech, {pause_s:.1f}s natural pause)")
    print(f"  PROCESSING SPEED   : {fps:.1f} FPS (Latency: {latency:.1f}ms)")
    print(f"  GLOBAL DATASET     : {total_mins:.1f} minutes recorded on disk")
    print(f"  AI READINESS       : {readiness}")
    print(f"  LAST HEARTBEAT     : {updated}")
    print("\n  Peaceful Stop Options:")
    print("  - Double-click STOP_STUDIO_PEACEFULLY.bat")
    print("  - Press Shift + Esc anywhere on Windows")
    print("  - Click [X] or [CLOSE] on the preview window\n")
    print("=" * 72)

if __name__ == "__main__":
    main()
