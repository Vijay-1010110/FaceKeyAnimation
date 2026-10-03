"""
Interactive Single-Face Verified-Speaking Gate & Motion Overlay Test Studio
===========================================================================
Displays live facial motion capture data acquisition with:
  - Teeth Biomechanics & Exposure Tracking (Maxillary/Mandibular Teeth, Inter-Dental Gap)
  - Cheek & Facial Muscle Dynamics (Cheek Puff, Cheek Squint, Nose Sneer, Brow Furrow)
  - 3D Face Orientation & Tilting (Euler Pitch/Yaw/Roll + Nose 3D Axis Tripod)
  - Eye Movement & Pupil Gaze Tracking (Iris Ring, Gaze Ray Vectors, Reticle Widget)
  - Atomic Monotonic Clocking (nanosecond-precision AV sync & stream gap detection)
  - Natural Conversational Pause & Idle Capture (records thinking & inter-word pauses)
  - Interactive On-Screen Clickable Buttons:
      [▶ PLAY] / [⏸ PAUSE]  - Freeze pipeline to rest CPU/RAM resources peacefully
      [⏹ STOP & SAVE]       - Finalize & persist recorded session to sessions/ directory
      [✂ CROP ROI]          - Select YouTube video rectangle
      [✕ CLOSE]             - Clean graceful shutdown
  - Global Multi-Session Dataset Tracker & AI Training Readiness Progress
  - 478-point 3D landmark mesh with distinct anatomical contours
"""

import os
import sys

# Silence TensorFlow & MediaPipe C++ informational logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["ABSL_LOG_LEVEL"] = "error"

import ctypes

# Immediately attach process thread to 'Default' desktop (user's real monitor surface)
if os.name == "nt":
    try:
        user32 = ctypes.windll.user32
        hdesk = user32.OpenDesktopW("Default", 0, False, 0x01FF)
        if not hdesk:
            hdesk = user32.OpenInputDesktop(0, False, 0x01FF)
        if hdesk:
            user32.SetThreadDesktop(hdesk)
    except Exception:
        pass

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import time
import math
import uuid
import argparse
import signal
import threading
import json
import cv2
import numpy as np
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.config import AppConfig
from src.core.face_pipeline import FacePipeline
from src.schema import AudioFrameData, FaceRole, EligibilityLevel, TrackedFaceFrame
from src.core.capture import ScreenCaptureSource, AudioCaptureSource, WindowCaptureSource, StreamCaptureSource
from src.storage.dataset_tracker import DatasetReadinessTracker, DatasetReadinessReport
from src.storage.dataset_writer import DatasetWriter
from src.storage.stream_registry import StreamRegistry
from src.utils.notifier import notify_user
def extract_audio_from_file(video_path: str, target_sr: int = 16000) -> Optional[np.ndarray]:
    """Extract audio track from video file as 16kHz mono float32 numpy array.
    Tries ffmpeg first, then PyAV, returns None if video has no audio or tools unavailable.
    """
    if not os.path.exists(video_path):
        return None
    try:
        import subprocess, tempfile
        import scipy.io.wavfile as wavfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            wav_path = tf.name
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-ac", "1", "-ar", str(target_sr),
            "-f", "wav", wav_path
        ]
        ret = subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if ret == 0 and os.path.exists(wav_path) and os.path.getsize(wav_path) > 44:
            sr, data = wavfile.read(wav_path)
            try:
                os.remove(wav_path)
            except Exception:
                pass
            if data.dtype == np.int16:
                return (data.astype(np.float32) / 32768.0)
            return data.astype(np.float32)
        if os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass
    except Exception:
        pass

    try:
        import av
        container = av.open(video_path)
        if container.streams.audio:
            resampler = av.AudioResampler(format='fltp', layout='mono', rate=target_sr)
            chunks = []
            for frame in container.decode(audio=0):
                for resampled in resampler.resample(frame):
                    chunks.append(resampled.to_ndarray().flatten())
            container.close()
            if chunks:
                return np.concatenate(chunks).astype(np.float32)
        container.close()
    except Exception:
        pass
    return None


class MasterAtomicClock:
    """Monotonic atomic nanosecond master clock for sub-millisecond audio-visual synchronization."""

    def __init__(self):
        self.t0_ns = time.perf_counter_ns()

    def now_ns(self) -> int:
        return time.perf_counter_ns() - self.t0_ns

    def now_sec(self) -> float:
        return (time.perf_counter_ns() - self.t0_ns) / 1e9

    def format_clock(self, ts_sec: float) -> str:
        mins = int(ts_sec // 60)
        secs = int(ts_sec % 60)
        millis = int((ts_sec * 1000) % 1000)
        return f"{mins:02d}:{secs:02d}.{millis:03d}"


def ensure_input_desktop():
    """Ensure the calling thread is attached to the active Windows input desktop."""
    if os.name == "nt":
        try:
            user32 = ctypes.windll.user32
            hdesk = user32.OpenInputDesktop(0, False, 0x01FF)
            if hdesk:
                user32.SetThreadDesktop(hdesk)
        except Exception:
            pass


# Landmark groups for 478-point 3D facial motion visualization
LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_TEETH_LIPS = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LOWER_TEETH_LIPS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]


# Interactive On-Screen Button Rectangles in 1280x720 HD display coordinates
# Standard 5-Button Layout for Window / Screen ROI Modes
BTN_PAUSE = {"x": 520,  "y": 672, "w": 130, "h": 38}
BTN_SAVE  = {"x": 660,  "y": 672, "w": 135, "h": 38}
BTN_FOCUS = {"x": 805,  "y": 672, "w": 155, "h": 38}
BTN_CROP  = {"x": 970,  "y": 672, "w": 140, "h": 38}
BTN_CLOSE = {"x": 1120, "y": 672, "w": 145, "h": 38}

# Dedicated Stream Mode 3-Button Layout (Zero browser needed: removes irrelevant Switch Tab & Crop ROI)
BTN_PAUSE_STREAM = {"x": 800,  "y": 672, "w": 140, "h": 38}
BTN_SAVE_STREAM  = {"x": 955,  "y": 672, "w": 150, "h": 38}
BTN_CLOSE_STREAM = {"x": 1120, "y": 672, "w": 145, "h": 38}

# Standard Windows Cursor handles for crisp, responsive mouse feedback
IDC_ARROW = 32512
IDC_HAND  = 32649
h_arrow = ctypes.windll.user32.LoadCursorW(None, IDC_ARROW) if sys.platform == "win32" else None
h_hand  = ctypes.windll.user32.LoadCursorW(None, IDC_HAND) if sys.platform == "win32" else None


def setup_window_position_and_affinity(window_name: str, x: int = 40, y: int = 30, w: int = 1280, h: int = 720) -> Optional[int]:
    """Place OpenCV window, bring to foreground initially, set crisp mouse cursor, allow user to freely minimize/move."""
    if sys.platform == "win32":
        try:
            user32 = ctypes.windll.user32
            hwnd = user32.FindWindowW(None, window_name)
            if hwnd:
                user32.ShowWindow(hwnd, 9)  # SW_RESTORE / SW_SHOWNORMAL
                user32.SetForegroundWindow(hwnd)
                user32.BringWindowToTop(hwnd)
                # Briefly promote to HWND_TOPMOST so DWM draws it in front of full-screen browsers,
                # then immediately release to HWND_NOTOPMOST so user can freely minimize or overlap
                HWND_TOPMOST = ctypes.c_void_p(-1)
                HWND_NOTOPMOST = ctypes.c_void_p(-2)
                SWP_SHOWWINDOW = 0x0040
                SWP_NOMOVE = 0x0002
                SWP_NOSIZE = 0x0001
                user32.SetWindowPos.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint
                ]
                user32.SetWindowPos(hwnd, HWND_TOPMOST, x, y, w, h, SWP_SHOWWINDOW)
                user32.SetWindowPos(hwnd, HWND_NOTOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW)
                # Set window class cursor to clean Windows Arrow cursor (prevent hourglass/spinning wheel)
                if h_arrow:
                    if hasattr(user32, 'SetClassLongPtrW'):
                        user32.SetClassLongPtrW(hwnd, -12, h_arrow)  # GCLP_HCURSOR = -12
                    else:
                        user32.SetClassLongW(hwnd, -12, h_arrow)
                    user32.SetCursor(h_arrow)
                # WDA_EXCLUDEFROMCAPTURE (0x00000011): Prevents screen capture from recording preview window
                user32.SetWindowDisplayAffinity(hwnd, 0x00000011)
                return hwnd
        except Exception:
            pass
    return None


def draw_mini_meter(canvas: np.ndarray, x: int, y: int, w: int, h: int, label: str, value: float, max_val: float = 1.0, color: tuple = (52, 235, 100)):
    """Draw a crisp, high-resolution telemetry progress bar with anti-aliased typography."""
    # Label in crisp slate-white
    cv2.putText(canvas, f"{label}:", (x, y + h - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (241, 245, 249), 1, cv2.LINE_AA)
    bar_x = x + 105
    bar_w = w - 160
    # Bar background
    cv2.rectangle(canvas, (bar_x, y), (bar_x + bar_w, y + h), (20, 26, 38), -1)
    fill_ratio = max(0.0, min(1.0, value / max(1e-4, max_val)))
    fill_w = int(bar_w * fill_ratio)
    if fill_w > 0:
        cv2.rectangle(canvas, (bar_x, y), (bar_x + fill_w, y + h), color, -1)
    cv2.rectangle(canvas, (bar_x, y), (bar_x + bar_w, y + h), (60, 75, 100), 1)
    # Numerical readout
    cv2.putText(canvas, f"{value:4.2f}", (bar_x + bar_w + 8, y + h - 2), cv2.FONT_HERSHEY_DUPLEX, 0.42, color, 1, cv2.LINE_AA)


def draw_head_pose_axes(canvas: np.ndarray, euler: tuple, nose_pt: tuple, axis_len: int = 50):
    """Draw 3D orientation coordinate tripod (Red=Pitch/Nod, Green=Yaw/Turn, Blue=Roll/Normal) on nose tip."""
    pitch = math.radians(euler[0])
    yaw = math.radians(euler[1])
    roll = math.radians(euler[2])
    x0, y0 = nose_pt

    # X-axis (Pitch / Nod - Red)
    x_x = x0 + int(axis_len * (math.cos(yaw) * math.cos(roll)))
    y_x = y0 + int(axis_len * (math.cos(pitch) * math.sin(roll) + math.cos(roll) * math.sin(pitch) * math.sin(yaw)))
    cv2.line(canvas, (x0, y0), (x_x, y_x), (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, "X", (x_x + 4, y_x + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)

    # Y-axis (Yaw / Turn - Green)
    x_y = x0 + int(axis_len * (-math.cos(yaw) * math.sin(roll)))
    y_y = y0 + int(axis_len * (math.cos(pitch) * math.cos(roll) - math.sin(pitch) * math.sin(yaw) * math.sin(roll)))
    cv2.line(canvas, (x0, y0), (x_y, y_y), (0, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, "Y", (x_y + 4, y_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)

    # Z-axis (Forward Normal / Roll - Blue)
    x_z = x0 + int(axis_len * (math.sin(yaw)))
    y_z = y0 - int(axis_len * (math.sin(pitch) * math.cos(yaw)))
    cv2.line(canvas, (x0, y0), (x_z, y_z), (255, 120, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, "Z", (x_z + 4, y_z + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 120, 0), 1, cv2.LINE_AA)

    cv2.circle(canvas, (x0, y0), 4, (255, 255, 255), -1, cv2.LINE_AA)


def draw_eye_gaze_rays(canvas: np.ndarray, lms: np.ndarray, bs: dict, w: int, h: int):
    """Draw pupil iris rings and 3D eye gaze ray vectors projecting from both eyes."""
    if lms is None or len(lms) < 478:
        return 0.0, 0.0, "FORWARD"

    lx, ly = int(lms[468, 0] * w), int(lms[468, 1] * h)
    rx, ry = int(lms[473, 0] * w), int(lms[473, 1] * h)

    cv2.circle(canvas, (lx, ly), 6, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(canvas, (lx, ly), 2, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, (rx, ry), 6, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(canvas, (rx, ry), 2, (0, 0, 255), -1, cv2.LINE_AA)

    gaze_x = ((bs.get("eyeLookOutLeft", 0.0) - bs.get("eyeLookInLeft", 0.0)) +
              (bs.get("eyeLookInRight", 0.0) - bs.get("eyeLookOutRight", 0.0))) / 2.0
    gaze_y = ((bs.get("eyeLookDownLeft", 0.0) - bs.get("eyeLookUpLeft", 0.0)) +
              (bs.get("eyeLookDownRight", 0.0) - bs.get("eyeLookUpRight", 0.0))) / 2.0

    ray_len = 34
    ray_lx = lx + int(gaze_x * ray_len * 2.8)
    ray_ly = ly + int(gaze_y * ray_len * 2.8)
    ray_rx = rx + int(gaze_x * ray_len * 2.8)
    ray_ry = ry + int(gaze_y * ray_len * 2.8)

    cv2.line(canvas, (lx, ly), (ray_lx, ray_ly), (0, 230, 255), 2, cv2.LINE_AA)
    cv2.line(canvas, (rx, ry), (ray_rx, ray_ry), (0, 230, 255), 2, cv2.LINE_AA)
    cv2.circle(canvas, (ray_lx, ray_ly), 3, (0, 230, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, (ray_rx, ray_ry), 3, (0, 230, 255), -1, cv2.LINE_AA)

    if abs(gaze_x) < 0.10 and abs(gaze_y) < 0.10:
        gaze_desc = "FORWARD (CAMERA)"
    elif abs(gaze_x) >= abs(gaze_y):
        gaze_desc = "LOOKING LEFT" if gaze_x < 0 else "LOOKING RIGHT"
    else:
        gaze_desc = "LOOKING UP" if gaze_y < 0 else "LOOKING DOWN"

    return gaze_x, gaze_y, gaze_desc


def draw_sound_graph(
    canvas: np.ndarray,
    gx: int,
    gy: int,
    gw: int,
    gh: int,
    samples: Optional[np.ndarray],
    audio_frame: AudioFrameData
):
    """Draw a live, responsive audio oscilloscope waveform and dynamic VU meter graph."""
    # 1. Background scope box with subtle border
    cv2.rectangle(canvas, (gx, gy), (gx + gw, gy + gh), (12, 16, 24), -1)
    border_col = (52, 235, 100) if audio_frame.is_speech else (45, 55, 75)
    cv2.rectangle(canvas, (gx, gy), (gx + gw, gy + gh), border_col, 1)

    cy = gy + gh // 2
    # Baseline grid
    cv2.line(canvas, (gx + 4, cy), (gx + gw - 46, cy), (30, 40, 55), 1)

    # 2. Oscilloscope Waveform
    wave_w = gw - 52
    if samples is not None and len(samples) > 4:
        indices = np.linspace(0, len(samples) - 1, wave_w).astype(int)
        pts = []
        amp_scale = (gh // 2 - 3) * 3.8
        for i, idx in enumerate(indices):
            val = np.clip(float(samples[idx]) * amp_scale, -(gh // 2 - 4), (gh // 2 - 4))
            pts.append([gx + 6 + i, int(cy - val)])
        col = (52, 235, 100) if audio_frame.is_speech else (100, 130, 150)
        thick = 2 if audio_frame.is_speech else 1
        cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, col, thick, cv2.LINE_AA)
    else:
        # Subtle idle baseline vibration
        t_now = time.perf_counter() * 12.0
        pts = []
        amp = 4.0 if audio_frame.is_speech else 1.0
        for i in range(wave_w):
            val = math.sin(t_now + i * 0.15) * amp
            pts.append([gx + 6 + i, int(cy - val)])
        col = (52, 235, 100) if audio_frame.is_speech else (70, 90, 110)
        cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, col, 1, cv2.LINE_AA)

    # 3. Peak VU Level Meter Bars on the right
    vu_x = gx + gw - 40
    vu_w = 32
    vu_h = gh - 8
    vu_y = gy + 4
    cv2.rectangle(canvas, (vu_x, vu_y), (vu_x + vu_w, vu_y + vu_h), (8, 12, 18), -1)
    cv2.rectangle(canvas, (vu_x, vu_y), (vu_x + vu_w, vu_y + vu_h), (40, 50, 70), 1)

    rms = max(0.0, min(1.0, audio_frame.energy_rms * 4.5))
    num_segs = 6
    seg_h = (vu_h - 4) // num_segs
    for s in range(num_segs):
        threshold = (s + 1) / float(num_segs)
        sy = vu_y + vu_h - (s + 1) * seg_h
        if s >= num_segs - 1:
            col = (68, 68, 245)      # Top segment: Red (Peak)
        elif s >= num_segs - 2:
            col = (250, 204, 21)     # Mid segment: Yellow
        else:
            col = (52, 235, 100)     # Normal: Green

        if rms >= threshold - (1.0 / (num_segs * 2)):
            cv2.rectangle(canvas, (vu_x + 3, sy), (vu_x + vu_w - 3, sy + seg_h - 2), col, -1)
        else:
            cv2.rectangle(canvas, (vu_x + 3, sy), (vu_x + vu_w - 3, sy + seg_h - 2), (20, 26, 36), -1)

    # 4. Sound Graph Micro-Legend
    lbl = "SOUND OSCILLOSCOPE [ACTIVE]" if audio_frame.is_speech else "SOUND OSCILLOSCOPE [SILENCE]"
    lbl_col = (52, 235, 100) if audio_frame.is_speech else (148, 163, 184)
    cv2.putText(canvas, lbl, (gx + 8, gy + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.32, lbl_col, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"RMS: {audio_frame.energy_rms:.3f}", (gx + gw - 110, gy + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 210, 220), 1, cv2.LINE_AA)


def draw_studio_hud(
    frame_bgr: np.ndarray,
    tracked_faces: list,
    audio_frame: AudioFrameData,
    mode_label: str,
    fps: float,
    frame_idx: int,
    recorded_frames_count: int,
    session_elapsed: float,
    session_speech_sec: float,
    session_pause_sec: float,
    is_gate_active: bool,
    gate_status: str,
    speaker_engine: any,
    atomic_clock: MasterAtomicClock,
    dataset_report: DatasetReadinessReport,
    latency_ms: float,
    is_stream_broken: bool,
    dt_ms: float,
    is_paused: bool,
    save_status_msg: str,
    audio_samples: Optional[np.ndarray] = None,
    session_inter_word_sec: float = 0.0
) -> np.ndarray:
    """Render 478-point motion tracking mesh + 3D Head Orientation + Teeth & Muscle Dynamics HUD
    directly on a native 1280x720 HD canvas with crystal-clear, anti-aliased typography.
    """
    TARGET_W = 1280
    TARGET_H = 720

    # 1. Scale background video frame directly to 1280x720 canvas
    orig_h, orig_w = frame_bgr.shape[:2]
    canvas = cv2.resize(frame_bgr, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
    w, h = TARGET_W, TARGET_H
    sx = TARGET_W / max(1, orig_w)
    sy = TARGET_H / max(1, orig_h)
    num_faces = len(tracked_faces)

    conv_state = getattr(speaker_engine, "conversational_state", "SILENCE")

    # Determine visual color theme based on Single-Face Gate state
    if is_paused:
        border_color = (250, 204, 21)      # Gold / Amber (RESTING)
        gate_badge_bg = (35, 30, 10)
        gate_badge_border = (250, 204, 21)
        collecting_title = "[⏸ CPU/RAM AT REST - SYSTEM PAUSED]"
    elif is_gate_active and conv_state == "ACTIVE_SPEECH":
        border_color = (64, 245, 96)       # Bright Neon Green (ACTIVE SPEECH RECORDING)
        gate_badge_bg = (20, 85, 38)
        gate_badge_border = (64, 245, 96)
        collecting_title = "[● REC: ACTIVE SYLLABLE SPEECH]"
    elif is_gate_active and conv_state == "BETWEEN_WORDS":
        border_color = (80, 240, 180)      # Spring Mint Green (BETWEEN WORDS / COARTICULATION)
        gate_badge_bg = (18, 75, 55)
        gate_badge_border = (80, 240, 180)
        collecting_title = "[● REC: BETWEEN WORDS / COARTICULATION]"
    elif is_gate_active and conv_state == "CONVERSATIONAL_PAUSE":
        border_color = (245, 215, 66)      # Cyan / Bright Teal (CONVERSATIONAL PAUSE RECORDING)
        gate_badge_bg = (20, 75, 80)
        gate_badge_border = (245, 215, 66)
        collecting_title = "[● REC: CONVERSATIONAL PAUSE / IDLE]"
    elif num_faces > 1:
        border_color = (68, 68, 245)       # Bright Red (PAUSED: MULTIPLE FACES)
        gate_badge_bg = (25, 25, 110)
        gate_badge_border = (68, 68, 245)
        collecting_title = "[⛔ PAUSED: MULTIPLE FACES DETECTED]"
    elif num_faces == 1:
        border_color = (16, 185, 245)      # Amber/Orange (PAUSED: SILENT / NARRATION)
        gate_badge_bg = (18, 70, 105)
        gate_badge_border = (16, 185, 245)
        collecting_title = "[⏸ PAUSED: PROLONGED SILENCE / NARRATION]"
    else:
        border_color = (148, 163, 184)     # Slate Grey (STANDBY: 0 FACES)
        gate_badge_bg = (40, 45, 55)
        gate_badge_border = (148, 163, 184)
        collecting_title = "[STANDBY: SEARCHING FOR FACE ON SCREEN]"

    # 1. Outer Screen Gate Status Border
    border_thickness = 6 if (is_gate_active and not is_paused and frame_idx % 12 < 8) else 3
    cv2.rectangle(canvas, (2, 2), (w - 3, h - 3), border_color, border_thickness)

    # 2. Top Header Bar (Height = 78px)
    header_h = 78
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), (10, 15, 26), -1)
    cv2.addWeighted(overlay, 0.90, canvas, 0.10, 0, canvas)
    cv2.line(canvas, (0, header_h), (w, header_h), border_color, 2)

    # Heartbeat Pulse Dot
    pulse_on = (frame_idx % 16) < 10
    dot_color = border_color if pulse_on else (35, 95, 45)
    cv2.circle(canvas, (24, 25), 8, dot_color, -1, cv2.LINE_AA)
    spinner_chars = ["|", "/", "-", "\\"]
    spin = spinner_chars[(frame_idx // 2) % 4]

    # Master Atomic Clock & Stream Continuity
    clock_str = atomic_clock.format_clock(session_elapsed)
    continuity_str = "GAP DETECTED!" if is_stream_broken else "STREAM CLEAN"

    cv2.putText(
        canvas,
        f"ATOMIC CLOCK [{spin}] {clock_str}  |  FPS: {fps:4.1f} (dt: {dt_ms:4.1f}ms | {continuity_str})",
        (44, 28),
        cv2.FONT_HERSHEY_DUPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    # Sub-Banner (Row 2 of Header)
    rec_tag_color = (64, 245, 96) if (is_gate_active and conv_state == "ACTIVE_SPEECH") else border_color
    banner_text = save_status_msg if save_status_msg else f"{collecting_title}   |   {gate_status}"
    cv2.putText(
        canvas,
        banner_text,
        (16, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        rec_tag_color if not save_status_msg else (52, 235, 100),
        2,
        cv2.LINE_AA
    )

    # Current Session Acquisition Box (Top Right)
    rec_box_w = 410
    rec_box_x = w - rec_box_w - 14
    cv2.rectangle(canvas, (rec_box_x, 8), (w - 14, 68), gate_badge_bg, -1)
    cv2.rectangle(canvas, (rec_box_x, 8), (w - 14, 68), gate_badge_border, 2)
    session_mb = (recorded_frames_count * 15.5) / 1024.0

    # Prominent Flashing RED REC Dot when actively recording frames to disk
    if is_gate_active and not is_paused:
        rec_dot_on = (frame_idx % 16) < 10
        dot_col = (0, 0, 255) if rec_dot_on else (40, 40, 160)
        cv2.circle(canvas, (rec_box_x + 18, 30), 6, dot_col, -1, cv2.LINE_AA)
        cv2.putText(canvas, "REC", (rec_box_x + 28, 35), cv2.FONT_HERSHEY_DUPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)
        text_x_offset = 64
    else:
        text_x_offset = 12

    cv2.putText(
        canvas,
        f"THIS SESSION: #{recorded_frames_count:05d} ({session_mb:4.1f} MB)",
        (rec_box_x + text_x_offset, 34),
        cv2.FONT_HERSHEY_DUPLEX,
        0.50,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )
    total_rec_sec = session_speech_sec + session_inter_word_sec + session_pause_sec
    cv2.putText(
        canvas,
        f"Words: {session_speech_sec:3.1f}s | Word-Gap: {session_inter_word_sec:3.1f}s | Idle: {session_pause_sec:3.1f}s",
        (rec_box_x + 8, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (226, 232, 240),
        1,
        cv2.LINE_AA
    )

    gaze_x, gaze_y, gaze_desc = 0.0, 0.0, "FORWARD"
    upper_teeth_score = 0.0
    lower_teeth_score = 0.0
    inter_dental_gap = 0.0
    dental_class = "TEETH OCCLUDED"

    # 3. Draw 478 3D Landmark Mesh, Head Pose 3D Axes, and Teeth Outlines
    for f in tracked_faces:
        x1, y1, x2, y2 = f.bbox
        bx1, by1, bx2, by2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
        role = f.role
        lms = f.clean.landmarks
        bs = f.clean.blendshapes
        euler = f.clean.head_pose_euler

        # Teeth metrics computation
        upper_teeth_score = min(1.0, max(bs.get("mouthUpperUpLeft", 0.0), bs.get("mouthUpperUpRight", 0.0)) * 1.5 +
                                max(bs.get("mouthSmileLeft", 0.0), bs.get("mouthSmileRight", 0.0)) * 0.4)
        lower_teeth_score = min(1.0, max(bs.get("mouthLowerDownLeft", 0.0), bs.get("mouthLowerDownRight", 0.0)) * 1.6 +
                                bs.get("jawOpen", 0.0) * 0.3)
        inter_dental_gap = min(1.0, bs.get("jawOpen", 0.0) * 1.8)

        if bs.get("jawOpen", 0.0) < 0.04 and max(upper_teeth_score, lower_teeth_score) < 0.12:
            dental_class = "TEETH OCCLUDED /P,B,M/"
        elif upper_teeth_score > 0.32 and bs.get("jawOpen", 0.0) < 0.20 and lower_teeth_score < 0.25:
            dental_class = "LABIODENTAL /F,V/"
        elif bs.get("jawOpen", 0.0) < 0.22 and (upper_teeth_score > 0.22 or lower_teeth_score > 0.22):
            dental_class = "DENTAL /S,Z,TH/"
        elif bs.get("jawOpen", 0.0) >= 0.22:
            dental_class = "OPEN VOWEL /AA,AH/"
        else:
            dental_class = "SMILE TEETH SHOW"

        if is_gate_active and role == FaceRole.SPEAKER:
            f_color = (52, 235, 100)  # Active Speech (Green)
            role_tag = "SPEAKER [ACTIVE SPEECH]"
        elif is_gate_active and role == FaceRole.SPEAKER_PAUSE:
            f_color = (245, 215, 66)  # Conversational Pause / Idle (Cyan/Teal)
            role_tag = "SPEAKER [CONVERSATIONAL PAUSE]"
        elif num_faces > 1:
            f_color = (68, 68, 245)   # Multi-face Rejected (Red)
            role_tag = "REJECTED (MULTI-FACE)"
        else:
            f_color = (16, 185, 245)  # Silent / Narration (Amber)
            role_tag = "SILENT / LISTENER"

        # Stylized Bounding Box
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), f_color, 2)
        corner_len = min(24, (bx2 - bx1) // 4, (by2 - by1) // 4)
        for cx, cy, dx, dy in [
            (bx1, by1, 1, 1), (bx2, by1, -1, 1),
            (bx1, by2, 1, -1), (bx2, by2, -1, -1)
        ]:
            cv2.line(canvas, (cx, cy), (cx + dx * corner_len, cy), f_color, 3, cv2.LINE_AA)
            cv2.line(canvas, (cx, cy), (cx, cy + dy * corner_len), f_color, 3, cv2.LINE_AA)

        # Draw 478 3D Landmarks
        if lms is not None and len(lms) >= 468:
            for idx in range(len(lms)):
                px = int(lms[idx, 0] * w)
                py = int(lms[idx, 1] * h)
                cv2.circle(canvas, (px, py), 1, (100, 210, 120), -1, cv2.LINE_AA)

            def draw_group(indices, pt_color, radius=2, closed=False):
                pts = []
                for i in indices:
                    if i < len(lms):
                        px = int(lms[i, 0] * w)
                        py = int(lms[i, 1] * h)
                        pts.append((px, py))
                        cv2.circle(canvas, (px, py), radius, pt_color, -1, cv2.LINE_AA)
                if len(pts) > 1:
                    cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], closed, pt_color, 1, cv2.LINE_AA)

            draw_group(FACE_OVAL, (80, 195, 110), radius=2, closed=True)
            draw_group(LEFT_EYEBROW, (255, 220, 0), radius=2, closed=False)
            draw_group(RIGHT_EYEBROW, (255, 220, 0), radius=2, closed=False)
            draw_group(LEFT_EYE, (0, 215, 255), radius=2, closed=True)
            draw_group(RIGHT_EYE, (0, 215, 255), radius=2, closed=True)

            # Draw Upper & Lower Teeth Highlight Contours
            upper_teeth_color = (255, 255, 255) if upper_teeth_score > 0.2 else (180, 180, 195)
            lower_teeth_color = (255, 255, 255) if lower_teeth_score > 0.2 else (180, 180, 195)
            draw_group(UPPER_TEETH_LIPS, upper_teeth_color, radius=2, closed=False)
            draw_group(LOWER_TEETH_LIPS, lower_teeth_color, radius=2, closed=False)

            # Draw Inter-Dental Gap clearance line (between inner upper incisor 13 and inner lower incisor 14)
            p13 = (int(lms[13, 0] * w), int(lms[13, 1] * h))
            p14 = (int(lms[14, 0] * w), int(lms[14, 1] * h))
            if inter_dental_gap > 0.05:
                cv2.line(canvas, p13, p14, (0, 255, 255), 2, cv2.LINE_AA)

            # Cheek muscle highlights (Landmark 117=Left Cheek, Landmark 346=Right Cheek)
            c_squint = max(bs.get("cheekSquintLeft", 0.0), bs.get("cheekSquintRight", 0.0))
            if c_squint > 0.15:
                cv2.circle(canvas, (int(lms[117, 0] * w), int(lms[117, 1] * h)), 8, (255, 180, 100), 2, cv2.LINE_AA)
                cv2.circle(canvas, (int(lms[346, 0] * w), int(lms[346, 1] * h)), 8, (255, 180, 100), 2, cv2.LINE_AA)

            # Draw Pupil Rings and Gaze Rays
            gaze_x, gaze_y, gaze_desc = draw_eye_gaze_rays(canvas, lms, bs, w, h)

            # Draw 3D Head Orientation Coordinate Tripod on Nose Tip (Landmark 1)
            nose_px = int(lms[1, 0] * w)
            nose_py = int(lms[1, 1] * h)
            draw_head_pose_axes(canvas, euler, (nose_px, nose_py), axis_len=max(42, int((bx2 - bx1) * 0.28)))

        # Face Telemetry Badge Above BBox
        jaw_open = bs.get("jawOpen", 0.0)
        badge_y2 = max(header_h + 4, by1 - 4)
        badge_y1 = max(header_h, badge_y2 - 38)
        badge_w = min(w - 10, bx1 + 380)
        cv2.rectangle(canvas, (bx1, badge_y1), (badge_w, badge_y2), (14, 18, 26), -1)
        cv2.rectangle(canvas, (bx1, badge_y1), (badge_w, badge_y2), f_color, 1)
        cv2.putText(canvas, f"FACE #{f.face_id} | {role_tag}", (bx1 + 8, badge_y1 + 16),
                    cv2.FONT_HERSHEY_DUPLEX, 0.46, f_color, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Conf: {f.speaker_probability * 100:.0f}% | Jaw: {jaw_open:.2f} | Teeth: {dental_class}", (bx1 + 8, badge_y1 + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (226, 232, 240), 1, cv2.LINE_AA)

    # 4. Left HUD Panel: 3D Head Orientation, Teeth & Muscle Gauges (330px wide, 570px high)
    if tracked_faces:
        primary_face = tracked_faces[0]
        bs = primary_face.clean.blendshapes
        euler = primary_face.clean.head_pose_euler
        pitch, yaw, roll = euler[0], euler[1], euler[2]

        card_w = 330
        card_h = 570
        card_x = 14
        card_y = header_h + 8

        card_overlay = canvas.copy()
        cv2.rectangle(card_overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (10, 15, 28), -1)
        cv2.addWeighted(card_overlay, 0.88, canvas, 0.12, 0, canvas)
        cv2.rectangle(canvas, (card_x, card_y), (card_x + card_w, card_y + card_h), (51, 65, 85), 1)

        # Card Title
        cv2.rectangle(canvas, (card_x, card_y), (card_x + card_w, card_y + 30), (20, 28, 46), -1)
        cv2.putText(canvas, "[ TEETH, MUSCLES & 3D POSE ]", (card_x + 12, card_y + 21),
                    cv2.FONT_HERSHEY_DUPLEX, 0.48, (226, 232, 240), 1, cv2.LINE_AA)

        # 3D Head Orientation & Face Tilting Readout
        tilt_side = "R-Tilt" if roll > 2.0 else ("L-Tilt" if roll < -2.0 else "Level")
        turn_side = "Turn-R" if yaw > 3.0 else ("Turn-L" if yaw < -3.0 else "Ahead")
        cv2.putText(canvas, f"Pose: P:{pitch:+4.1f} | Y:{yaw:+4.1f} ({turn_side}) | R:{roll:+4.1f} ({tilt_side})", (card_x + 10, card_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (56, 189, 248), 1, cv2.LINE_AA)

        # Teeth & Jaw Biomechanics Section
        t_sep_y = card_y + 60
        cv2.line(canvas, (card_x + 8, t_sep_y), (card_x + card_w - 8, t_sep_y), (51, 65, 85), 1)
        cv2.putText(canvas, "[ DENTAL STATE ]", (card_x + 10, t_sep_y + 18),
                    cv2.FONT_HERSHEY_DUPLEX, 0.44, (148, 163, 184), 1, cv2.LINE_AA)

        # Dental State Badge
        cv2.rectangle(canvas, (card_x + 10, t_sep_y + 24), (card_x + card_w - 10, t_sep_y + 54), (20, 32, 48), -1)
        cv2.rectangle(canvas, (card_x + 10, t_sep_y + 24), (card_x + card_w - 10, t_sep_y + 54), (250, 204, 21), 1)
        cv2.putText(canvas, dental_class, (card_x + 16, t_sep_y + 44),
                    cv2.FONT_HERSHEY_DUPLEX, 0.44, (250, 204, 21), 1, cv2.LINE_AA)

        meter_y = t_sep_y + 64
        meter_step = 26
        draw_mini_meter(canvas, card_x + 10, meter_y, card_w - 20, 14, "Upper Teeth", upper_teeth_score, 0.8, (255, 255, 255))
        draw_mini_meter(canvas, card_x + 10, meter_y + meter_step, card_w - 20, 14, "Lower Teeth", lower_teeth_score, 0.8, (226, 232, 240))
        draw_mini_meter(canvas, card_x + 10, meter_y + meter_step * 2, card_w - 20, 14, "Dental Gap", inter_dental_gap, 0.8, (0, 255, 255))

        # Cheek & Other Facial Muscles Section
        m_sep_y = meter_y + meter_step * 3 + 12
        cv2.line(canvas, (card_x + 8, m_sep_y), (card_x + card_w - 8, m_sep_y), (51, 65, 85), 1)
        cv2.putText(canvas, "[ CHEEKS & FACIAL MUSCLES ]", (card_x + 10, m_sep_y + 18),
                    cv2.FONT_HERSHEY_DUPLEX, 0.44, (148, 163, 184), 1, cv2.LINE_AA)

        c_squint = max(bs.get("cheekSquintLeft", 0.0), bs.get("cheekSquintRight", 0.0))
        c_puff = bs.get("cheekPuff", 0.0)
        n_sneer = max(bs.get("noseSneerLeft", 0.0), bs.get("noseSneerRight", 0.0))
        b_furrow = max(bs.get("browDownLeft", 0.0), bs.get("browDownRight", 0.0))
        m_gauge_y = m_sep_y + 26
        draw_mini_meter(canvas, card_x + 10, m_gauge_y, card_w - 20, 14, "CheekSquint", c_squint, 0.6, (255, 180, 100))
        draw_mini_meter(canvas, card_x + 10, m_gauge_y + meter_step, card_w - 20, 14, "Cheek Puff", c_puff, 0.5, (168, 85, 247))
        draw_mini_meter(canvas, card_x + 10, m_gauge_y + meter_step * 2, card_w - 20, 14, "Nose Sneer", n_sneer, 0.5, (244, 114, 182))
        draw_mini_meter(canvas, card_x + 10, m_gauge_y + meter_step * 3, card_w - 20, 14, "Brow Furrow", b_furrow, 0.6, (239, 68, 68))
        draw_mini_meter(canvas, card_x + 10, m_gauge_y + meter_step * 4, card_w - 20, 14, "Jaw Open", bs.get("jawOpen", 0.0), 0.6, (52, 235, 100))

        # Eye Gaze Mini Reticle Section
        e_sep_y = m_gauge_y + meter_step * 5 + 8
        cv2.line(canvas, (card_x + 8, e_sep_y), (card_x + card_w - 8, e_sep_y), (51, 65, 85), 1)
        cv2.putText(canvas, f"Pupil Gaze: {gaze_desc}", (card_x + 10, e_sep_y + 18),
                    cv2.FONT_HERSHEY_DUPLEX, 0.44, (0, 230, 255), 1, cv2.LINE_AA)

        ret_box_x = card_x + 12
        ret_box_y = e_sep_y + 24
        ret_box_w = 64
        ret_box_h = 36
        cv2.rectangle(canvas, (ret_box_x, ret_box_y), (ret_box_x + ret_box_w, ret_box_y + ret_box_h), (20, 28, 42), -1)
        cv2.rectangle(canvas, (ret_box_x, ret_box_y), (ret_box_x + ret_box_w, ret_box_y + ret_box_h), (71, 85, 105), 1)
        rcx = ret_box_x + ret_box_w // 2
        rcy = ret_box_y + ret_box_h // 2
        cv2.line(canvas, (ret_box_x + 4, rcy), (ret_box_x + ret_box_w - 4, rcy), (55, 68, 88), 1)
        cv2.line(canvas, (rcx, ret_box_y + 4), (rcx, ret_box_y + ret_box_h - 4), (55, 68, 88), 1)
        p_dot_x = int(rcx + max(-24, min(24, gaze_x * 40)))
        p_dot_y = int(rcy + max(-14, min(14, gaze_y * 22)))
        cv2.circle(canvas, (p_dot_x, p_dot_y), 4, (0, 230, 255), -1, cv2.LINE_AA)
        cv2.putText(canvas, f"Gx: {gaze_x:+4.2f}\nGy: {gaze_y:+4.2f}", (ret_box_x + ret_box_w + 14, ret_box_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (226, 232, 240), 1, cv2.LINE_AA)

    # 5. Right HUD Panel: Global Multi-Session Dataset Tracker & AI Training Readiness (330px wide, 570px high)
    rcard_w = 330
    rcard_h = 570
    rcard_x = w - rcard_w - 14
    rcard_y = header_h + 8

    rcard_overlay = canvas.copy()
    cv2.rectangle(rcard_overlay, (rcard_x, rcard_y), (rcard_x + rcard_w, rcard_y + rcard_h), (10, 15, 28), -1)
    cv2.addWeighted(rcard_overlay, 0.88, canvas, 0.12, 0, canvas)
    cv2.rectangle(canvas, (rcard_x, rcard_y), (rcard_x + rcard_w, rcard_y + rcard_h), (51, 65, 85), 1)

    # Card Title
    cv2.rectangle(canvas, (rcard_x, rcard_y), (rcard_x + rcard_w, rcard_y + 30), (20, 28, 46), -1)
    cv2.putText(canvas, "[ GLOBAL DATASET TRACKER ]", (rcard_x + 12, rcard_y + 21),
                cv2.FONT_HERSHEY_DUPLEX, 0.48, (226, 232, 240), 1, cv2.LINE_AA)

    tot_mins = dataset_report.total_duration_seconds / 60.0
    tot_mb = dataset_report.total_size_bytes / (1024.0 * 1024.0)

    cv2.putText(canvas, f"Sessions on Disk : {dataset_report.total_sessions_count} sessions", (rcard_x + 12, rcard_y + 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (226, 232, 240), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Total Samples    : {dataset_report.total_samples_frames:,} frames", (rcard_x + 12, rcard_y + 86),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (52, 235, 100), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Total Data Time  : {tot_mins:4.1f} mins ({tot_mb:4.1f} MB)", (rcard_x + 12, rcard_y + 114),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (56, 189, 248), 1, cv2.LINE_AA)

    tot_speech_mins = dataset_report.total_active_speech_sec / 60.0
    tot_pause_mins = dataset_report.total_pause_sec / 60.0
    cv2.putText(canvas, f"  -> Active Speech: {tot_speech_mins:4.1f} mins", (rcard_x + 12, rcard_y + 142),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (203, 213, 225), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"  -> Natural Idle : {tot_pause_mins:4.1f} mins", (rcard_x + 12, rcard_y + 168),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 215, 66), 1, cv2.LINE_AA)

    # AI Training Readiness Progress
    r_sep_y = rcard_y + 196
    cv2.line(canvas, (rcard_x + 8, r_sep_y), (rcard_x + rcard_w - 8, r_sep_y), (51, 65, 85), 1)
    cv2.putText(canvas, "[ AI TRAINING READINESS ]", (rcard_x + 12, r_sep_y + 24),
                cv2.FONT_HERSHEY_DUPLEX, 0.48, (226, 232, 240), 1, cv2.LINE_AA)

    m1_pct = dataset_report.baseline_progress_pct
    cv2.putText(canvas, f"Baseline Target: 30 mins ({m1_pct:4.1f}%)", (rcard_x + 12, r_sep_y + 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (52, 235, 100) if dataset_report.is_baseline_ready else (226, 232, 240), 1, cv2.LINE_AA)

    m1_bar_w = rcard_w - 24
    cv2.rectangle(canvas, (rcard_x + 12, r_sep_y + 64), (rcard_x + 12 + m1_bar_w, r_sep_y + 78), (20, 26, 38), -1)
    m1_fill = int(m1_bar_w * (m1_pct / 100.0))
    if m1_fill > 0:
        cv2.rectangle(canvas, (rcard_x + 12, r_sep_y + 64), (rcard_x + 12 + m1_fill, r_sep_y + 78), (52, 235, 100), -1)
    cv2.rectangle(canvas, (rcard_x + 12, r_sep_y + 64), (rcard_x + 12 + m1_bar_w, r_sep_y + 78), (60, 75, 100), 1)

    m2_pct = dataset_report.hifi_progress_pct
    cv2.putText(canvas, f"Hi-Fi Generative: 120 mins ({m2_pct:4.1f}%)", (rcard_x + 12, r_sep_y + 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (226, 232, 240), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (rcard_x + 12, r_sep_y + 118), (rcard_x + 12 + m1_bar_w, r_sep_y + 132), (20, 26, 38), -1)
    m2_fill = int(m1_bar_w * (m2_pct / 100.0))
    if m2_fill > 0:
        cv2.rectangle(canvas, (rcard_x + 12, r_sep_y + 118), (rcard_x + 12 + m2_fill, r_sep_y + 132), (56, 189, 248), -1)
    cv2.rectangle(canvas, (rcard_x + 12, r_sep_y + 118), (rcard_x + 12 + m1_bar_w, r_sep_y + 132), (60, 75, 100), 1)

    # Readiness status box
    cv2.rectangle(canvas, (rcard_x + 12, r_sep_y + 154), (rcard_x + rcard_w - 12, r_sep_y + 190), (20, 28, 42), -1)
    cv2.rectangle(canvas, (rcard_x + 12, r_sep_y + 154), (rcard_x + rcard_w - 12, r_sep_y + 190), (250, 204, 21), 1)
    cv2.putText(canvas, dataset_report.readiness_label, (rcard_x + 18, r_sep_y + 178),
                cv2.FONT_HERSHEY_DUPLEX, 0.44, (250, 204, 21), 1, cv2.LINE_AA)

    # 6. Bottom Controls & Audio Telemetry Bar (Height = 55px)
    bot_bar_h = 55
    bot_overlay = canvas.copy()
    cv2.rectangle(bot_overlay, (0, h - bot_bar_h), (w, h), (10, 15, 26), -1)
    cv2.addWeighted(bot_overlay, 0.90, canvas, 0.10, 0, canvas)
    cv2.line(canvas, (0, h - bot_bar_h), (w, h - bot_bar_h), border_color, 1)

    vad_state = "SPEECH DETECTED" if audio_frame.is_speech else "AUDIO SILENCE"
    vad_col = (52, 235, 100) if audio_frame.is_speech else (148, 163, 184)
    is_stream = mode_label.startswith("STREAM")

    if is_stream:
        # Stream Mode: Wide 545px Sound Oscilloscope Graph + 3 Action Buttons on Right
        cv2.putText(canvas, f"AUDIO: {vad_state}", (14, h - 33), cv2.FONT_HERSHEY_DUPLEX, 0.44, vad_col, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"RMS: {audio_frame.energy_rms:.3f} | Lat: {latency_ms:.0f}ms", (14, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (203, 213, 225), 1, cv2.LINE_AA)
        draw_sound_graph(canvas, gx=235, gy=h - bot_bar_h + 8, gw=545, gh=39, samples=audio_samples, audio_frame=audio_frame)
    else:
        # Window / Screen Mode: Compact Audio readout + 305px Sound Graph + 5 Action Buttons
        cv2.putText(canvas, f"AUDIO: {vad_state}", (14, h - 33), cv2.FONT_HERSHEY_DUPLEX, 0.40, vad_col, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"RMS: {audio_frame.energy_rms:.3f}", (14, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (203, 213, 225), 1, cv2.LINE_AA)
        draw_sound_graph(canvas, gx=205, gy=h - bot_bar_h + 8, gw=305, gh=39, samples=audio_samples, audio_frame=audio_frame)

    return canvas


def draw_buttons_on_display(display_bgr: np.ndarray, is_paused: bool, mode: str = "window"):
    """Draw stylish, mouse-clickable button controls on the bottom right of the 1280x720 display window."""
    pause_text = "PLAY" if is_paused else "PAUSE"
    pause_col = (52, 235, 100) if is_paused else (245, 158, 11)

    if mode == "stream":
        # Video Streaming Mode: Only stream-relevant controls (removes irrelevant Switch Tab & Crop ROI)
        # Button 1: Play / Pause
        cv2.rectangle(display_bgr, (BTN_PAUSE_STREAM["x"], BTN_PAUSE_STREAM["y"]),
                      (BTN_PAUSE_STREAM["x"] + BTN_PAUSE_STREAM["w"], BTN_PAUSE_STREAM["y"] + BTN_PAUSE_STREAM["h"]), (25, 32, 45), -1)
        cv2.rectangle(display_bgr, (BTN_PAUSE_STREAM["x"], BTN_PAUSE_STREAM["y"]),
                      (BTN_PAUSE_STREAM["x"] + BTN_PAUSE_STREAM["w"], BTN_PAUSE_STREAM["y"] + BTN_PAUSE_STREAM["h"]), pause_col, 2)
        cv2.putText(display_bgr, pause_text, (BTN_PAUSE_STREAM["x"] + 38, BTN_PAUSE_STREAM["y"] + 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.46, pause_col, 1, cv2.LINE_AA)

        # Button 2: Stop & Save
        cv2.rectangle(display_bgr, (BTN_SAVE_STREAM["x"], BTN_SAVE_STREAM["y"]),
                      (BTN_SAVE_STREAM["x"] + BTN_SAVE_STREAM["w"], BTN_SAVE_STREAM["y"] + BTN_SAVE_STREAM["h"]), (35, 20, 45), -1)
        cv2.rectangle(display_bgr, (BTN_SAVE_STREAM["x"], BTN_SAVE_STREAM["y"]),
                      (BTN_SAVE_STREAM["x"] + BTN_SAVE_STREAM["w"], BTN_SAVE_STREAM["y"] + BTN_SAVE_STREAM["h"]), (168, 85, 247), 2)
        cv2.putText(display_bgr, "STOP & SAVE", (BTN_SAVE_STREAM["x"] + 20, BTN_SAVE_STREAM["y"] + 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.44, (168, 85, 247), 1, cv2.LINE_AA)

        # Button 3: Close
        cv2.rectangle(display_bgr, (BTN_CLOSE_STREAM["x"], BTN_CLOSE_STREAM["y"]),
                      (BTN_CLOSE_STREAM["x"] + BTN_CLOSE_STREAM["w"], BTN_CLOSE_STREAM["y"] + BTN_CLOSE_STREAM["h"]), (40, 20, 25), -1)
        cv2.rectangle(display_bgr, (BTN_CLOSE_STREAM["x"], BTN_CLOSE_STREAM["y"]),
                      (BTN_CLOSE_STREAM["x"] + BTN_CLOSE_STREAM["w"], BTN_CLOSE_STREAM["y"] + BTN_CLOSE_STREAM["h"]), (239, 68, 68), 2)
        cv2.putText(display_bgr, "CLOSE", (BTN_CLOSE_STREAM["x"] + 40, BTN_CLOSE_STREAM["y"] + 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.46, (239, 68, 68), 1, cv2.LINE_AA)
    else:
        # Window & Screen Modes: All 5 controls
        # Button 1: Play / Pause
        cv2.rectangle(display_bgr, (BTN_PAUSE["x"], BTN_PAUSE["y"]),
                      (BTN_PAUSE["x"] + BTN_PAUSE["w"], BTN_PAUSE["y"] + BTN_PAUSE["h"]), (25, 32, 45), -1)
        cv2.rectangle(display_bgr, (BTN_PAUSE["x"], BTN_PAUSE["y"]),
                      (BTN_PAUSE["x"] + BTN_PAUSE["w"], BTN_PAUSE["y"] + BTN_PAUSE["h"]), pause_col, 2)
        cv2.putText(display_bgr, pause_text, (BTN_PAUSE["x"] + 34, BTN_PAUSE["y"] + 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.46, pause_col, 1, cv2.LINE_AA)

        # Button 2: Stop & Save
        cv2.rectangle(display_bgr, (BTN_SAVE["x"], BTN_SAVE["y"]),
                      (BTN_SAVE["x"] + BTN_SAVE["w"], BTN_SAVE["y"] + BTN_SAVE["h"]), (35, 20, 45), -1)
        cv2.rectangle(display_bgr, (BTN_SAVE["x"], BTN_SAVE["y"]),
                      (BTN_SAVE["x"] + BTN_SAVE["w"], BTN_SAVE["y"] + BTN_SAVE["h"]), (168, 85, 247), 2)
        cv2.putText(display_bgr, "STOP & SAVE", (BTN_SAVE["x"] + 12, BTN_SAVE["y"] + 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.44, (168, 85, 247), 1, cv2.LINE_AA)

        # Button 3: Switch Tab (Direct 1-Click Cycle - No Dropdown)
        switch_col = (14, 165, 233)
        cv2.rectangle(display_bgr, (BTN_FOCUS["x"], BTN_FOCUS["y"]),
                      (BTN_FOCUS["x"] + BTN_FOCUS["w"], BTN_FOCUS["y"] + BTN_FOCUS["h"]), (15, 30, 45), -1)
        cv2.rectangle(display_bgr, (BTN_FOCUS["x"], BTN_FOCUS["y"]),
                      (BTN_FOCUS["x"] + BTN_FOCUS["w"], BTN_FOCUS["y"] + BTN_FOCUS["h"]), switch_col, 2)
        cv2.putText(display_bgr, "SWITCH TAB", (BTN_FOCUS["x"] + 16, BTN_FOCUS["y"] + 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.44, switch_col, 1, cv2.LINE_AA)

        # Button 4: Crop ROI
        cv2.rectangle(display_bgr, (BTN_CROP["x"], BTN_CROP["y"]),
                      (BTN_CROP["x"] + BTN_CROP["w"], BTN_CROP["y"] + BTN_CROP["h"]), (20, 35, 45), -1)
        cv2.rectangle(display_bgr, (BTN_CROP["x"], BTN_CROP["y"]),
                      (BTN_CROP["x"] + BTN_CROP["w"], BTN_CROP["y"] + BTN_CROP["h"]), (56, 189, 248), 2)
        cv2.putText(display_bgr, "CROP ROI", (BTN_CROP["x"] + 24, BTN_CROP["y"] + 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.44, (56, 189, 248), 1, cv2.LINE_AA)

        # Button 5: Close
        cv2.rectangle(display_bgr, (BTN_CLOSE["x"], BTN_CLOSE["y"]),
                      (BTN_CLOSE["x"] + BTN_CLOSE["w"], BTN_CLOSE["y"] + BTN_CLOSE["h"]), (40, 20, 25), -1)
        cv2.rectangle(display_bgr, (BTN_CLOSE["x"], BTN_CLOSE["y"]),
                      (BTN_CLOSE["x"] + BTN_CLOSE["w"], BTN_CLOSE["y"] + BTN_CLOSE["h"]), (239, 68, 68), 2)
        cv2.putText(display_bgr, "CLOSE", (BTN_CLOSE["x"] + 40, BTN_CLOSE["y"] + 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.46, (239, 68, 68), 1, cv2.LINE_AA)


def synthesize_simulation_audio(frame_idx: int, timestamp: float) -> np.ndarray:
    sr = 16000
    n_samples = int(sr / 30)
    t = np.linspace(timestamp, timestamp + (1.0 / 30.0), n_samples, endpoint=False)
    envelope = 0.5 + 0.5 * np.sin(2.0 * np.pi * 4.5 * t)
    carrier = (
        0.6 * np.sin(2.0 * np.pi * 185.0 * t)
        + 0.3 * np.sin(2.0 * np.pi * 370.0 * t)
        + 0.1 * np.random.randn(n_samples)
    )
    return (0.14 * envelope * carrier).astype(np.float32)


def main():
    ensure_input_desktop()

    parser = argparse.ArgumentParser(description="Single-Face Verified-Speaking Gate Test Studio")
    parser.add_argument("--mode", choices=["sim", "screen", "window", "stream", "file"], default="screen",
                        help="Capture mode: 'screen' (desktop ROI), 'window' (specific app in background), 'stream' (phone camera URL), 'file' (offline turbo video), or 'sim'")
    parser.add_argument("--window", type=str, default="Brave",
                        help="Window title query to capture in background (e.g. 'Brave', 'Chrome', 'YouTube')")
    parser.add_argument("--stream-url", type=str, default="",
                        help="Phone camera video stream URL or YouTube link (e.g. 'http://192.168.1.15:8080/video')")
    parser.add_argument("--video-file", type=str, default="",
                        help="Path to local video file for high-speed offline turbo decoding")
    parser.add_argument("--turbo", action="store_true", default=False,
                        help="Enable uncapped turbo decoding speed (disables frame throttling sleeps)")
    parser.add_argument("--stream-title", type=str, default="",
                        help="Human-readable title of the stream/video being tracked")
    parser.add_argument("--canonical-url", type=str, default="",
                        help="Canonical source URL (e.g. YouTube watch URL) for persistent registry tracking")
    parser.add_argument("--quality", choices=["480p", "720p", "1080p", "best"], default="480p",
                        help="Stream resolution / quality mode (default: '480p' for optimal low-overhead tracking)")
    parser.add_argument("--headless", action="store_true", default=False,
                        help="Run completely silently in background with no on-screen display window")
    parser.add_argument("--roi", type=str, default="0,0,1920,1080",
                        help="Screen ROI x,y,w,h (defaults to full 1920x1080 display)")
    parser.add_argument("--audio-device", type=str, default=None,
                        help="Audio device index or name (e.g. 'ASUS', 'VAC', or 8) for internal silent capture")
    parser.add_argument("--list-audio-devices", action="store_true", default=False,
                        help="List available audio input/virtual recording devices and exit")
    args = parser.parse_args()

    if args.list_audio_devices:
        import sounddevice as sd
        print("\nAvailable Audio Input & Loopback Devices:")
        print("=" * 65)
        for idx, dev in enumerate(sd.query_devices()):
            if dev.get("max_input_channels", 0) > 0:
                print(f"  [{idx:2d}] {dev['name']}")
        print("=" * 65)
        print("Tip: Route your browser to a virtual/secondary line in Windows Volume Mixer,")
        print("then pass --audio-device <index> so your speakers remain 100% silent!\n")
        return

    # Resolve audio device index if specified
    audio_dev_idx = None
    if args.audio_device is not None:
        try:
            audio_dev_idx = int(args.audio_device)
        except ValueError:
            import sounddevice as sd
            q = args.audio_device.lower()
            for idx, dev in enumerate(sd.query_devices()):
                if dev.get("max_input_channels", 0) > 0 and q in dev['name'].lower():
                    audio_dev_idx = idx
                    print(f"[*] Matched audio device '{args.audio_device}' -> [{idx}] {dev['name']}")
                    break

    if args.mode == "video" and not args.video:
        sim_clip_path = "tests/data/podcast_multi_face_test.mp4"
        if not os.path.exists(sim_clip_path):
            try:
                from tests.generate_podcast_clip import create_podcast_video
                create_podcast_video()
            except Exception:
                pass

    config = AppConfig()
    config.speaker.strict_single_face_only = True
    config.speaker.inter_syllable_hold_sec = 0.35
    config.speaker.conversational_pause_sec = 1.80

    print("[*] FaceKey AI Pipeline loading MediaPipe face mesh...", flush=True)
    pipeline = FacePipeline(config)
    pipeline.initialize()
    print("[+] FaceKey AI Pipeline active & tracking!", flush=True)

    # Master Atomic Clock & Dataset Readiness Tracker & Writer & Stream Registry
    atomic_clock = MasterAtomicClock()
    dataset_tracker = DatasetReadinessTracker(config.sessions_dir)
    dataset_writer = DatasetWriter(config.sessions_dir)
    stream_registry = StreamRegistry(os.path.dirname(os.path.abspath(__file__)))

    active_stream_title = [args.stream_title or ""]
    canonical_source_url = [args.canonical_url or args.stream_url or args.video_file or ""]

    mode = args.mode
    if args.video_file or (args.mode == "file"):
        mode = "file"

    audio_file_waveform = [None]

    roi_parts = [int(x) for x in args.roi.split(",")]
    current_roi = (roi_parts[0], roi_parts[1], roi_parts[2], roi_parts[3])

    cap = None
    screen_src = None
    audio_src = None
    latest_screen_rgb = [None]
    latest_audio_samples = [None]
    latest_grab_ts_ns = [0]

    # In-memory session accumulator for peace of mind persistence
    session_face_frames = []
    session_audio_frames = []

    def start_sim_mode():
        nonlocal cap, screen_src, audio_src
        if screen_src:
            screen_src.stop()
            screen_src = None
        if audio_src:
            audio_src.stop()
            audio_src = None
        cap = cv2.VideoCapture(sim_clip_path)
        pipeline.reset()

    def start_screen_mode():
        nonlocal cap, screen_src, audio_src
        if cap:
            cap.release()
            cap = None
        if screen_src:
            screen_src.stop()
            screen_src = None
        if audio_src:
            audio_src.stop()
            audio_src = None
        pipeline.reset()

        def on_screen(ts, rgb):
            latest_grab_ts_ns[0] = atomic_clock.now_ns()
            latest_screen_rgb[0] = rgb

        def on_audio(ts, samples):
            latest_audio_samples[0] = samples

        screen_src = ScreenCaptureSource(roi=current_roi, target_fps=30)
        screen_src.start(on_screen)
        audio_src = AudioCaptureSource(device_index=audio_dev_idx)
        audio_src.start(on_audio)

    def start_window_mode(target_hwnd=None, target_title=""):
        nonlocal cap, screen_src, audio_src, mode
        mode = "window"
        if cap:
            cap.release()
            cap = None
        if screen_src:
            screen_src.stop()
            screen_src = None
        if audio_src:
            audio_src.stop()
            audio_src = None
        pipeline.reset()

        def on_screen(ts, rgb):
            latest_grab_ts_ns[0] = atomic_clock.now_ns()
            if args.quality == "480p" and rgb.shape[0] > 480:
                scale = 480.0 / rgb.shape[0]
                latest_screen_rgb[0] = cv2.resize(rgb, (int(rgb.shape[1] * scale), 480), interpolation=cv2.INTER_AREA)
            else:
                latest_screen_rgb[0] = rgb

        def on_audio(ts, samples):
            latest_audio_samples[0] = samples

        query = target_title if target_title else (args.window or "brave")
        print(f"[*] Attaching background window capture to: '{query}'...")
        screen_src = WindowCaptureSource(window_title_query=query, target_fps=30)
        if target_hwnd:
            screen_src.target_hwnd = target_hwnd
            screen_src.target_title = target_title
        elif args.window and args.window.isdigit():
            screen_src.target_hwnd = int(args.window)
            screen_src.target_title = WindowCaptureSource.get_window_title_by_hwnd(int(args.window))
        try:
            screen_src.start(on_screen)
        except Exception as e:
            print(f"[!] Warning: Could not bind window capture for '{query}' ({e}).")
            print("[*] Falling back to screen capture mode. (Use [SWITCH TAB] or press F9 once target window is open).")
            mode = "screen"
            screen_src = ScreenCaptureSource(roi=current_roi, target_fps=30)
            screen_src.start(on_screen)

        audio_src = AudioCaptureSource(device_index=audio_dev_idx)
        audio_src.start(on_audio)

    def start_stream_mode():
        nonlocal cap, screen_src, audio_src
        if cap:
            cap.release()
            cap = None
        if screen_src:
            screen_src.stop()
            screen_src = None
        if audio_src:
            audio_src.stop()
            audio_src = None
        pipeline.reset()

        stream_target = args.stream_url
        if any(yt in stream_target.lower() for yt in ("youtube.com", "youtu.be")):
            print(f"[*] YouTube link detected. Resolving {args.quality} direct stream via yt-dlp...")
            try:
                import yt_dlp
                fmt_map = {
                    "480p": "best[height<=480]/bestvideo[height<=480]/best",
                    "720p": "best[height<=720]/bestvideo[height<=720]/best",
                    "1080p": "best[height<=1080]/bestvideo[height<=1080]/best",
                    "best": "best/bestvideo/best"
                }
                ydl_opts = {
                    'format': fmt_map.get(args.quality, fmt_map["480p"]),
                    'js_runtimes': {'node': {}, 'deno': {}},
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
                    info = ydl.extract_info(stream_target, download=False)
                    stream_target = info.get('url') or stream_target
                    yt_title = info.get('title', 'YouTube Video')
                    if not active_stream_title[0] or active_stream_title[0] == "YouTube Video":
                        active_stream_title[0] = yt_title
                    if not canonical_source_url[0]:
                        canonical_source_url[0] = info.get('webpage_url') or args.stream_url
                    print(f"[*] Connected: '{active_stream_title[0]}' ({args.quality})")
            except Exception as e:
                print(f"[!] Warning: yt-dlp resolution failed: {e}")

        def on_screen(ts, rgb):
            latest_grab_ts_ns[0] = atomic_clock.now_ns()
            if args.quality == "480p" and rgb.shape[0] > 480:
                scale = 480.0 / rgb.shape[0]
                latest_screen_rgb[0] = cv2.resize(rgb, (int(rgb.shape[1] * scale), 480), interpolation=cv2.INTER_AREA)
            else:
                latest_screen_rgb[0] = rgb

        def on_audio(ts, samples):
            latest_audio_samples[0] = samples

        print(f"[*] Connecting to video stream ({args.quality}): '{stream_target[:55]}...'")
        screen_src = StreamCaptureSource(stream_url=stream_target, target_fps=30)
        def on_stream_done():
            print("\n[*] Video stream playback completed (reached end of video). Finalizing and saving...")
            app_controls["quit_requested"] = True
        screen_src.on_complete_callback = on_stream_done
        screen_src.start(on_screen)
        audio_src = AudioCaptureSource(device_index=audio_dev_idx)
        audio_src.start(on_audio)

    def start_file_mode():
        nonlocal cap, screen_src, audio_src, mode
        mode = "file"
        if cap:
            cap.release()
            cap = None
        if screen_src:
            screen_src.stop()
            screen_src = None
        if audio_src:
            audio_src.stop()
            audio_src = None
        pipeline.reset()

        target_file = args.video_file or args.stream_url
        if not target_file or not os.path.exists(target_file):
            raise FileNotFoundError(f"Offline video file not found: {target_file}")

        print(f"[*] Loading offline turbo video: '{target_file}'")
        cap = cv2.VideoCapture(target_file)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {target_file}")

        fps_val = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_f = cap.get(cv2.CAP_PROP_FRAME_COUNT) or -1
        w_val = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h_val = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        print(f"[*] Video Properties: {w_val}x{h_val} @ {fps_val:.2f} FPS ({total_f:,.0f} total frames)")

        # Extract 16kHz audio track for genuine speech & VAD attribution
        print(f"[*] Extracting audio track for genuine speech & VAD attribution...")
        audio_file_waveform[0] = extract_audio_from_file(target_file, target_sr=16000)
        if audio_file_waveform[0] is not None:
            dur_sec = len(audio_file_waveform[0]) / 16000.0
            print(f"[+] Loaded {len(audio_file_waveform[0]):,} audio samples ({dur_sec:.1f}s genuine audio)")
        else:
            print("[!] No audio stream found or extraction failed; using synthetic fallback.")

    # Check if a pre-selected target was set via switch.flag by launcher/window_picker
    script_dir = os.path.dirname(os.path.abspath(__file__))
    startup_flag = os.path.join(script_dir, "switch.flag")
    if os.path.exists(startup_flag):
        try:
            with open(startup_flag, "r", encoding="utf-8") as sf:
                s_val = sf.read().strip()
            os.remove(startup_flag)
            if s_val == "screen":
                mode = "screen"
            elif s_val:
                mode = "window"
                args.window = s_val
        except Exception:
            pass

    if mode == "screen":
        start_screen_mode()
    elif mode == "window":
        start_window_mode()
    elif mode == "stream":
        start_stream_mode()
    elif mode == "file":
        start_file_mode()
    else:
        start_sim_mode()

    # Controls state
    app_controls = {
        "paused": False,
        "save_requested": False,
        "crop_requested": False,
        "switch_tab_requested": False,
        "quit_requested": False
    }

    # Intercept termination signals (Ctrl+C, Ctrl+Break) for guaranteed peaceful save
    def sig_handler(sig, frame):
        print("\n[*] Intercepted termination signal. Stopping peacefully and flushing data to disk...")
        app_controls["quit_requested"] = True

    try:
        signal.signal(signal.SIGINT, sig_handler)
        signal.signal(signal.SIGTERM, sig_handler)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, sig_handler)
    except Exception:
        pass

    # Background Console Listener Thread (type 'stop', 'save', 'pause', 'switch', 'q' in terminal)
    def console_listener():
        while not app_controls["quit_requested"]:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                cmd = line.strip().lower()
                if cmd in ("q", "quit", "stop", "exit"):
                    print("\n[*] Console 'stop' command received. Finalizing peacefully...")
                    app_controls["quit_requested"] = True
                    break
                elif cmd in ("s", "save"):
                    print("\n[*] Console 'save' command received. Flushing frames to disk...")
                    app_controls["save_requested"] = True
                elif cmd.startswith("switch") or cmd.startswith("tab") or cmd in ("f", "focus", "w", "window"):
                    parts = cmd.split(maxsplit=1)
                    if len(parts) > 1:
                        app_controls["target_query_request"] = parts[1].strip()
                        print(f"\n[*] Console switch request for '{parts[1].strip()}' received...")
                    else:
                        print("\n[*] Console 'switch' command received. Switching to next open window...")
                    app_controls["switch_tab_requested"] = True
                elif cmd in ("p", "pause", "play"):
                    app_controls["paused"] = not app_controls["paused"]
                    print(f"[*] Pause toggled: paused={app_controls['paused']}")
            except Exception:
                break

    t_console = threading.Thread(target=console_listener, daemon=True)
    t_console.start()

    # Dedicated Global Hotkey & Desktop Hook Thread (Works even when viewing YouTube or reading)
    def global_hotkey_listener():
        if sys.platform != "win32":
            return
        ensure_input_desktop()
        user32 = ctypes.windll.user32
        while not app_controls["quit_requested"]:
            try:
                # F9: Instantly lock onto whichever tab or window is currently foreground
                if user32.GetAsyncKeyState(0x78) & 0x8000:  # VK_F9
                    ensure_input_desktop()
                    fg_hwnd = user32.GetForegroundWindow()
                    if fg_hwnd and fg_hwnd != hwnd:
                        fg_title = WindowCaptureSource.get_window_title_by_hwnd(fg_hwnd)
                        if fg_title and "facekey" not in fg_title.lower() and "program manager" not in fg_title.lower():
                            cur_h = getattr(screen_src, 'target_hwnd', None) if mode == 'window' else None
                            if fg_hwnd != cur_h:
                                print(f"\n[*] Global Hotkey F9 pressed! Locking focus onto: [{fg_hwnd}] '{fg_title}'")
                                try:
                                    import winsound
                                    winsound.MessageBeep(winsound.MB_ICONASTERISK)
                                except Exception:
                                    pass
                                app_controls["target_query_request"] = str(fg_hwnd)
                                app_controls["switch_tab_requested"] = True
                                time.sleep(0.5)

                # F8: Cycle through next window
                if user32.GetAsyncKeyState(0x77) & 0x8000:  # VK_F8
                    print("\n[*] Global Hotkey F8 pressed! Cycling window...")
                    app_controls["switch_tab_requested"] = True
                    time.sleep(0.4)

                # Shift+Esc: Peaceful Quit
                if (user32.GetAsyncKeyState(0x10) & 0x8000) and (user32.GetAsyncKeyState(0x1B) & 0x8000):
                    print("\n[*] Global Hotkey Shift+Esc detected! Finalizing peacefully...")
                    app_controls["quit_requested"] = True
                    break
            except Exception:
                pass
            time.sleep(0.04)

    t_hotkey = threading.Thread(target=global_hotkey_listener, daemon=True)
    t_hotkey.start()

    win_name = None
    hwnd = None
    if not args.headless:
        win_name = "FaceKey Studio - Single-Face Verified-Speaking Gate (HD Preview)"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 1280, 720)
        hwnd = setup_window_position_and_affinity(win_name, x=50, y=30, w=1280, h=720)

        def on_mouse(event, x, y, flags, param):
            # Dynamic Cursor Feedback: Hand over active buttons, Arrow over canvas
            is_stream_mode = (mode == "stream")
            curr_buttons = (BTN_PAUSE_STREAM, BTN_SAVE_STREAM, BTN_CLOSE_STREAM) if is_stream_mode else (BTN_PAUSE, BTN_SAVE, BTN_FOCUS, BTN_CROP, BTN_CLOSE)
            if sys.platform == "win32" and h_hand and h_arrow:
                is_over_button = any(
                    btn["x"] <= x <= btn["x"] + btn["w"] and btn["y"] <= y <= btn["y"] + btn["h"]
                    for btn in curr_buttons
                )
                if is_over_button:
                    ctypes.windll.user32.SetCursor(h_hand)
                else:
                    ctypes.windll.user32.SetCursor(h_arrow)

            if event == cv2.EVENT_LBUTTONDOWN:
                if is_stream_mode:
                    if (BTN_PAUSE_STREAM["x"] <= x <= BTN_PAUSE_STREAM["x"] + BTN_PAUSE_STREAM["w"] and
                        BTN_PAUSE_STREAM["y"] <= y <= BTN_PAUSE_STREAM["y"] + BTN_PAUSE_STREAM["h"]):
                        app_controls["paused"] = not app_controls["paused"]
                    elif (BTN_SAVE_STREAM["x"] <= x <= BTN_SAVE_STREAM["x"] + BTN_SAVE_STREAM["w"] and
                          BTN_SAVE_STREAM["y"] <= y <= BTN_SAVE_STREAM["y"] + BTN_SAVE_STREAM["h"]):
                        app_controls["save_requested"] = True
                    elif (BTN_CLOSE_STREAM["x"] <= x <= BTN_CLOSE_STREAM["x"] + BTN_CLOSE_STREAM["w"] and
                          BTN_CLOSE_STREAM["y"] <= y <= BTN_CLOSE_STREAM["y"] + BTN_CLOSE_STREAM["h"]):
                        app_controls["quit_requested"] = True
                else:
                    if (BTN_PAUSE["x"] <= x <= BTN_PAUSE["x"] + BTN_PAUSE["w"] and
                        BTN_PAUSE["y"] <= y <= BTN_PAUSE["y"] + BTN_PAUSE["h"]):
                        app_controls["paused"] = not app_controls["paused"]
                    elif (BTN_SAVE["x"] <= x <= BTN_SAVE["x"] + BTN_SAVE["w"] and
                          BTN_SAVE["y"] <= y <= BTN_SAVE["y"] + BTN_SAVE["h"]):
                        app_controls["save_requested"] = True
                    elif (BTN_FOCUS["x"] <= x <= BTN_FOCUS["x"] + BTN_FOCUS["w"] and
                          BTN_FOCUS["y"] <= y <= BTN_FOCUS["y"] + BTN_FOCUS["h"]):
                        print("\n[*] On-Screen [SWITCH TAB] button clicked! Opening Window Picker dialog...")
                        def _launch_picker():
                            try:
                                import subprocess
                                subprocess.call(
                                    [sys.executable, os.path.join(script_dir, "src", "ui", "window_picker.py")],
                                    cwd=script_dir
                                )
                            except Exception:
                                app_controls["switch_tab_requested"] = True
                        threading.Thread(target=_launch_picker, daemon=True).start()
                    elif (BTN_CROP["x"] <= x <= BTN_CROP["x"] + BTN_CROP["w"] and
                          BTN_CROP["y"] <= y <= BTN_CROP["y"] + BTN_CROP["h"]):
                        app_controls["crop_requested"] = True
                    elif (BTN_CLOSE["x"] <= x <= BTN_CLOSE["x"] + BTN_CLOSE["w"] and
                          BTN_CLOSE["y"] <= y <= BTN_CLOSE["y"] + BTN_CLOSE["h"]):
                        app_controls["quit_requested"] = True

        cv2.setMouseCallback(win_name, on_mouse)

    sim_frame_idx = 0
    global_frame_idx = 0
    recorded_frames_count = 0
    session_speech_sec = 0.0
    session_pause_sec = 0.0
    session_inter_word_sec = 0.0
    save_status_msg = ""
    save_msg_expiry = 0.0

    fps_ema = 30.0
    last_loop_ns = time.perf_counter_ns()
    last_print_ts = 0.0
    last_grab_ns = 0
    t_session_start_perf = time.perf_counter()

    print("\n" + "=" * 82)
    print(" [RUNNING] YOUTUBE LIVE FACIAL MOTION, TEETH & MUSCLE TRACKING STUDIO")
    print("  -> BUTTONS     : Click [PLAY/PAUSE], [STOP & SAVE], [CROP ROI], [CLOSE] on screen!")
    print("  -> TEETH DATA  : Maxillary / Mandibular Exposure + Inter-Dental Gap Tracking")
    print("  -> CHEEKS & MUS: Cheek Squint, Cheek Puff, Nose Sneer, Brow Furrowing")
    print("  -> 3D HEAD POSE: 3D Coordinate Tripod on Nose Tip (Pitch, Yaw, Roll, Tilt)")
    print("  -> EYE TRACKING: Pupil Iris Rings + 3D Gaze Ray Vectors + Reticle Widget")
    print("  -> ATOMIC CLOCK: Nanosecond Master Clock with Stream Gap Detection")
    print("=" * 82 + "\n")

    try:
        while True:
            # Check Quit
            if app_controls["quit_requested"]:
                break

            # 1. Check file-based stop trigger (created by STOP_STUDIO_PEACEFULLY.bat)
            if os.path.exists("stop.flag"):
                print("\n[*] Detected 'stop.flag' trigger. Finalizing and saving peacefully...")
                try:
                    os.remove("stop.flag")
                except Exception:
                    pass
                app_controls["quit_requested"] = True
                break

            # 2. Check file-based switch tab trigger (created by SWITCH_TAB.bat or shortcuts)
            if os.path.exists("switch.flag"):
                try:
                    with open("switch.flag", "r", encoding="utf-8") as sf:
                        flag_content = sf.read().strip()
                    os.remove("switch.flag")
                except Exception:
                    flag_content = ""
                if flag_content:
                    app_controls["target_query_request"] = flag_content
                app_controls["switch_tab_requested"] = True

            # 3. Check if video stream finished playback (EOF)
            if screen_src and getattr(screen_src, "is_completed", False):
                print("\n[*] Video stream playback completed (reached end of video). Finalizing and saving...")
                app_controls["quit_requested"] = True
                break

            loop_now_ns = time.perf_counter_ns()
            loop_now_sec = loop_now_ns / 1e9

            # Clear expired save status message
            if save_status_msg and loop_now_sec > save_msg_expiry:
                save_status_msg = ""

            # Check Save Request
            if app_controls["save_requested"]:
                app_controls["save_requested"] = False
                if session_face_frames:
                    s_id = time.strftime("%Y%m%d_%H%M%S") + "_save_" + uuid.uuid4().hex[:4]
                    t_end_perf = time.perf_counter()
                    saved_count = len(session_face_frames)
                    session_telemetry = {
                        "inter_word_seconds": round(session_inter_word_sec, 2),
                        "speech_seconds": round(session_speech_sec, 2),
                        "pause_seconds": round(session_pause_sec, 2)
                    }
                    if canonical_source_url[0]:
                        session_telemetry["source_url"] = canonical_source_url[0]
                    if active_stream_title[0]:
                        session_telemetry["video_title"] = active_stream_title[0]

                    saved_path = dataset_writer.write_session(
                        session_id=s_id,
                        mode=f"live_{mode}",
                        source_description=f"Mode {mode} (ROI {current_roi})",
                        face_frames=session_face_frames,
                        audio_frames=session_audio_frames,
                        session_start_time=t_session_start_perf,
                        session_end_time=t_end_perf,
                        capture_fps=fps_ema,
                        inference_fps=fps_ema,
                        telemetry=session_telemetry
                    )
                    dataset_tracker.scan_sessions()

                    # Record to StreamRegistry if this was a stream session
                    if mode == "stream" and canonical_source_url[0]:
                        try:
                            stream_registry.record_stream_session(
                                url=canonical_source_url[0],
                                title=active_stream_title[0] or "Stream Video",
                                quality=args.quality,
                                session_id=s_id,
                                frame_count=saved_count,
                                duration_seconds=session_speech_sec + session_pause_sec + session_inter_word_sec
                            )
                        except Exception as reg_err:
                            print(f"[!] Warning: Could not register stream session: {reg_err}")

                    save_status_msg = f"[SAVED TO DISK] {saved_count} frames -> session_{s_id}"
                    save_msg_expiry = loop_now_sec + 5.0
                    session_face_frames.clear()
                    session_audio_frames.clear()
                    recorded_frames_count = 0
                    session_speech_sec = 0.0
                    session_pause_sec = 0.0
                    session_inter_word_sec = 0.0
                    print(f"[*] {save_status_msg}", flush=True)
                    notify_user(
                        "FaceKey Studio - Session Saved",
                        f"Successfully saved {saved_count} frames to disk under session_{s_id}"
                    )
                else:
                    save_status_msg = "[NOTICE] No frames recorded yet to save."
                    save_msg_expiry = loop_now_sec + 3.0

            # Check Crop Request
            if app_controls["crop_requested"] and not args.headless:
                app_controls["crop_requested"] = False
                ensure_input_desktop()
                import mss
                with mss.mss() as sct:
                    mon = sct.monitors[1]
                    full_bgra = np.array(sct.grab(mon))
                    full_bgr = cv2.cvtColor(full_bgra, cv2.COLOR_BGRA2BGR)
                    roi_win = "DRAG BOX AROUND YOUTUBE VIDEO (Press ENTER or SPACE to confirm)"
                    cv2.namedWindow(roi_win, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(roi_win, 1280, 720)
                    r = cv2.selectROI(roi_win, full_bgr, fromCenter=False)
                    cv2.destroyWindow(roi_win)
                    if r[2] > 32 and r[3] > 32:
                        current_roi = (int(r[0]), int(r[1]), int(r[2]), int(r[3]))
                        mode = "screen"
                        start_screen_mode()

            # Direct Tab / Window Switch
            if app_controls["switch_tab_requested"]:
                app_controls["switch_tab_requested"] = False
                req_query = app_controls.pop("target_query_request", "").strip().lower()

                if req_query == "screen":
                    mode = "screen"
                    start_screen_mode()
                    save_status_msg = "[MODE] Switched to Full Screen ROI"
                    save_msg_expiry = loop_now_sec + 3.0
                    continue

                cur_h = getattr(screen_src, 'target_hwnd', None) if mode == 'window' else None
                windows = WindowCaptureSource.list_all_windows()
                candidates = [(h, t) for h, t in windows if h != hwnd and "facekey" not in t.lower() and "program manager" not in t.lower()]

                sel_h, sel_t = None, ""
                if req_query:
                    # 1. Direct HWND check (guarantees locking onto exact window requested)
                    if req_query.isdigit():
                        cand_h = int(req_query)
                        try:
                            import win32gui
                            if win32gui.IsWindow(cand_h):
                                sel_h = cand_h
                                sel_t = WindowCaptureSource.get_window_title_by_hwnd(cand_h)
                        except Exception:
                            pass
                    # 2. Match against candidates list by HWND or title substring
                    if not sel_h and candidates:
                        for h, t in candidates:
                            if req_query == str(h) or req_query in t.lower():
                                sel_h, sel_t = h, t
                                break

                # 3. Only cycle to next window if NO specific target was requested (e.g. F8 cycle)
                if not sel_h and not req_query and candidates:
                    cur_idx = -1
                    for idx, (h, t) in enumerate(candidates):
                        if h == cur_h:
                            cur_idx = idx
                            break
                    next_idx = (cur_idx + 1) % len(candidates)
                    sel_h, sel_t = candidates[next_idx]

                if sel_h:
                    start_window_mode(target_hwnd=sel_h, target_title=sel_t)
                    save_status_msg = f"🔒 LOCKED -> '{sel_t[:28]}'"
                    save_msg_expiry = loop_now_sec + 4.0
                    print(f"[*] Successfully locked capture target onto: [{sel_h}] '{sel_t}'", flush=True)
                else:
                    if req_query:
                        save_status_msg = f"[NOTICE] Window '{req_query}' not found."
                    else:
                        save_status_msg = "[NOTICE] No other open windows found."
                    save_msg_expiry = loop_now_sec + 3.0

            # PAUSED / LOW-RESOURCE REST STATE
            if app_controls["paused"]:
                time.sleep(0.04)  # Rest CPU & RAM resources peacefully
                if win_name and 'vis_bgr' in locals():
                    # Check if user clicked the window [X] close button
                    try:
                        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                            print("\n[*] Window close button [X] clicked while paused. Stopping peacefully...")
                            app_controls["quit_requested"] = True
                            break
                    except Exception:
                        pass

                    is_min = False
                    if hwnd and sys.platform == "win32":
                        try:
                            is_min = bool(ctypes.windll.user32.IsIconic(hwnd))
                        except Exception:
                            is_min = False

                    if not is_min:
                        pause_display = vis_bgr.copy()
                        cv2.rectangle(pause_display, (340, 310), (940, 395), (12, 16, 24), -1)
                        cv2.rectangle(pause_display, (340, 310), (940, 395), (250, 204, 21), 2)
                        cv2.putText(pause_display, "SYSTEM PAUSED - CPU/RAM AT REST", (375, 348),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.72, (250, 204, 21), 2, cv2.LINE_AA)
                        cv2.putText(pause_display, "Click [PLAY] button or press [SPACE] to resume", (410, 376),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (226, 232, 240), 1, cv2.LINE_AA)
                        draw_buttons_on_display(pause_display, is_paused=True, mode=mode)
                        cv2.imshow(win_name, pause_display)
                        key = cv2.waitKey(25) & 0xFF
                    else:
                        key = cv2.waitKey(25) & 0xFF

                    if key in (ord("q"), ord("Q"), 27):
                        break
                    elif key == ord(" "):
                        app_controls["paused"] = False
                    elif key in (ord("s"), ord("S")):
                        app_controls["save_requested"] = True
                continue

            dt_sec = max(1e-4, (loop_now_ns - last_loop_ns) / 1e9)
            dt_ms = dt_sec * 1000.0
            last_loop_ns = loop_now_ns
            fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt_sec)
            session_elapsed = atomic_clock.now_sec()
            ts = session_elapsed

            cur_grab_ns = latest_grab_ts_ns[0]
            is_stream_broken = (cur_grab_ns > 0 and last_grab_ns > 0 and (cur_grab_ns - last_grab_ns) > 55_000_000)
            last_grab_ns = cur_grab_ns

            t_infer_start = time.perf_counter()
            global_frame_idx += 1

            if mode == "file":
                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None:
                    print("\n[*] Offline turbo video reached EOF. Finalizing & saving...")
                    app_controls["quit_requested"] = True
                    break

                fps_source = cap.get(cv2.CAP_PROP_FPS) or 30.0
                frame_dt_sec = 1.0 / fps_source
                frame_dt_ms = frame_dt_sec * 1000.0
                ts = global_frame_idx * frame_dt_sec
                session_elapsed = ts

                if args.quality == "480p" and frame_bgr.shape[0] > 480:
                    scale = 480.0 / frame_bgr.shape[0]
                    frame_bgr = cv2.resize(frame_bgr, (int(frame_bgr.shape[1] * scale), 480), interpolation=cv2.INTER_AREA)

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                if audio_file_waveform[0] is not None:
                    aud = audio_file_waveform[0]
                    idx_start = max(0, int((ts - frame_dt_sec) * 16000))
                    idx_end = min(len(aud), int(ts * 16000))
                    if idx_end > idx_start:
                        cur_audio_samples = aud[idx_start:idx_end]
                    else:
                        cur_audio_samples = np.zeros(int(frame_dt_sec * 16000), dtype=np.float32)
                    audio_frame = pipeline.speaker_engine.update_audio(cur_audio_samples, ts)
                else:
                    cur_audio_samples = None
                    audio_frame = AudioFrameData(timestamp=ts, energy_rms=0.02, is_speech=True, vad_confidence=0.8)

                audio_frame.timestamp_ns = int(ts * 1e9)
                t_infer_start = time.perf_counter()
                tracked_faces = pipeline.process_frame(frame_rgb, ts, audio_frame)
                infer_latency_ms = (time.perf_counter() - t_infer_start) * 1000.0
                mode_label = f"TURBO FILE ({args.quality}): ({frame_bgr.shape[1]}x{frame_bgr.shape[0]})"
            elif mode == "sim":
                ret, frame_bgr = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    sim_frame_idx = 0
                    ret, frame_bgr = cap.read()
                else:
                    sim_frame_idx += 1

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                sim_audio = synthesize_simulation_audio(sim_frame_idx, ts)
                cur_audio_samples = sim_audio
                audio_frame = pipeline.speaker_engine.update_audio(sim_audio, ts)
                audio_frame.timestamp_ns = loop_now_ns
                tracked_faces = pipeline.process_frame(frame_rgb, ts, audio_frame)
                mode_label = "MODE 1: 4-PHASE SIMULATION"
            else:
                if latest_screen_rgb[0] is not None:
                    frame_rgb = latest_screen_rgb[0].copy()
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                else:
                    # If stream is completed, exit immediately - never show a blank screen!
                    if mode == "stream" and screen_src and getattr(screen_src, "is_completed", False):
                        app_controls["quit_requested"] = True
                        break
                    # Elegant loading splash card instead of pure black
                    frame_bgr = np.full((720, 1280, 3), (16, 20, 30), dtype=np.uint8)
                    cv2.rectangle(frame_bgr, (360, 300), (920, 420), (24, 32, 48), -1)
                    cv2.rectangle(frame_bgr, (360, 300), (920, 420), (52, 235, 100), 2)
                    cv2.putText(frame_bgr, "CONNECTING TO VIDEO STREAM...", (400, 345),
                                cv2.FONT_HERSHEY_DUPLEX, 0.65, (52, 235, 100), 1, cv2.LINE_AA)
                    cv2.putText(frame_bgr, "Buffering stream frames & initializing AI tracking...", (405, 385),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 210, 225), 1, cv2.LINE_AA)
                    frame_rgb = frame_bgr.copy()

                cur_audio_samples = latest_audio_samples[0]
                if latest_audio_samples[0] is not None:
                    audio_frame = pipeline.speaker_engine.update_audio(latest_audio_samples[0], ts)
                else:
                    audio_frame = AudioFrameData(timestamp=ts, energy_rms=0.02, is_speech=True, vad_confidence=0.8)

                audio_frame.timestamp_ns = loop_now_ns
                tracked_faces = pipeline.process_frame(frame_rgb, ts, audio_frame)
                if mode == "window":
                    target_name = getattr(screen_src, "target_title", args.window)
                    mode_label = f"WINDOW ({args.quality}): [{target_name[:26]}] ({frame_bgr.shape[1]}x{frame_bgr.shape[0]})"
                elif mode == "stream":
                    mode_label = f"STREAM ({args.quality}): ({frame_bgr.shape[1]}x{frame_bgr.shape[0]})"
                else:
                    mode_label = f"SCREEN ROI ({args.quality}): {current_roi} ({frame_bgr.shape[1]}x{frame_bgr.shape[0]})"

            infer_latency_ms = (time.perf_counter() - t_infer_start) * 1000.0

            is_gate_active = pipeline.speaker_engine.is_gate_active
            gate_status = pipeline.speaker_engine.gate_status
            conv_state = getattr(pipeline.speaker_engine, "conversational_state", "SILENCE")

            if is_gate_active and len(tracked_faces) == 1:
                recorded_frames_count += 1
                increment_sec = frame_dt_sec if mode == "file" else dt_sec
                if conv_state == "ACTIVE_SPEECH":
                    session_speech_sec += increment_sec
                elif conv_state == "BETWEEN_WORDS":
                    session_inter_word_sec += increment_sec
                elif conv_state == "CONVERSATIONAL_PAUSE":
                    session_pause_sec += increment_sec
                # Accumulate for persistence
                session_face_frames.extend(tracked_faces)
                if audio_frame:
                    session_audio_frames.append(audio_frame)

            for f in tracked_faces:
                f.timestamp_ns = loop_now_ns
                f.capture_hw_ts_ns = cur_grab_ns
                f.inference_latency_ms = infer_latency_ms
                f.inter_frame_delta_ms = dt_ms
                f.is_stream_broken = is_stream_broken
                f.conversational_state = conv_state

            dataset_report = dataset_tracker.compute_cumulative_report(
                live_frames=recorded_frames_count,
                live_speech_sec=session_speech_sec,
                live_pause_sec=session_pause_sec
            )

            if win_name:
                vis_bgr = draw_studio_hud(
                    frame_bgr=frame_bgr,
                    tracked_faces=tracked_faces,
                    audio_frame=audio_frame,
                    mode_label=mode_label,
                    fps=fps_ema,
                    frame_idx=global_frame_idx,
                    recorded_frames_count=recorded_frames_count,
                    session_elapsed=session_elapsed,
                    session_speech_sec=session_speech_sec,
                    session_pause_sec=session_pause_sec,
                    is_gate_active=is_gate_active,
                    gate_status=gate_status,
                    speaker_engine=pipeline.speaker_engine,
                    atomic_clock=atomic_clock,
                    dataset_report=dataset_report,
                    latency_ms=infer_latency_ms,
                    is_stream_broken=is_stream_broken,
                    dt_ms=dt_ms,
                    is_paused=False,
                    save_status_msg=save_status_msg,
                    audio_samples=cur_audio_samples,
                    session_inter_word_sec=session_inter_word_sec
                )

            if (loop_now_ns / 1e9) - last_print_ts >= 1.5:
                last_print_ts = loop_now_ns / 1e9
                euler_str = ""
                if tracked_faces:
                    e = tracked_faces[0].clean.head_pose_euler
                    euler_str = f" | Pose: P:{e[0]:+4.1f} Y:{e[1]:+4.1f} R:{e[2]:+4.1f}"
                target_desc = getattr(screen_src, 'target_title', active_stream_title[0] if mode in ('file', 'stream') else (args.window if mode == 'window' else f"Screen ROI {current_roi}"))
                if mode == "file":
                    prefix = "[TURBO COLLECTOR]"
                    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    tot_f = cap.get(cv2.CAP_PROP_FRAME_COUNT) or -1
                    pct_str = f" ({global_frame_idx/max(1, tot_f)*100:.1f}%)" if tot_f > 0 else ""
                    speed_str = f" | Speed: {fps_ema/fps_src:.1f}x ({fps_ema:.1f} FPS)"
                    print(
                        f"{prefix} Target: '{target_desc[:26]}' | Frame #{global_frame_idx:05d}{pct_str}{speed_str} | "
                        f"State: {conv_state}{euler_str} | Session: {recorded_frames_count} frames ({session_speech_sec:.1f}s) | "
                        f"Total: {dataset_report.total_duration_seconds/60.0:.1f} mins",
                        flush=True
                    )
                else:
                    prefix = "[BACKGROUND CAPTURE]" if args.headless else "[DATASET MONITOR]"
                    print(
                        f"{prefix} Target: '{target_desc[:32]}' | Frame #{global_frame_idx:04d} | State: {conv_state}{euler_str} | "
                        f"Session: {recorded_frames_count} frames ({session_speech_sec:.1f}s) | "
                        f"Total: {dataset_report.total_duration_seconds/60.0:.1f} mins | "
                        f"{dataset_report.readiness_label}",
                        flush=True
                    )

                # Persist live dashboard status to disk for external status checkers
                try:
                    with open("live_status.json", "w", encoding="utf-8") as sf:
                        json.dump({
                            "is_running": True,
                            "pid": os.getpid(),
                            "mode": mode,
                            "target": target_desc,
                            "conversational_state": conv_state,
                            "is_recording": is_gate_active and len(tracked_faces) == 1,
                            "faces_detected": len(tracked_faces),
                            "session_frames": recorded_frames_count,
                            "speech_seconds": round(session_speech_sec, 1),
                            "inter_word_seconds": round(session_inter_word_sec, 1),
                            "pause_seconds": round(session_pause_sec, 1),
                            "fps": round(fps_ema, 1),
                            "latency_ms": round(infer_latency_ms, 1),
                            "total_dataset_mins": round(dataset_report.total_duration_seconds / 60.0, 1),
                            "readiness_label": dataset_report.readiness_label,
                            "last_updated": time.strftime("%H:%M:%S")
                        }, sf, indent=2)
                except Exception:
                    pass

                # Dynamically update the Windows Taskbar window title via OpenCV
                if win_name:
                    try:
                        rec_sym = "● [REC]" if (is_gate_active and len(tracked_faces) == 1) else "⏸ [WAIT]"
                        t_short = (target_desc if len(target_desc) <= 24 else target_desc[:22] + "..")
                        taskbar_title = f"FaceKey Studio | {rec_sym} #{recorded_frames_count:04d} ({session_speech_sec:.1f}s) | {conv_state} | {t_short}"
                        cv2.setWindowTitle(win_name, taskbar_title)
                    except Exception:
                        pass

            if win_name:
                if hwnd is None and sys.platform == "win32":
                    try:
                        hwnd = setup_window_position_and_affinity(win_name, x=50, y=30, w=1280, h=720)
                    except Exception:
                        hwnd = None

                # 1. Detect if user clicked the window [X] close button
                try:
                    if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                        print("\n[*] Window close button [X] clicked. Stopping peacefully and flushing data...")
                        app_controls["quit_requested"] = True
                        break
                except Exception:
                    pass

                # 2. Check if window is minimized (IsIconic)
                is_minimized = False
                if hwnd and sys.platform == "win32":
                    try:
                        is_minimized = bool(ctypes.windll.user32.IsIconic(hwnd))
                    except Exception:
                        is_minimized = False

                if is_minimized:
                    # Minimized to taskbar: continue capturing in background without rendering overhead
                    key = cv2.waitKey(15) & 0xFF
                else:
                    # 1:1 Pixel HD Rendering - Razor sharp overlay text & metrics with zero downsampling
                    display_vis = vis_bgr.copy()
                    draw_buttons_on_display(display_vis, is_paused=False, mode=mode)
                    cv2.imshow(win_name, display_vis)
                    key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), ord("Q"), 27):
                    break
                elif key == ord(" "):
                    app_controls["paused"] = not app_controls["paused"]
                elif key in (ord("s"), ord("S")):
                    app_controls["save_requested"] = True
                elif key in (ord("w"), ord("W")):
                    def _launch_picker():
                        try:
                            import subprocess
                            subprocess.call(
                                [sys.executable, os.path.join(script_dir, "src", "ui", "window_picker.py")],
                                cwd=script_dir
                            )
                        except Exception:
                            app_controls["switch_tab_requested"] = True
                    threading.Thread(target=_launch_picker, daemon=True).start()
                elif key in (ord("f"), ord("F"), 9):  # 9 is TAB key
                    app_controls["switch_tab_requested"] = True
                elif key in (ord("r"), ord("R")):
                    app_controls["crop_requested"] = True
                elif key == ord("1") and mode != "sim":
                    mode = "sim"
                    start_sim_mode()
                elif key == ord("2") and mode != "screen":
                    mode = "screen"
                    start_screen_mode()
            else:
                if mode != "file" and not getattr(args, "turbo", False):
                    time.sleep(0.015)

    finally:
        # Save remaining frames on exit if any
        if session_face_frames:
            try:
                s_id = time.strftime("%Y%m%d_%H%M%S") + "_peaceful_" + uuid.uuid4().hex[:4]
                session_telemetry = {
                    "inter_word_seconds": round(session_inter_word_sec, 2),
                    "speech_seconds": round(session_speech_sec, 2),
                    "pause_seconds": round(session_pause_sec, 2)
                }
                if canonical_source_url[0]:
                    session_telemetry["source_url"] = canonical_source_url[0]
                if active_stream_title[0]:
                    session_telemetry["video_title"] = active_stream_title[0]

                dataset_writer.write_session(
                    session_id=s_id,
                    mode="peaceful_exit",
                    source_description=f"Mode {mode} (ROI {current_roi})",
                    face_frames=session_face_frames,
                    audio_frames=session_audio_frames,
                    session_start_time=t_session_start_perf,
                    session_end_time=time.perf_counter(),
                    capture_fps=fps_ema,
                    inference_fps=fps_ema,
                    telemetry=session_telemetry
                )

                if (mode in ("stream", "file")) and canonical_source_url[0]:
                    try:
                        stream_registry.record_stream_session(
                            url=canonical_source_url[0],
                            title=active_stream_title[0] or "Stream Video",
                            quality=args.quality,
                            session_id=s_id,
                            frame_count=len(session_face_frames),
                            duration_seconds=session_speech_sec + session_pause_sec + session_inter_word_sec
                        )
                    except Exception as reg_err:
                        print(f"[!] Warning: Could not register stream session: {reg_err}")

                print("\n" + "=" * 82)
                print(" [PEACEFULLY FINALIZED & SAVED]")
                print(f" Successfully saved {len(session_face_frames)} speech/pause frames to disk!")
                print(f" Saved under: sessions/session_{s_id}/")
                print("=" * 82 + "\n")
                notify_user(
                    "FaceKey Studio - Video Processed",
                    f"Successfully saved {len(session_face_frames)} frames ({session_speech_sec:.1f}s speech) under session_{s_id}"
                )
            except Exception as e:
                print(f"[!] Error saving on exit: {e}")
        elif screen_src and getattr(screen_src, "is_completed", False):
            try:
                notify_user(
                    "FaceKey Studio - Video Complete",
                    "Video stream playback finished and processing is complete."
                )
            except Exception:
                pass

        if cap:
            cap.release()
        if screen_src:
            screen_src.stop()
        if audio_src:
            audio_src.stop()
        if win_name:
            cv2.destroyAllWindows()
        pipeline.close()
        try:
            with open("live_status.json", "w", encoding="utf-8") as sf:
                json.dump({"is_running": False, "status": "Stopped peacefully", "last_updated": time.strftime("%H:%M:%S")}, sf, indent=2)
        except Exception:
            pass

        # Terminate cleanly with zero respawn
        os._exit(0)


if __name__ == "__main__":
    main()
