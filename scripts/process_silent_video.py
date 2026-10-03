"""Silent Video & YouTube Ingestion Engine: Decodes video and audio directly in RAM.
Zero audio output through Windows speakers, zero on-screen clutter, 100% bit-perfect digital sync.
Allows training data collection while working in complete peace and silence.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
import uuid
from typing import Optional, List, Dict
import numpy as np

# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import av
except ImportError:
    av = None

try:
    import cv2
except ImportError:
    cv2 = None

from src.config import AppConfig
from src.core.face_pipeline import FacePipeline
from src.schema import AudioFrameData, TrackedFaceFrame, FaceRole
from src.storage.dataset_writer import DatasetWriter
from src.storage.dataset_tracker import DatasetReadinessTracker


def download_youtube_clip(url: str, output_path: str, max_duration_sec: int = 1800) -> str:
    """Downloads YouTube video up to 720p using yt-dlp."""
    print(f"[*] Downloading YouTube stream via yt-dlp: {url}")
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        "--merge-output-format", "mp4",
        "-o", output_path,
        "--no-playlist",
        url
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {res.stderr}")
    return output_path


def process_video_silently(
    video_path: str,
    source_name: str = "Video Ingestion",
    max_duration_sec: Optional[float] = None
):
    print("=" * 82)
    print(" [SILENT INGESTION] PROCESSING VIDEO & AUDIO DIRECTLY IN RAM")
    print(f" Source File        : {video_path}")
    print(" Audio Playback     : 100% MUTED (0.0 dB to speakers - complete silence)")
    print(" Screen Display     : HEADLESS (Zero windows on screen)")
    print("=" * 82 + "\n")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    config = AppConfig()
    config.speaker.strict_single_face_only = True
    config.speaker.inter_syllable_hold_sec = 0.35
    config.speaker.conversational_pause_sec = 1.80

    pipeline = FacePipeline(config)
    pipeline.initialize()

    dataset_writer = DatasetWriter(config.sessions_dir)
    dataset_tracker = DatasetReadinessTracker(config.sessions_dir)

    # 1. Open Video & Audio via PyAV
    container = av.open(video_path)
    video_stream = container.streams.video[0] if container.streams.video else None
    audio_stream = container.streams.audio[0] if container.streams.audio else None

    if not video_stream:
        raise ValueError("No video stream found in file.")

    fps = float(video_stream.average_rate) if video_stream.average_rate else 30.0
    duration_sec = float(container.duration) / av.time_base if container.duration else 0.0
    if max_duration_sec and duration_sec > max_duration_sec:
        duration_sec = max_duration_sec

    # Resample audio to 16kHz mono float32
    resampler = None
    if audio_stream:
        resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)

    # In-memory accumulators
    face_frames = []
    audio_frames = []

    # Map timestamps to audio chunks
    audio_sample_rate = 16000
    chunk_size = int(audio_sample_rate * (1.0 / max(1.0, fps)))

    # Pre-decode audio buffer into single continuous array if stream exists
    print("[*] Decoding digital audio track in memory...")
    all_audio_samples = []
    if audio_stream:
        for frame in container.decode(audio=0):
            resampled = resampler.resample(frame)
            for packet in resampled:
                arr = packet.to_ndarray()[0]
                all_audio_samples.append(arr)
        if all_audio_samples:
            all_audio_samples = np.concatenate(all_audio_samples, axis=0)
        else:
            all_audio_samples = np.zeros(int(duration_sec * audio_sample_rate), dtype=np.float32)
    else:
        all_audio_samples = np.zeros(int(duration_sec * audio_sample_rate), dtype=np.float32)

    total_audio_len = len(all_audio_samples)
    print(f"[*] Audio decoded: {total_audio_len / audio_sample_rate:.1f}s at 16kHz (PCM float32)")

    # 2. Seek back and decode video frames
    container.seek(0)
    frame_idx = 0
    t0_perf = time.perf_counter()
    last_print = 0.0
    recorded_speech_frames = 0
    recorded_pause_frames = 0

    print(f"[*] Processing visual kinematics at native {fps:.1f} FPS...")

    for frame in container.decode(video=0):
        ts = float(frame.pts * video_stream.time_base) if frame.pts is not None else (frame_idx / fps)
        if max_duration_sec and ts > max_duration_sec:
            break

        frame_rgb = frame.to_ndarray(format="rgb24")
        frame_idx += 1

        # Extract corresponding 16kHz audio chunk
        sample_start = int(ts * audio_sample_rate)
        sample_end = sample_start + chunk_size
        if sample_start < total_audio_len:
            chunk = all_audio_samples[sample_start:min(total_audio_len, sample_end)]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        else:
            chunk = np.zeros(chunk_size, dtype=np.float32)

        # Update speech attribution
        audio_frame = pipeline.speaker_engine.update_audio(chunk, ts)
        audio_frame.timestamp_ns = int(ts * 1e9)

        # Process MediaPipe 478 mesh
        tracked = pipeline.process_frame(frame_rgb, ts, audio_frame)

        is_gate_active = pipeline.speaker_engine.is_gate_active
        conv_state = getattr(pipeline.speaker_engine, "conversational_state", "SILENCE")

        if is_gate_active and len(tracked) == 1:
            if conv_state == "ACTIVE_SPEECH":
                recorded_speech_frames += 1
            elif conv_state == "CONVERSATIONAL_PAUSE":
                recorded_pause_frames += 1

            for f in tracked:
                f.timestamp_ns = int(ts * 1e9)
                f.capture_hw_ts_ns = int(ts * 1e9)
                f.conversational_state = conv_state
                face_frames.append(f)
            audio_frames.append(audio_frame)

        # Progress reporting
        now = time.perf_counter()
        if now - last_print >= 2.0:
            last_print = now
            pct = (ts / max(1e-4, duration_sec)) * 100.0
            speed = frame_idx / max(1e-4, (now - t0_perf))
            print(
                f"  [{pct:5.1f}%] Frame #{frame_idx:05d} ({ts:5.1f}s) | "
                f"Speed: {speed:.1f} FPS | "
                f"Recorded: {len(face_frames)} frames (Speech: {recorded_speech_frames}, Pause: {recorded_pause_frames})",
                flush=True
            )

    container.close()

    # 3. Save Session
    if face_frames:
        s_id = time.strftime("%Y%m%d_%H%M%S") + "_silent_" + uuid.uuid4().hex[:4]
        saved_path = dataset_writer.write_session(
            session_id=s_id,
            mode="silent_video_ingestion",
            source_description=source_name,
            face_frames=face_frames,
            audio_frames=audio_frames,
            session_start_time=0.0,
            session_end_time=ts,
            capture_fps=fps,
            inference_fps=fps
        )
        dataset_tracker.scan_sessions()

        print("\n" + "=" * 82)
        print(" [COMPLETE] SILENT VIDEO INGESTION FINISHED")
        print(f" Total Frames Processed : {frame_idx}")
        print(f" Accepted Speech Frames  : {len(face_frames)} ({len(face_frames)/fps/60.0:.2f} mins)")
        print(f" Output Location        : {saved_path}")
        print(" Total Audio Heard      : 0.0 dB (100% Silence for you)")
        print("=" * 82 + "\n")
    else:
        print("[!] No single-speaker face frames detected in video.")

    pipeline.close()


def main():
    parser = argparse.ArgumentParser(description="Silently ingest and process video/YouTube with 0 dB audio output.")
    parser.add_argument("--url", type=str, default=None, help="YouTube video URL")
    parser.add_argument("--file", type=str, default=None, help="Local video file path (MP4, MKV, WebM)")
    parser.add_argument("--max-duration", type=float, default=None, help="Max duration in seconds to process")
    args = parser.parse_args()

    if not args.url and not args.file:
        print("[ERROR] Please provide either --url <youtube_url> or --file <video_file>")
        print("Example: python scripts/process_silent_video.py --url 'https://www.youtube.com/watch?v=...'")
        print("Example: python scripts/process_silent_video.py --file 'my_video.mp4'")
        return

    video_path = args.file
    temp_download = False

    if args.url:
        os.makedirs("temp_cache", exist_ok=True)
        cached_file = os.path.join("temp_cache", f"yt_{uuid.uuid4().hex[:6]}.mp4")
        try:
            download_youtube_clip(args.url, cached_file, max_duration_sec=int(args.max_duration or 1800))
            video_path = cached_file
            temp_download = True
        except Exception as e:
            print(f"[ERROR] Failed to download YouTube video: {e}")
            return

    try:
        process_video_silently(
            video_path=video_path,
            source_name=args.url if args.url else os.path.basename(video_path),
            max_duration_sec=args.max_duration
        )
    finally:
        if temp_download and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception:
                pass


if __name__ == "__main__":
    main()
