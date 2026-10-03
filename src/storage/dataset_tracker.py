"""Persistent dataset tracker and training readiness monitor across all sessions."""

import os
import json
import glob
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class DatasetReadinessReport:
    """Consolidated summary of all acquired facial motion data across historical & live sessions."""
    total_sessions_count: int = 0
    total_samples_frames: int = 0
    total_duration_seconds: float = 0.0
    total_size_bytes: int = 0
    total_active_speech_sec: float = 0.0
    total_pause_sec: float = 0.0
    # Training Milestones
    baseline_target_sec: float = 1800.0      # 30 Minutes of clean data for baseline model
    hifi_target_sec: float = 7200.0          # 120 Minutes for high-fidelity generative model
    phase1_target_hours: float = 100.0       # Phase 1 Target: 100 Hours of collected data
    scale_target_hours: float = 1000.0       # Scale Target: 1,000 Hours of collected data
    baseline_progress_pct: float = 0.0
    hifi_progress_pct: float = 0.0
    phase1_progress_pct: float = 0.0
    scale_progress_pct: float = 0.0
    is_baseline_ready: bool = False
    is_hifi_ready: bool = False
    is_phase1_ready: bool = False
    is_scale_ready: bool = False
    estimated_gb_for_100h: float = 47.9      # ~48 GB based on 490.6 MB/hr empirical rate
    estimated_gb_for_1000h: float = 479.1    # ~480 GB based on 490.6 MB/hr empirical rate
    readiness_label: str = "ACCUMULATING DATA"


class DatasetReadinessTracker:
    """Scans and tracks cumulative animation dataset readiness across all sessions on disk."""

    def __init__(self, sessions_dir: str = "sessions"):
        self.sessions_dir = os.path.abspath(sessions_dir)
        self.historical_sessions_count = 0
        self.historical_frames = 0
        self.historical_duration_sec = 0.0
        self.historical_size_bytes = 0
        self.historical_speech_sec = 0.0
        self.historical_pause_sec = 0.0
        self.scan_sessions()

    def scan_sessions(self):
        """Scan disk for all previous sessions and aggregate dataset metrics."""
        self.historical_sessions_count = 0
        self.historical_frames = 0
        self.historical_duration_sec = 0.0
        self.historical_size_bytes = 0
        self.historical_speech_sec = 0.0
        self.historical_pause_sec = 0.0

        if not os.path.exists(self.sessions_dir):
            return

        session_dirs = [
            d for d in glob.glob(os.path.join(self.sessions_dir, "session_*"))
            if os.path.isdir(d)
        ]

        for s_dir in session_dirs:
            meta_file = os.path.join(s_dir, "metadata.json")
            npz_file = os.path.join(s_dir, "face_motion.npz")
            s_size = 0

            if os.path.exists(npz_file):
                s_size += os.path.getsize(npz_file)
            if os.path.exists(meta_file):
                s_size += os.path.getsize(meta_file)

            if s_size > 0:
                self.historical_sessions_count += 1
                self.historical_size_bytes += s_size

            if os.path.exists(meta_file):
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        frames = meta.get("total_video_frames", 0)
                        dur = meta.get("duration_seconds", 0.0)
                        self.historical_frames += frames
                        self.historical_duration_sec += dur
                        # Check role counts for pause vs speech if available
                        roles = meta.get("role_counts", {})
                        speaker_count = roles.get("SPEAKER", frames)
                        pause_count = roles.get("SPEAKER_PAUSE", 0)
                        total_role_count = max(1, speaker_count + pause_count)
                        self.historical_speech_sec += dur * (speaker_count / total_role_count)
                        self.historical_pause_sec += dur * (pause_count / total_role_count)
                except Exception:
                    pass
            elif os.path.exists(npz_file):
                # Fallback estimation from npz file size (~15.2 KB per frame @ 30fps)
                est_frames = max(1, int(s_size / 15500))
                self.historical_frames += est_frames
                self.historical_duration_sec += est_frames / 30.0

    def compute_cumulative_report(
        self,
        live_frames: int = 0,
        live_speech_sec: float = 0.0,
        live_pause_sec: float = 0.0
    ) -> DatasetReadinessReport:
        """Combine persistent historical disk dataset with current live session stream."""
        live_dur = live_speech_sec + live_pause_sec
        live_bytes = int(live_frames * 15500)  # ~15.2 KB per sample

        total_sessions = self.historical_sessions_count + (1 if live_frames > 0 else 0)
        total_frames = self.historical_frames + live_frames
        total_dur = self.historical_duration_sec + live_dur
        total_size = self.historical_size_bytes + live_bytes
        total_speech = self.historical_speech_sec + live_speech_sec
        total_pause = self.historical_pause_sec + live_pause_sec

        baseline_target = 1800.0   # 30 Mins
        hifi_target = 7200.0       # 120 Mins (2 Hours)
        phase1_target_sec = 360000.0  # 100 Hours
        scale_target_sec = 3600000.0  # 1,000 Hours

        baseline_pct = min(100.0, (total_dur / baseline_target) * 100.0) if baseline_target > 0 else 0.0
        hifi_pct = min(100.0, (total_dur / hifi_target) * 100.0) if hifi_target > 0 else 0.0
        phase1_pct = min(100.0, (total_dur / phase1_target_sec) * 100.0) if phase1_target_sec > 0 else 0.0
        scale_pct = min(100.0, (total_dur / scale_target_sec) * 100.0) if scale_target_sec > 0 else 0.0

        is_baseline_ready = total_dur >= baseline_target
        is_hifi_ready = total_dur >= hifi_target
        is_phase1_ready = total_dur >= phase1_target_sec
        is_scale_ready = total_dur >= scale_target_sec

        total_hours = total_dur / 3600.0
        if is_scale_ready:
            label = f"READY: 1,000h SCALE DATASET COMPLETE ({total_hours:.1f}h)"
        elif is_phase1_ready:
            label = f"PHASE 1 COMPLETE ({total_hours:.1f}h / 100h) - ADVANCING TO 1,000h TARGET"
        else:
            remaining_hours = max(0.0, 100.0 - total_hours)
            label = f"PHASE 1 (100h TARGET): {total_hours:.2f}h / 100.0h ({phase1_pct:.1f}%) | {remaining_hours:.1f}h TO GO"

        return DatasetReadinessReport(
            total_sessions_count=total_sessions,
            total_samples_frames=total_frames,
            total_duration_seconds=total_dur,
            total_size_bytes=total_size,
            total_active_speech_sec=total_speech,
            total_pause_sec=total_pause,
            baseline_target_sec=baseline_target,
            hifi_target_sec=hifi_target,
            phase1_target_hours=100.0,
            scale_target_hours=1000.0,
            baseline_progress_pct=baseline_pct,
            hifi_progress_pct=hifi_pct,
            phase1_progress_pct=phase1_pct,
            scale_progress_pct=scale_pct,
            is_baseline_ready=is_baseline_ready,
            is_hifi_ready=is_hifi_ready,
            is_phase1_ready=is_phase1_ready,
            is_scale_ready=is_scale_ready,
            estimated_gb_for_100h=47.9,
            estimated_gb_for_1000h=479.1,
            readiness_label=label
        )
