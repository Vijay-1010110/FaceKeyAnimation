"""Configuration management and defaults for Facial Motion Capture System."""

import os
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List
import yaml


@dataclass
class FaceThresholds:
    """Configurable geometric and optical criteria for face eligibility."""
    min_width_px: int = 70
    min_height_px: int = 70
    min_area_ratio: float = 0.025
    min_detection_confidence: float = 0.35
    min_presence_confidence: float = 0.35
    min_tracking_confidence: float = 0.35
    min_blur_laplacian: float = 35.0
    max_abs_yaw_deg: float = 50.0
    max_abs_pitch_deg: float = 40.0
    max_abs_roll_deg: float = 35.0
    max_jitter_velocity: float = 1.2


@dataclass
class SpeakerAttributionConfig:
    """Settings for single-face verified speaking gate and speaker attribution."""
    min_speaker_confidence: float = 0.65
    min_listener_confidence: float = 0.50
    lip_motion_velocity_threshold: float = 0.16
    min_mouth_articulation_range: float = 0.012
    vad_energy_threshold: float = 0.008
    sliding_window_sec: float = 0.40
    narration_threshold: float = 0.40
    strict_single_face_only: bool = True
    inter_syllable_hold_sec: float = 0.25      # Inter-syllable closure hold within spoken words
    inter_word_hold_sec: float = 0.55          # Natural micro-pauses & coarticulation between spoken words
    conversational_pause_sec: float = 1.80     # Natural pauses, breathing, and thinking pauses between phrases
    atomic_clock_drift_warning_ms: float = 2.0 # Sub-millisecond AV sync precision target


@dataclass
class HysteresisConfig:
    """Threshold hysteresis settings to avoid rapid state flickering."""
    quality_enter_threshold: float = 0.65
    quality_exit_threshold: float = 0.50
    quality_min_duration_sec: float = 0.15
    watcher_trigger_sec: float = 0.30
    watcher_loss_sec: float = 1.80


@dataclass
class TemporalFilterConfig:
    """Parameters for adaptive One Euro filtering."""
    min_cutoff: float = 1.00
    beta: float = 0.007
    d_cutoff: float = 1.00


@dataclass
class AppConfig:
    """Top-level application settings."""
    model_path: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "face_landmarker.task")
    sessions_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sessions")
    target_fps: int = 30
    max_faces: int = 2
    ring_buffer_seconds: float = 2.0
    queue_max_size: int = 60
    debug_mode: bool = False
    save_diagnostic_frames: bool = False  # Off by default as per requirement
    face: FaceThresholds = field(default_factory=FaceThresholds)
    speaker: SpeakerAttributionConfig = field(default_factory=SpeakerAttributionConfig)
    hysteresis: HysteresisConfig = field(default_factory=HysteresisConfig)
    filter: TemporalFilterConfig = field(default_factory=TemporalFilterConfig)

    def to_yaml(self, path: str):
        """Serialize configuration to a YAML file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> "AppConfig":
        """Load configuration from a YAML file with graceful fallback."""
        if not os.path.exists(path):
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        face_data = raw.pop("face", {})
        speaker_data = raw.pop("speaker", {})
        hys_data = raw.pop("hysteresis", {})
        filter_data = raw.pop("filter", {})

        return cls(
            face=FaceThresholds(**face_data) if face_data else FaceThresholds(),
            speaker=SpeakerAttributionConfig(**speaker_data) if speaker_data else SpeakerAttributionConfig(),
            hysteresis=HysteresisConfig(**hys_data) if hys_data else HysteresisConfig(),
            filter=TemporalFilterConfig(**filter_data) if filter_data else TemporalFilterConfig(),
            **raw
        )
