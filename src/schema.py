"""Core schema and data models for facial key-animation data acquisition."""

from dataclasses import dataclass, field
from enum import IntEnum, Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class EligibilityLevel(IntEnum):
    """Multi-level dataset eligibility classification."""
    LEVEL_0_DETECTED = 0        # Face exists in image
    LEVEL_1_TRACKABLE = 1       # Persistent identity stable across frames
    LEVEL_2_GEOMETRICALLY_USABLE = 2  # Sufficient size, visibility, acceptable pose
    LEVEL_3_ANIMATION_QUALITY = 3     # Stable tracking, low jitter, temporal consistency
    LEVEL_4_SPEAKER_PAIRED = 4        # Animation quality + confident speaker attribution


class FaceRole(str, Enum):
    """Semantic role of a tracked face in relation to speech."""
    SPEAKER = "SPEAKER"
    SPEAKER_PAUSE = "SPEAKER_PAUSE"  # Natural conversational breathing / thinking pause in speech turn
    LISTENER = "LISTENER"
    UNKNOWN = "UNKNOWN"


@dataclass
class RawObservation:
    """Raw sensor perception data directly from the face model.
    Never overwrite or blend this with inferred values.
    """
    landmarks: np.ndarray             # Shape: (478, 3) normalized x, y, z
    blendshapes: Dict[str, float]     # 52 standard ARKit blendshape keys -> [0.0, 1.0]
    transformation_matrix: np.ndarray # 4x4 rigid transformation matrix
    head_pose_euler: Tuple[float, float, float]  # (pitch, yaw, roll) in degrees
    head_translation: Tuple[float, float, float]  # (tx, ty, tz)
    detection_confidence: float = 1.0
    tracking_confidence: float = 1.0


@dataclass
class CleanObservation:
    """Temporally filtered and gap-repaired observation for animation training.
    Maintains explicit provenance via is_reconstructed.
    """
    landmarks: np.ndarray             # Shape: (478, 3) filtered landmarks
    blendshapes: Dict[str, float]     # Filtered blendshape dictionary
    head_pose_euler: Tuple[float, float, float]  # Filtered pitch, yaw, roll
    head_translation: Tuple[float, float, float]  # Filtered translation
    is_observed: bool = True          # True if backed by real observation
    is_reconstructed: bool = False    # True if missing/interpolated


@dataclass
class QualityMetrics:
    """Detailed quality evaluation metrics for a facial sample."""
    composite_score: float            # Combined quality score [0.0, 1.0]
    blur_score: float                 # Laplacian variance (> threshold is sharp)
    is_sharp: bool
    face_width_px: int
    face_height_px: int
    area_ratio: float                 # Bbox area / ROI area
    pose_quality: float               # [0.0, 1.0] penalizes extreme yaw/pitch/roll
    temporal_jitter: float            # Euclidean velocity of landmark movement
    is_valid: bool                    # Overall validity flag
    rejection_reason: Optional[str] = None


@dataclass
class TrackedFaceFrame:
    """Complete timestamped observation of a single face at a specific moment."""
    timestamp: float                  # Master session timestamp in seconds (t >= 0.0)
    face_id: int                      # Persistent face identity within session
    is_primary: bool                  # Primary face of interest
    bbox: Tuple[int, int, int, int]   # (x, y, w, h) in pixels within ROI
    raw: RawObservation               # Layer 1: Raw observed perception
    clean: CleanObservation           # Layer 2: Clean animation-oriented signal
    quality: QualityMetrics           # Diagnostic quality metrics
    role: FaceRole = FaceRole.UNKNOWN # SPEAKER, SPEAKER_PAUSE, LISTENER, or UNKNOWN
    speaker_probability: float = 0.0  # Cross-modal speech correlation probability
    eligibility_level: EligibilityLevel = EligibilityLevel.LEVEL_0_DETECTED
    is_narration: bool = False        # True if audio speech was active but face was listening to narration
    # Atomic Clocking & Sub-millisecond AV Sync
    timestamp_ns: int = 0             # Monotonic atomic master clock (perf_counter_ns)
    capture_hw_ts_ns: int = 0         # Instant frame buffer was grabbed from display hardware
    inference_latency_ms: float = 0.0 # Time taken for MediaPipe inference + tracking in ms
    inter_frame_delta_ms: float = 0.0 # Inter-frame delta dt in ms
    is_stream_broken: bool = False    # True if frame dropped / stream hiccup detected
    conversational_state: str = "ACTIVE_SPEECH"  # ACTIVE_SPEECH, CONVERSATIONAL_PAUSE, or SILENT


@dataclass
class AudioFrameData:
    """Timestamped audio analysis slice."""
    timestamp: float                  # Time in seconds
    energy_rms: float                 # Short-time root-mean-square energy
    is_speech: bool                   # Voice activity detection decision
    vad_confidence: float             # Confidence in speech presence [0.0, 1.0]
    is_narration: bool = False        # True when speech is detected but no visible face is speaking
    active_speaker_face_id: Optional[int] = None # Face ID of the active speaker if identified
    # Atomic Clocking
    timestamp_ns: int = 0             # Monotonic atomic master clock in nanoseconds
    sample_start_idx: int = 0         # Continuous audio sample counter from session start
    sample_count: int = 0             # Number of audio samples in this slice


@dataclass
class FramePacket:
    """Transient in-memory packet during live processing.
    The raw image buffer is discarded immediately after landmark inference.
    """
    timestamp: float
    roi_image: Optional[np.ndarray]   # Ephemeral RAM buffer, set to None once processed
    audio_frame: Optional[AudioFrameData] = None
    tracked_faces: List[TrackedFaceFrame] = field(default_factory=list)


@dataclass
class SessionMetadata:
    """Permanent session summary and configuration metadata."""
    session_id: str
    mode: str                         # "watcher", "manual", or "uploaded_video"
    source_description: str
    created_at: str
    duration_seconds: float
    total_video_frames: int
    total_audio_frames: int
    tracked_face_ids: List[int]
    primary_face_id: Optional[int]
    eligibility_counts: Dict[str, int]
    role_counts: Dict[str, int]
    average_capture_fps: float
    average_inference_fps: float
    hardware_telemetry: Dict[str, Any] = field(default_factory=dict)
