"""Core face processing pipeline coordinating MediaPipe inference, tracking, quality, and filtering."""

import math
import os
import time
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

from src.config import AppConfig
from src.schema import (
    TrackedFaceFrame, RawObservation, CleanObservation,
    QualityMetrics, AudioFrameData, FramePacket
)
from src.core.tracker import MultiFaceTracker
from src.core.quality import QualityEvaluator
from src.core.speaker_attribution import SpeakerAttributionEngine
from src.core.temporal_filter import FaceTemporalFilter


class FacePipeline:
    """End-to-end perception pipeline for facial landmark and motion capture.
    Processes frames in RAM and immediately releases raw image buffers.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.landmarker: Optional[FaceLandmarker] = None
        self.tracker = MultiFaceTracker()
        self.quality_evaluator = QualityEvaluator(config.face, config.hysteresis)
        self.speaker_engine = SpeakerAttributionEngine(config.speaker)
        self.temporal_filter = FaceTemporalFilter(config.filter)
        self._last_video_timestamp_ms: int = -1

    def initialize(self):
        """Load and initialize the MediaPipe FaceLandmarker model."""
        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"MediaPipe task model not found at: {self.config.model_path}")

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=self.config.model_path,
                delegate=BaseOptions.Delegate.CPU  # Highly optimized CPU SIMD via XNNPACK
            ),
            running_mode=RunningMode.VIDEO,
            num_faces=self.config.max_faces,
            min_face_detection_confidence=self.config.face.min_detection_confidence,
            min_face_presence_confidence=self.config.face.min_presence_confidence,
            min_tracking_confidence=self.config.face.min_tracking_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True
        )
        self.landmarker = FaceLandmarker.create_from_options(options)
        self.reset()

    def reset(self):
        """Reset internal pipeline states."""
        self.tracker.reset()
        self.quality_evaluator.reset()
        self.speaker_engine.reset()
        self.temporal_filter.reset()
        self._last_video_timestamp_ms = -1

    def close(self):
        """Release neural network resources."""
        if self.landmarker:
            self.landmarker.close()
            self.landmarker = None

    @staticmethod
    def matrix_to_euler_angles(mat: np.ndarray) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Decompose a 4x4 transformation matrix into (pitch, yaw, roll in degrees) and (tx, ty, tz)."""
        if mat is None or mat.shape != (4, 4):
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

        # Extract translation
        tx, ty, tz = float(mat[0, 3]), float(mat[1, 3]), float(mat[2, 3])

        # Rotation matrix decomposition (standard aerospace Z-Y-X convention)
        r00, r01, r02 = mat[0, 0], mat[0, 1], mat[0, 2]
        r10, r11, r12 = mat[1, 0], mat[1, 1], mat[1, 2]
        r20, r21, r22 = mat[2, 0], mat[2, 1], mat[2, 2]

        sy = math.sqrt(r00 * r00 + r10 * r10)
        singular = sy < 1e-6

        if not singular:
            pitch = math.atan2(r21, r22)
            yaw = math.atan2(-r20, sy)
            roll = math.atan2(r10, r00)
        else:
            pitch = math.atan2(-r12, r11)
            yaw = math.atan2(-r20, sy)
            roll = 0.0

        deg = 180.0 / math.pi
        euler = (round(pitch * deg, 2), round(yaw * deg, 2), round(roll * deg, 2))
        trans = (round(tx, 3), round(ty, 3), round(tz, 3))
        return euler, trans

    def process_frame(
        self,
        frame_rgb: np.ndarray,
        timestamp: float,
        audio_frame: Optional[AudioFrameData] = None
    ) -> List[TrackedFaceFrame]:
        """Execute full facial perception pipeline on a single frame.
        The input frame_rgb is only read in RAM and is freed once this function returns.
        """
        if self.landmarker is None:
            raise RuntimeError("Pipeline must be initialized before processing frames.")

        # Ensure monotonically increasing timestamps for MediaPipe VIDEO mode
        ts_ms = max(self._last_video_timestamp_ms + 1, int(round(timestamp * 1000)))
        self._last_video_timestamp_ms = ts_ms

        # 1. Landmark & Blendshape Inference
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect_for_video(mp_image, ts_ms)

        num_faces_detected = len(result.face_landmarks)
        if num_faces_detected == 0:
            self.speaker_engine.attribute_speakers(
                [],
                AudioFrameData(timestamp=timestamp, energy_rms=0.0, is_speech=False, vad_confidence=0.0)
            )
            return []

        # Convert landmarks to normalized numpy array (N, 478, 3)
        detected_landmarks_list = []
        for face_lms in result.face_landmarks:
            pts = np.zeros((len(face_lms), 3), dtype=np.float32)
            for i, lm in enumerate(face_lms):
                pts[i] = [lm.x, lm.y, lm.z]
            detected_landmarks_list.append(pts)

        # 2. Multi-Face Identity Tracking (Spatial-Temporal centroid matching)
        tracks = self.tracker.update(detected_landmarks_list, timestamp, frame_rgb.shape)

        # 3. Extract Blendshapes and Pose for each face
        raw_observations = []
        tracked_faces_info = []

        for idx, (face_id, is_primary, bbox) in enumerate(tracks):
            lms = detected_landmarks_list[idx]

            # Blendshapes dictionary (52 ARKit keys)
            bs_dict = {}
            if result.face_blendshapes and idx < len(result.face_blendshapes):
                for cat in result.face_blendshapes[idx]:
                    bs_dict[cat.category_name] = round(float(cat.score), 4)

            # Transformation matrix and Head Pose Euler
            mat = None
            if result.facial_transformation_matrixes and idx < len(result.facial_transformation_matrixes):
                mat = result.facial_transformation_matrixes[idx]
            euler, trans = self.matrix_to_euler_angles(mat)

            # Raw Observation Layer
            raw_obs = RawObservation(
                landmarks=lms,
                blendshapes=bs_dict,
                transformation_matrix=mat if mat is not None else np.eye(4),
                head_pose_euler=euler,
                head_translation=trans
            )
            raw_observations.append(raw_obs)

            # Quality Evaluation
            quality = self.quality_evaluator.evaluate(
                face_id=face_id,
                image_rgb=frame_rgb,
                bbox=bbox,
                landmarks=lms,
                head_pose_euler=euler,
                timestamp=timestamp
            )

            # Update mouth dynamics for speaker attribution (pose-invariant inner-lip aperture + blendshapes)
            mouth_vel = self.speaker_engine.update_face_dynamics(face_id, bs_dict, timestamp, landmarks=lms)

            tracked_faces_info.append({
                "face_id": face_id,
                "is_primary": is_primary,
                "bbox": bbox,
                "quality": quality,
                "mouth_velocity": mouth_vel,
                "hit_count": self.tracker.tracks[face_id].hit_count
            })

        # 4. Cross-Modal Speaker Attribution & Eligibility Grading
        if audio_frame is not None:
            default_audio = audio_frame
        elif hasattr(self.speaker_engine, "latest_audio") and self.speaker_engine.latest_audio is not None:
            la = self.speaker_engine.latest_audio
            default_audio = AudioFrameData(
                timestamp=timestamp,
                energy_rms=la.energy_rms,
                is_speech=la.is_speech,
                vad_confidence=la.vad_confidence,
                is_narration=la.is_narration,
                active_speaker_face_id=la.active_speaker_face_id
            )
        else:
            default_audio = AudioFrameData(timestamp=timestamp, energy_rms=0.0, is_speech=False, vad_confidence=0.0)

        attribution_results = self.speaker_engine.attribute_speakers(tracked_faces_info, default_audio)

        # 5. Temporal Filtering (One Euro Filter) & Final Packaging
        output_frames = []
        for idx, (face_info, (role, speaker_prob, elig_level)) in enumerate(zip(tracked_faces_info, attribution_results)):
            face_id = face_info["face_id"]
            raw_obs = raw_observations[idx]

            # Generate Clean Observation Layer
            clean_obs = self.temporal_filter.filter_observation(face_id, raw_obs, timestamp)

            face_frame = TrackedFaceFrame(
                timestamp=timestamp,
                face_id=face_id,
                is_primary=face_info["is_primary"],
                bbox=face_info["bbox"],
                raw=raw_obs,
                clean=clean_obs,
                quality=face_info["quality"],
                role=role,
                speaker_probability=speaker_prob,
                eligibility_level=elig_level,
                is_narration=default_audio.is_narration
            )
            output_frames.append(face_frame)

        return output_frames
