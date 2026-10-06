"""Structured dataset persistence saving clean animation data without temporary frame files."""

import json
import os
import time
from typing import List, Dict, Any, Optional
import numpy as np

from src.schema import TrackedFaceFrame, AudioFrameData, SessionMetadata, EligibilityLevel, FaceRole


class DatasetWriter:
    """Writes completed session datasets into compact, columnar binary formats with metadata.
    Zero disk files are created during live capture; persistence occurs only on finalization.
    """

    def __init__(self, base_output_dir: str):
        self.base_output_dir = base_output_dir

    def write_session(
        self,
        session_id: str,
        mode: str,
        source_description: str,
        face_frames: List[TrackedFaceFrame],
        audio_frames: List[AudioFrameData],
        session_start_time: float,
        session_end_time: float,
        capture_fps: float,
        inference_fps: float,
        telemetry: Optional[Dict[str, Any]] = None
    ) -> str:
        """Serialize complete session data to disk."""
        session_dir = os.path.join(self.base_output_dir, f"session_{session_id}")
        os.makedirs(session_dir, exist_ok=True)

        duration = max(0.0, session_end_time - session_start_time)
        total_samples = len(face_frames)

        # 1. Prepare Columnar Numpy Arrays
        tracked_ids = sorted(list(set(f.face_id for f in face_frames)))
        primary_id = None
        for f in reversed(face_frames):
            if f.is_primary:
                primary_id = f.face_id
                break

        # Standard 52 ARKit blendshape keys
        bs_keys = sorted(face_frames[0].raw.blendshapes.keys()) if face_frames and face_frames[0].raw.blendshapes else []

        timestamps = np.zeros(total_samples, dtype=np.float32)
        face_ids = np.zeros(total_samples, dtype=np.int32)
        is_primary = np.zeros(total_samples, dtype=bool)
        bboxes = np.zeros((total_samples, 4), dtype=np.int32)
        raw_landmarks = np.zeros((total_samples, 478, 3), dtype=np.float32)
        clean_landmarks = np.zeros((total_samples, 478, 3), dtype=np.float32)
        raw_bs = np.zeros((total_samples, len(bs_keys)), dtype=np.float32)
        clean_bs = np.zeros((total_samples, len(bs_keys)), dtype=np.float32)
        raw_euler = np.zeros((total_samples, 3), dtype=np.float32)
        clean_euler = np.zeros((total_samples, 3), dtype=np.float32)
        raw_trans = np.zeros((total_samples, 3), dtype=np.float32)
        clean_trans = np.zeros((total_samples, 3), dtype=np.float32)
        quality_scores = np.zeros(total_samples, dtype=np.float32)
        blur_scores = np.zeros(total_samples, dtype=np.float32)
        is_valid = np.zeros(total_samples, dtype=bool)
        eligibility = np.zeros(total_samples, dtype=np.int32)
        speaker_probs = np.zeros(total_samples, dtype=np.float32)
        roles = []

        eligibility_counts: Dict[str, int] = {e.name: 0 for e in EligibilityLevel}
        role_counts: Dict[str, int] = {r.name: 0 for r in FaceRole}

        for i, frame in enumerate(face_frames):
            timestamps[i] = frame.timestamp
            face_ids[i] = frame.face_id
            is_primary[i] = frame.is_primary
            bboxes[i] = frame.bbox
            raw_landmarks[i] = frame.raw.landmarks
            clean_landmarks[i] = frame.clean.landmarks

            if bs_keys:
                raw_bs[i] = [frame.raw.blendshapes.get(k, 0.0) for k in bs_keys]
                clean_bs[i] = [frame.clean.blendshapes.get(k, 0.0) for k in bs_keys]

            raw_euler[i] = frame.raw.head_pose_euler
            clean_euler[i] = frame.clean.head_pose_euler
            raw_trans[i] = frame.raw.head_translation
            clean_trans[i] = frame.clean.head_translation

            quality_scores[i] = frame.quality.composite_score
            blur_scores[i] = frame.quality.blur_score
            is_valid[i] = frame.quality.is_valid
            eligibility[i] = int(frame.eligibility_level)
            speaker_probs[i] = frame.speaker_probability
            roles.append(frame.role.value if hasattr(frame.role, "value") else str(frame.role))

            eligibility_counts[frame.eligibility_level.name] += 1
            role_name = frame.role.name if hasattr(frame.role, "name") else str(frame.role)
            if role_name in role_counts:
                role_counts[role_name] += 1

        # 2. Extract and Align Audio Features (64-band Log-Mel filterbanks)
        audio_features = np.zeros((total_samples, 64), dtype=np.float32)
        audio_energy = np.zeros(total_samples, dtype=np.float32)
        audio_speech_prob = np.zeros(total_samples, dtype=np.float32)

        if audio_frames:
            aud_timestamps = np.array([a.timestamp for a in audio_frames], dtype=np.float32)
            for i, frame in enumerate(face_frames):
                if len(audio_frames) == total_samples:
                    a_match = audio_frames[i]
                else:
                    closest_idx = int(np.argmin(np.abs(aud_timestamps - frame.timestamp)))
                    a_match = audio_frames[closest_idx]

                audio_energy[i] = a_match.energy_rms
                audio_speech_prob[i] = a_match.vad_confidence
                if a_match.spectral_features is not None and len(a_match.spectral_features) == 64:
                    audio_features[i] = a_match.spectral_features

        # 3. Save Columnar Binary File (face_motion.npz)
        npz_path = os.path.join(session_dir, "face_motion.npz")
        np.savez_compressed(
            npz_path,
            timestamps=timestamps,
            face_ids=face_ids,
            is_primary=is_primary,
            bboxes=bboxes,
            raw_landmarks=raw_landmarks,
            clean_landmarks=clean_landmarks,
            blendshape_names=np.array(bs_keys),
            raw_blendshapes=raw_bs,
            clean_blendshapes=clean_bs,
            raw_pose_euler=raw_euler,
            clean_pose_euler=clean_euler,
            raw_pose_trans=raw_trans,
            clean_pose_trans=clean_trans,
            quality_scores=quality_scores,
            blur_scores=blur_scores,
            is_valid=is_valid,
            eligibility_levels=eligibility,
            speaker_probabilities=speaker_probs,
            roles=np.array(roles),
            audio_features=audio_features,
            audio_energy=audio_energy,
            audio_speech_prob=audio_speech_prob
        )

        # 3. Save Audio Features JSON
        if audio_frames:
            audio_json_path = os.path.join(session_dir, "audio_features.json")
            audio_data = [
                {
                    "timestamp": round(a.timestamp, 4),
                    "energy_rms": round(a.energy_rms, 5),
                    "is_speech": a.is_speech,
                    "vad_confidence": round(a.vad_confidence, 3)
                }
                for a in audio_frames
            ]
            with open(audio_json_path, "w", encoding="utf-8") as f:
                json.dump(audio_data, f, indent=2)

        # 4. Save Metadata JSON
        meta = SessionMetadata(
            session_id=session_id,
            mode=mode,
            source_description=source_description,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            duration_seconds=round(duration, 3),
            total_video_frames=total_samples,
            total_audio_frames=len(audio_frames),
            tracked_face_ids=tracked_ids,
            primary_face_id=primary_id,
            eligibility_counts=eligibility_counts,
            role_counts=role_counts,
            average_capture_fps=round(capture_fps, 1),
            average_inference_fps=round(inference_fps, 1),
            hardware_telemetry=telemetry or {}
        )

        meta_json_path = os.path.join(session_dir, "metadata.json")
        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump(meta.__dict__, f, indent=2)

        return session_dir
