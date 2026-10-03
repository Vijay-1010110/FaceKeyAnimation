"""Cross-modal audio-visual speaker attribution and multi-level eligibility classifier."""

import collections
import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from src.schema import FaceRole, EligibilityLevel, AudioFrameData, QualityMetrics
from src.config import SpeakerAttributionConfig


class VoiceActivityDetector:
    """Lightweight in-memory Voice Activity Detection using energy and spectral flux."""

    def __init__(self, sample_rate: int = 16000, energy_threshold: float = 0.008):
        self.sample_rate = sample_rate
        self.base_energy_threshold = energy_threshold
        self.noise_floor: float = 0.004
        self.last_speech_time: float = -1.0
        self.speech_hangover_sec: float = 0.38

    def process_chunk(self, audio_samples: np.ndarray, timestamp: float) -> AudioFrameData:
        """Compute short-time RMS energy and speech activity decision with adaptive noise floor."""
        if audio_samples is None or len(audio_samples) == 0:
            return AudioFrameData(timestamp=timestamp, energy_rms=0.0, is_speech=False, vad_confidence=0.0)

        # 1. Root-Mean-Square Energy
        rms = float(np.sqrt(np.mean(np.square(audio_samples))))

        # 2. Zero-Crossing Rate
        signs = np.sign(audio_samples)
        zcr = float(np.mean(np.abs(np.diff(signs))) / 2.0) if len(signs) > 1 else 0.0

        # Adapt noise floor slowly during quiet periods
        if rms < self.noise_floor * 1.5 or self.noise_floor == 0.004:
            self.noise_floor = 0.96 * self.noise_floor + 0.04 * rms

        dynamic_threshold = max(self.base_energy_threshold, self.noise_floor * 2.2)

        # Raw speech detection
        raw_speech = (rms >= dynamic_threshold) and (0.01 <= zcr <= 0.45)
        if raw_speech:
            self.last_speech_time = timestamp

        # Apply speech hangover to maintain temporal coherence across syllables
        is_speech = raw_speech or (self.last_speech_time > 0 and (timestamp - self.last_speech_time) <= self.speech_hangover_sec)
        vad_confidence = min(1.0, rms / max(1e-4, dynamic_threshold * 2.5)) if is_speech else 0.0

        return AudioFrameData(
            timestamp=timestamp,
            energy_rms=round(rms, 5),
            is_speech=is_speech,
            vad_confidence=round(vad_confidence, 3)
        )


class SpeakerAttributionEngine:
    """Attributes active speech to visible face identities using lip-speech correlation.
    Supports narration detection, active listener classification, and eligibility grading.
    """

    def __init__(self, config: SpeakerAttributionConfig):
        self.config = config
        self.vad = VoiceActivityDetector(energy_threshold=config.vad_energy_threshold)
        # Sliding buffer of (timestamp, is_speech, energy)
        self.audio_history: collections.deque = collections.deque(maxlen=30)
        # Per-face sliding buffer: face_id -> deque of (timestamp, total_mouth, mouth_velocity)
        self.face_mouth_history: Dict[int, collections.deque] = {}
        self.last_blendshapes: Dict[int, Tuple[float, float]] = {}  # face_id -> (timestamp, total_mouth)
        self.latest_audio: AudioFrameData = AudioFrameData(0.0, 0.0, False, 0.0)

        # Single-Face Speaking Gate state
        self.is_gate_active: bool = False
        self.gate_status: str = "STANDBY: WAITING FOR SINGLE SPEAKER"
        self.speaking_streak: int = 0
        self.last_verified_speech_time: float = -10.0

    def reset(self):
        self.audio_history.clear()
        self.face_mouth_history.clear()
        self.last_blendshapes.clear()
        self.latest_audio = AudioFrameData(0.0, 0.0, False, 0.0)
        self.is_gate_active = False
        self.gate_status = "STANDBY: WAITING FOR SINGLE SPEAKER"
        self.speaking_streak = 0
        self.last_verified_speech_time = -10.0

    def update_audio(self, audio_samples: Optional[np.ndarray], timestamp: float) -> AudioFrameData:
        """Process new audio frame and store in sliding buffer."""
        audio_frame = self.vad.process_chunk(audio_samples, timestamp)
        self.latest_audio = audio_frame
        self.audio_history.append((timestamp, audio_frame.is_speech, audio_frame.energy_rms))
        return audio_frame

    def update_face_dynamics(
        self,
        face_id: int,
        blendshapes: Dict[str, float],
        timestamp: float,
        landmarks: Optional[np.ndarray] = None
    ) -> float:
        """Calculate pose-invariant mouth articulation dynamics for a face."""
        jaw_open = blendshapes.get("jawOpen", 0.0)
        mouth_pucker = blendshapes.get("mouthPucker", 0.0)
        mouth_stretch = blendshapes.get("mouthStretchLeft", 0.0) + blendshapes.get("mouthStretchRight", 0.0)

        # Compute inner-lip vertical aperture normalized by vertical face height (invariant to head nodding/translation)
        lip_aperture = jaw_open
        if landmarks is not None and len(landmarks) >= 468:
            face_h = float(np.linalg.norm(landmarks[10, :2] - landmarks[152, :2]))
            if face_h > 1e-4:
                inner_lip_dist = float(np.linalg.norm(landmarks[13, :2] - landmarks[14, :2]))
                lip_aperture = inner_lip_dist / face_h

        total_mouth = (jaw_open * 0.60) + (lip_aperture * 1.40) + (mouth_pucker * 0.25) + (mouth_stretch * 0.15)

        velocity = 0.0
        if face_id in self.last_blendshapes:
            prev_t, prev_mouth = self.last_blendshapes[face_id]
            dt = timestamp - prev_t
            if dt > 1e-4:
                velocity = abs(total_mouth - prev_mouth) / dt

        self.last_blendshapes[face_id] = (timestamp, total_mouth)

        if face_id not in self.face_mouth_history:
            self.face_mouth_history[face_id] = collections.deque(maxlen=30)
        self.face_mouth_history[face_id].append((timestamp, total_mouth, velocity))

        return velocity

    def attribute_speakers(
        self,
        tracked_faces_info: List[Dict[str, Any]],
        current_audio: AudioFrameData
    ) -> List[Tuple[FaceRole, float, EligibilityLevel]]:
        """Evaluate Single-Face Speaking Gate, speaker probability, semantic role, and dataset eligibility."""
        results = []
        is_speech_active = current_audio.is_speech
        num_faces = len(tracked_faces_info)

        # -----------------------------------------------------------------
        # CASE 1: 0 Faces Detected -> Gate OFF
        # -----------------------------------------------------------------
        if num_faces == 0:
            self.is_gate_active = False
            self.speaking_streak = 0
            self.gate_status = "STANDBY: NO FACE DETECTED (0 Faces)"
            current_audio.is_narration = is_speech_active
            current_audio.active_speaker_face_id = None
            return []

        # Calculate mouth articulation metrics for each face
        face_metrics = []
        for face_info in tracked_faces_info:
            fid = face_info["face_id"]
            velocity = face_info.get("mouth_velocity", 0.0)
            quality: QualityMetrics = face_info["quality"]
            hit_count: int = face_info.get("hit_count", 1)

            history = self.face_mouth_history.get(fid, [])
            recent = [item for item in history if (current_audio.timestamp - item[0]) <= self.config.sliding_window_sec]
            mean_vel = float(np.mean([item[2] for item in recent])) if recent else velocity
            mouth_vals = [item[1] for item in recent] if recent else [0.0]
            mouth_range = float(max(mouth_vals) - min(mouth_vals)) if len(mouth_vals) >= 2 else 0.0
            max_mouth = float(max(mouth_vals)) if mouth_vals else 0.0

            motion_score = min(1.0, mean_vel / max(1e-4, self.config.lip_motion_velocity_threshold * 2.2))
            face_metrics.append((fid, motion_score, mean_vel, mouth_range, max_mouth, quality, hit_count))

        # -----------------------------------------------------------------
        # CASE 2: Multiple Faces Detected (> 1 Face) -> Reject / Pause when strict_single_face_only is True
        # -----------------------------------------------------------------
        if getattr(self.config, "strict_single_face_only", True) and num_faces > 1:
            self.is_gate_active = False
            self.speaking_streak = 0
            self.gate_status = f"PAUSED: {num_faces} FACES DETECTED (Only 1 Face Allowed)"
            current_audio.is_narration = False
            current_audio.active_speaker_face_id = None
            if hasattr(self, "latest_audio") and self.latest_audio:
                self.latest_audio.is_narration = False
                self.latest_audio.active_speaker_face_id = None

            for fid, motion_score, _, _, _, quality, hit_count in face_metrics:
                # Cap eligibility at LEVEL_2 so multi-face frames are never saved as clean single-speaker data
                level = EligibilityLevel.LEVEL_0_DETECTED
                if hit_count >= 5:
                    level = EligibilityLevel.LEVEL_1_TRACKABLE
                if level >= EligibilityLevel.LEVEL_1_TRACKABLE and quality.rejection_reason is None:
                    level = EligibilityLevel.LEVEL_2_GEOMETRICALLY_USABLE
                results.append((FaceRole.LISTENER, round(motion_score * 0.25, 3), level))
            return results

        # -----------------------------------------------------------------
        # CASE 3: Exactly 1 Face Detected -> Careful True-Speaking Verification
        # -----------------------------------------------------------------
        fid, motion_score, mean_vel, mouth_range, max_mouth, quality, hit_count = face_metrics[0]

        # Careful articulation test:
        # 1. Velocity must exceed lip_motion_velocity_threshold
        # 2. Mouth aperture must dynamically oscillate (mouth_range >= min_mouth_articulation_range)
        # 3. Peak opening must be non-trivial (max_mouth >= 0.018)
        min_range = getattr(self.config, "min_mouth_articulation_range", 0.012)
        is_mouth_articulating = (
            mean_vel >= self.config.lip_motion_velocity_threshold
            and mouth_range >= min_range
            and max_mouth >= 0.018
        )

        if is_mouth_articulating:
            self.speaking_streak = min(4, self.speaking_streak + 1)
            if self.speaking_streak >= 2:
                self.last_verified_speech_time = current_audio.timestamp
        else:
            self.speaking_streak = 0

        # Conversational Timing: Syllables -> Between Words -> Conversational Pause -> Silence
        inter_syllable_hold = getattr(self.config, "inter_syllable_hold_sec", 0.25)
        inter_word_hold = getattr(self.config, "inter_word_hold_sec", 0.55)
        conversational_pause_limit = getattr(self.config, "conversational_pause_sec", 1.80)
        time_since_speech = max(0.0, current_audio.timestamp - self.last_verified_speech_time)
        self.time_since_speech = time_since_speech

        is_narration = False
        active_speaker_fid: Optional[int] = None

        if (is_mouth_articulating and self.speaking_streak >= 2) or time_since_speech <= inter_syllable_hold:
            # 1. Active Syllable Articulation
            role = FaceRole.SPEAKER
            speaker_prob = round(float(min(0.99, max(0.75, motion_score))), 3)
            active_speaker_fid = fid
            self.is_gate_active = True
            self.conversational_state = "ACTIVE_SPEECH"
            self.gate_status = "ACTIVE & RECORDING: 1 FACE + VERIFIED SPEAKING"
            level = EligibilityLevel.LEVEL_4_SPEAKER_PAIRED if quality.is_valid else EligibilityLevel.LEVEL_2_GEOMETRICALLY_USABLE

        elif time_since_speech <= inter_word_hold:
            # 2. Natural Inter-Word Transition / Coarticulation / Micro-Pause between words (RECORDED as SPEAKER!)
            role = FaceRole.SPEAKER
            speaker_prob = round(float(max(0.70, motion_score * 0.85)), 3)
            active_speaker_fid = fid
            self.is_gate_active = True
            self.conversational_state = "BETWEEN_WORDS"
            self.gate_status = f"ACTIVE & RECORDING: BETWEEN WORDS / COARTICULATION ({time_since_speech:.2f}s)"
            level = EligibilityLevel.LEVEL_4_SPEAKER_PAIRED if quality.is_valid else EligibilityLevel.LEVEL_2_GEOMETRICALLY_USABLE

        elif time_since_speech <= conversational_pause_limit:
            # 3. Natural Conversational Pause / Thinking / Breathing between sentences / clauses (RECORDED as SPEAKER_PAUSE!)
            role = FaceRole.SPEAKER_PAUSE
            speaker_prob = round(float(max(0.60, motion_score * 0.75)), 3)
            active_speaker_fid = fid
            self.is_gate_active = True
            self.conversational_state = "CONVERSATIONAL_PAUSE"
            self.gate_status = f"ACTIVE & RECORDING: CONVERSATIONAL PAUSE / IDLE ({time_since_speech:.1f}s / {conversational_pause_limit:.1f}s)"
            level = EligibilityLevel.LEVEL_3_ANIMATION_QUALITY if quality.is_valid else EligibilityLevel.LEVEL_2_GEOMETRICALLY_USABLE

        else:
            # 4. Prolonged Silence (> 1.8s) / Turn Finished / Narration
            role = FaceRole.LISTENER if quality.is_valid else FaceRole.UNKNOWN
            speaker_prob = round(float(motion_score * 0.15), 3)
            self.is_gate_active = False
            self.conversational_state = "SILENCE"
            level = EligibilityLevel.LEVEL_2_GEOMETRICALLY_USABLE if quality.is_valid else EligibilityLevel.LEVEL_0_DETECTED
            if is_speech_active:
                is_narration = True
                self.gate_status = "PAUSED: 1 FACE SILENT (Off-Screen Narration / Voiceover)"
            else:
                self.gate_status = f"PAUSED: 1 FACE SILENT (> {conversational_pause_limit:.1f}s) / NOT SPEAKING"

        results.append((role, speaker_prob, level))

        current_audio.is_narration = is_narration
        current_audio.active_speaker_face_id = active_speaker_fid
        if hasattr(self, "latest_audio") and self.latest_audio:
            self.latest_audio.is_narration = is_narration
            self.latest_audio.active_speaker_face_id = active_speaker_fid

        return results
