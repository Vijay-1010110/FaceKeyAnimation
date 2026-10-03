"""Verification test for StreamRegistry duplicate detection and between-words attribution."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.storage.stream_registry import StreamRegistry, extract_youtube_id, normalize_stream_url
from src.config import AppConfig
from src.core.speaker_attribution import SpeakerAttributionEngine
from src.schema import AudioFrameData, FaceRole, EligibilityLevel, TrackedFaceFrame, RawObservation, CleanObservation, QualityMetrics
import numpy as np

def test_registry():
    reg = StreamRegistry(".")
    test_urls = [
        "https://www.youtube.com/watch?v=MjduMAagEDk",
        "https://youtu.be/MjduMAagEDk",
        "MjduMAagEDk",
        "https://www.youtube.com/shorts/MjduMAagEDk",
        "https://www.youtube.com/watch?v=MjduMAagEDk&feature=youtu.be"
    ]

    for u in test_urls:
        entry = reg.find_entry(u)
        assert entry is not None, f"Failed to match: {u}"
        assert entry["video_id"] == "MjduMAagEDk", f"ID mismatch for {u}"
        print(f"[PASS] Duplicate Match: {u:55s} -> {entry['title']}")

    # Test an unrecorded URL
    unrecorded = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert reg.find_entry(unrecorded) is None, "Should not match unrecorded URL"
    print(f"[PASS] Unrecorded URL correctly not found: {unrecorded}")

def test_between_words_attribution():
    config = AppConfig()
    engine = SpeakerAttributionEngine(config.speaker)

    # Helper to create dummy face frame
    def make_face(face_id=0, lip_vel=0.0):
        raw = RawObservation(
            landmarks=np.zeros((478, 3)),
            blendshapes={"mouthOpen": 0.3},
            transformation_matrix=np.eye(4),
            head_pose_euler=(0.0, 0.0, 0.0),
            head_translation=(0.0, 0.0, 0.0)
        )
        clean = CleanObservation(
            landmarks=np.zeros((478, 3)),
            blendshapes={"mouthOpen": 0.3},
            head_pose_euler=(0.0, 0.0, 0.0),
            head_translation=(0.0, 0.0, 0.0)
        )
        q = QualityMetrics(
            composite_score=0.9,
            blur_score=100.0,
            is_sharp=True,
            face_width_px=100,
            face_height_px=100,
            area_ratio=0.15,
            pose_quality=0.95,
            temporal_jitter=0.01,
            is_valid=True
        )
        return TrackedFaceFrame(
            face_id=face_id,
            timestamp=0.0,
            is_primary=True,
            bbox=(100, 100, 200, 200),
            raw=raw,
            clean=clean,
            quality=q
        )

    # 1. Warm up with active speech
    q = QualityMetrics(composite_score=0.9, blur_score=100.0, is_sharp=True, face_width_px=100, face_height_px=100, area_ratio=0.15, pose_quality=0.95, temporal_jitter=0.01, is_valid=True)
    for t in [0.0, 0.05, 0.10, 0.15, 0.20]:
        audio = AudioFrameData(timestamp=t, energy_rms=0.1, is_speech=True, vad_confidence=0.9)
        face_info = {
            "face_id": 0,
            "bbox": (100, 100, 200, 200),
            "quality": q,
            "is_primary": True,
            "mouth_velocity": 0.45,
            "hit_count": 10
        }
        engine.face_mouth_history[0] = [(t - 0.03, 0.15, 0.45), (t, 0.35, 0.45)]
        engine.attribute_speakers([face_info], audio)

    assert engine.conversational_state == "ACTIVE_SPEECH", f"Expected ACTIVE_SPEECH, got {engine.conversational_state}"
    assert engine.is_gate_active == True, "Gate should be active"
    print(f"[PASS] Active speech state: {engine.conversational_state} (Gate: {engine.is_gate_active})")

    # 2. Simulate 0.35s inter-word pause (bilabial stop, coarticulation hold < 0.55s, > 0.25s)
    # Audio is temporarily quiet (stop closure)
    t_closure = 0.55
    audio_closure = AudioFrameData(timestamp=t_closure, energy_rms=0.005, is_speech=False, vad_confidence=0.1)
    face_info_closure = {
        "face_id": 0,
        "bbox": (100, 100, 200, 200),
        "quality": q,
        "is_primary": True,
        "mouth_velocity": 0.02,
        "hit_count": 10
    }
    engine.face_mouth_history[0] = [(t_closure - 0.03, 0.05, 0.02), (t_closure, 0.05, 0.02)]
    results = engine.attribute_speakers([face_info_closure], audio_closure)

    assert engine.conversational_state == "BETWEEN_WORDS", f"Expected BETWEEN_WORDS, got {engine.conversational_state}"
    assert engine.is_gate_active == True, "Gate should remain active during between-words coarticulation"
    role, prob, elig = results[0]
    assert role == FaceRole.SPEAKER, f"Role should be SPEAKER, got {role}"
    assert elig == EligibilityLevel.LEVEL_4_SPEAKER_PAIRED, f"Eligibility should be LEVEL_4, got {elig}"
    print(f"[PASS] Inter-word coarticulation state: {engine.conversational_state} (Gate: {engine.is_gate_active}, Role: {role}, Level: {elig.name})")

if __name__ == "__main__":
    test_registry()
    print("-" * 65)
    test_between_words_attribution()
    print("=" * 65)
    print("ALL TESTS PASSED SUCCESSFULLY!")
