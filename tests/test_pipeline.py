"""Unit and integration tests for facial motion acquisition system."""

import os
import shutil
import tempfile
import time
import unittest
import numpy as np

from src.config import AppConfig, FaceThresholds, HysteresisConfig, SpeakerAttributionConfig, TemporalFilterConfig
from src.schema import RawObservation, CleanObservation, EligibilityLevel, FaceRole, AudioFrameData
from src.core.temporal_filter import FaceTemporalFilter, OneEuroFilterArray
from src.core.tracker import MultiFaceTracker
from src.core.quality import QualityEvaluator
from src.core.speaker_attribution import SpeakerAttributionEngine, VoiceActivityDetector
from src.core.ring_buffer import PreRollRingBuffer, BoundedFrameQueue
from src.storage.dataset_writer import DatasetWriter
from src.storage.dataset_reader import DatasetReader
from src.schema import TrackedFaceFrame, QualityMetrics


class TestTemporalFilter(unittest.TestCase):
    def test_one_euro_filter_jitter_reduction(self):
        filt = OneEuroFilterArray(min_cutoff=1.0, beta=0.007)
        # Constant signal with additive high frequency noise
        t0 = 0.0
        val = np.array([0.5, 0.5])
        outputs = []
        for i in range(50):
            t = t0 + i * (1.0 / 30.0)
            noise = (0.05 if i % 2 == 0 else -0.05)
            noisy_val = val + noise
            out = filt.filter(noisy_val, t)
            outputs.append(out)

        # Output variance should be significantly smaller than noisy input variance
        noisy_var = np.var([0.55, 0.45])
        filt_var = np.var([o[0] for o in outputs[10:]])
        self.assertLess(filt_var, noisy_var)

    def test_face_temporal_filter_preserves_raw_data(self):
        conf = TemporalFilterConfig()
        f_filter = FaceTemporalFilter(conf)
        raw_bs = {"jawOpen": 0.45, "mouthSmileLeft": 0.12}
        raw_lms = np.zeros((478, 3), dtype=np.float32)
        raw_obs = RawObservation(
            landmarks=raw_lms,
            blendshapes=raw_bs,
            transformation_matrix=np.eye(4),
            head_pose_euler=(10.0, 5.0, 0.0),
            head_translation=(0.0, 0.0, 0.5)
        )

        clean = f_filter.filter_observation(face_id=0, raw=raw_obs, timestamp=0.0)
        self.assertTrue(clean.is_observed)
        self.assertFalse(clean.is_reconstructed)
        # Ensure raw observation blendshape is untouched
        self.assertEqual(raw_obs.blendshapes["jawOpen"], 0.45)
        self.assertIn("jawOpen", clean.blendshapes)


class TestMultiFaceTracker(unittest.TestCase):
    def test_persistent_identity_and_primary_selection(self):
        tracker = MultiFaceTracker()
        img_shape = (480, 640)

        # Frame 1: Two faces detected (Face 0: large center, Face 1: small right)
        face0_lms = np.zeros((478, 3), dtype=np.float32)
        face0_lms[:, 0] = 0.5  # center x
        face0_lms[:, 1] = 0.5  # center y
        face0_lms[0, 0] = 0.35
        face0_lms[1, 0] = 0.65
        face0_lms[0, 1] = 0.35
        face0_lms[1, 1] = 0.65

        face1_lms = np.zeros((478, 3), dtype=np.float32)
        face1_lms[:, 0] = 0.85
        face1_lms[:, 1] = 0.5
        face1_lms[0, 0] = 0.82
        face1_lms[1, 0] = 0.88
        face1_lms[0, 1] = 0.45
        face1_lms[1, 1] = 0.55

        tracks1 = tracker.update([face0_lms, face1_lms], timestamp=0.0, image_shape=img_shape)
        self.assertEqual(len(tracks1), 2)
        self.assertEqual(tracks1[0][0], 0)  # Face ID 0
        self.assertEqual(tracks1[1][0], 1)  # Face ID 1
        self.assertTrue(tracks1[0][1])      # Face 0 should be primary (larger, central)
        self.assertFalse(tracks1[1][1])     # Face 1 should not be primary

        # Frame 2: Faces shifted slightly - IDs must be preserved
        face0_lms_shift = face0_lms.copy() + 0.01
        face1_lms_shift = face1_lms.copy() - 0.01
        tracks2 = tracker.update([face0_lms_shift, face1_lms_shift], timestamp=0.033, image_shape=img_shape)
        self.assertEqual(tracks2[0][0], 0)  # Preserved Face ID 0
        self.assertEqual(tracks2[1][0], 1)  # Preserved Face ID 1


class TestQualityEvaluator(unittest.TestCase):
    def test_size_and_blur_evaluation(self):
        f_thresh = FaceThresholds(min_width_px=60, min_height_px=60, min_blur_laplacian=20.0)
        h_conf = HysteresisConfig(quality_enter_threshold=0.60, quality_exit_threshold=0.40)
        evaluator = QualityEvaluator(f_thresh, h_conf)

        # Synthetic image (640x480) with sharp high contrast patch
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        lms = np.zeros((478, 3), dtype=np.float32)
        bbox_valid = (100, 100, 150, 150)

        metrics = evaluator.evaluate(
            face_id=0,
            image_rgb=img,
            bbox=bbox_valid,
            landmarks=lms,
            head_pose_euler=(5.0, 5.0, 0.0),
            timestamp=0.0
        )
        self.assertGreater(metrics.face_width_px, 60)
        self.assertGreater(metrics.blur_score, 0.0)
        self.assertIsNone(metrics.rejection_reason)


class TestSpeakerAttribution(unittest.TestCase):
    def test_vad_and_speaker_attribution(self):
        vad = VoiceActivityDetector(energy_threshold=0.01)
        # Pure silence
        silence = np.zeros(1600, dtype=np.float32)
        silent_frame = vad.process_chunk(silence, 0.0)
        self.assertFalse(silent_frame.is_speech)

        # Synthesized audio with energy
        speech_synth = np.sin(np.linspace(0, 100 * np.pi, 1600)).astype(np.float32) * 0.1
        speech_frame = vad.process_chunk(speech_synth, 0.1)
        self.assertTrue(speech_frame.is_speech)

        cfg = SpeakerAttributionConfig(strict_single_face_only=False)
        engine = SpeakerAttributionEngine(cfg)
        # Track 2 faces: Face 0 talking (rapid jawOpen changes), Face 1 stationary listener
        engine.update_face_dynamics(0, {"jawOpen": 0.1}, 0.0)
        engine.update_face_dynamics(0, {"jawOpen": 0.6}, 0.05)
        engine.update_face_dynamics(1, {"jawOpen": 0.0}, 0.0)
        engine.update_face_dynamics(1, {"jawOpen": 0.0}, 0.05)

        q_dummy = QualityMetrics(0.85, 50.0, True, 120, 120, 0.1, 0.9, 0.1, True)
        tracked_info = [
            {"face_id": 0, "mouth_velocity": 10.0, "quality": q_dummy, "hit_count": 10},
            {"face_id": 1, "mouth_velocity": 0.0, "quality": q_dummy, "hit_count": 10},
        ]
        results = engine.attribute_speakers(tracked_info, speech_frame)
        self.assertEqual(results[0][0], FaceRole.SPEAKER)
        self.assertEqual(results[0][2], EligibilityLevel.LEVEL_4_SPEAKER_PAIRED)
        self.assertEqual(results[1][0], FaceRole.LISTENER)
        self.assertEqual(results[1][2], EligibilityLevel.LEVEL_3_ANIMATION_QUALITY)


class TestStorageRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_writer_and_reader_roundtrip(self):
        writer = DatasetWriter(self.tmp_dir)

        # Build sample TrackedFaceFrame
        raw_obs = RawObservation(
            landmarks=np.ones((478, 3), dtype=np.float32) * 0.5,
            blendshapes={"jawOpen": 0.7, "eyeBlinkLeft": 0.1},
            transformation_matrix=np.eye(4),
            head_pose_euler=(1.0, 2.0, 3.0),
            head_translation=(0.1, 0.2, 0.3)
        )
        clean_obs = CleanObservation(
            landmarks=np.ones((478, 3), dtype=np.float32) * 0.5,
            blendshapes={"jawOpen": 0.69, "eyeBlinkLeft": 0.1},
            head_pose_euler=(1.0, 2.0, 3.0),
            head_translation=(0.1, 0.2, 0.3)
        )
        q = QualityMetrics(0.9, 80.0, True, 100, 100, 0.08, 0.95, 0.05, True)
        frame = TrackedFaceFrame(
            timestamp=1.234,
            face_id=0,
            is_primary=True,
            bbox=(50, 50, 100, 100),
            raw=raw_obs,
            clean=clean_obs,
            quality=q,
            role=FaceRole.SPEAKER,
            speaker_probability=0.92,
            eligibility_level=EligibilityLevel.LEVEL_4_SPEAKER_PAIRED
        )

        audio_sample = AudioFrameData(timestamp=1.234, energy_rms=0.03, is_speech=True, vad_confidence=0.88)

        session_path = writer.write_session(
            session_id="test_001",
            mode="manual",
            source_description="Test Stream",
            face_frames=[frame],
            audio_frames=[audio_sample],
            session_start_time=100.0,
            session_end_time=105.0,
            capture_fps=30.0,
            inference_fps=30.0
        )

        # Read back using DatasetReader
        reader = DatasetReader(session_path)
        self.assertEqual(reader.total_samples, 1)
        sample = reader.get_sample_at(0)
        self.assertEqual(sample["face_id"], 0)
        self.assertEqual(sample["eligibility_level"], int(EligibilityLevel.LEVEL_4_SPEAKER_PAIRED))
        self.assertEqual(sample["role"], "SPEAKER")
        self.assertAlmostEqual(sample["clean_blendshapes"]["jawOpen"], 0.69, places=2)

        # Test JSONL export
        jsonl_path = os.path.join(self.tmp_dir, "export.jsonl")
        exported_count = reader.export_to_jsonl(jsonl_path, min_eligibility=EligibilityLevel.LEVEL_3_ANIMATION_QUALITY)
        self.assertEqual(exported_count, 1)
        self.assertTrue(os.path.exists(jsonl_path))


if __name__ == "__main__":
    unittest.main()
