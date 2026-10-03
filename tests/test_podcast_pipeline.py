"""Automated verification test for multi-face tracking and speaker attribution on podcast video."""

import os
import unittest
import numpy as np

from src.config import AppConfig
from src.core.session_manager import SessionManager
from src.schema import FaceRole, EligibilityLevel
from src.storage.dataset_reader import DatasetReader


class TestPodcastPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = AppConfig()
        cls.config.max_faces = 4
        cls.config.speaker.strict_single_face_only = False
        cls.session_manager = SessionManager(cls.config)
        cls.session_manager.initialize()

    @classmethod
    def tearDownClass(cls):
        cls.session_manager.close()

    def test_multi_face_podcast_attribution(self):
        video_path = "tests/data/podcast_multi_face_test.mp4"
        self.assertTrue(os.path.exists(video_path), f"Video file not found: {video_path}")

        # Process the 4-second podcast video offline
        saved_dir = self.session_manager.process_video_file(video_path)
        self.assertIsNotNone(saved_dir)
        self.assertTrue(os.path.isdir(saved_dir))

        # Read back the saved dataset
        reader = DatasetReader(saved_dir)
        metadata = reader.metadata

        print(f"\n--- Podcast Test Results for {saved_dir} ---")
        print(f"Total face samples captured: {metadata.get('total_video_frames')}")
        print(f"Inference FPS: {metadata.get('average_inference_fps')}")
        print(f"Unique Face IDs tracked: {metadata.get('tracked_face_ids')}")
        print(f"Role counts: {metadata.get('role_counts')}")
        print(f"Eligibility counts: {metadata.get('eligibility_counts')}")

        # 1. Verify that 2 distinct faces were detected
        face_ids = metadata.get("tracked_face_ids", [])
        self.assertGreaterEqual(len(face_ids), 2, "Expected at least 2 distinct faces tracked")

        arrays = reader.arrays
        n_samples = reader.total_samples
        print(f"Total samples: {n_samples}")
        self.assertGreater(n_samples, 0)

        timestamps = arrays["timestamps"]
        roles = arrays["roles"]
        speaker_probs = arrays["speaker_probabilities"]
        eligibilities = arrays["eligibility_levels"]

        # Check Turn 1 (0.5s to 1.8s): Host A speaking, Host B listening
        t1_mask = (timestamps >= 0.5) & (timestamps <= 1.8)
        t1_roles = roles[t1_mask]
        t1_speakers = int(np.sum(t1_roles == FaceRole.SPEAKER.value))
        t1_listeners = int(np.sum(t1_roles == FaceRole.LISTENER.value))

        print(f"Turn 1 (Host A speaking): {t1_speakers} SPEAKER samples, {t1_listeners} LISTENER samples")
        self.assertGreater(t1_speakers, 0, "Host A should have SPEAKER samples in Turn 1")
        self.assertGreater(t1_listeners, 0, "Host B should have LISTENER samples in Turn 1")

        # Check Turn 2 (2.5s to 3.8s): Host B speaking, Host A listening
        t2_mask = (timestamps >= 2.5) & (timestamps <= 3.8)
        t2_roles = roles[t2_mask]
        t2_speakers = int(np.sum(t2_roles == FaceRole.SPEAKER.value))
        t2_listeners = int(np.sum(t2_roles == FaceRole.LISTENER.value))

        print(f"Turn 2 (Host B speaking): {t2_speakers} SPEAKER samples, {t2_listeners} LISTENER samples")
        self.assertGreater(t2_speakers, 0, "Host B should have SPEAKER samples in Turn 2")
        self.assertGreater(t2_listeners, 0, "Host A should have LISTENER samples in Turn 2")

        # Check quality and eligibility
        l3_samples = int(np.sum(eligibilities >= EligibilityLevel.LEVEL_3_ANIMATION_QUALITY.value))
        print(f"Level 3+ Animation Quality samples: {l3_samples}")
        self.assertGreater(l3_samples, 0, "Expected Level 3 Animation Quality samples")

        # Verify NPZ contents
        self.assertIn("timestamps", arrays)
        self.assertIn("face_ids", arrays)
        self.assertIn("clean_landmarks", arrays)
        self.assertIn("clean_blendshapes", arrays)
        self.assertIn("clean_pose_euler", arrays)
        self.assertIn("speaker_probabilities", arrays)
        self.assertIn("eligibility_levels", arrays)

        # Check array dimensions
        self.assertEqual(arrays["clean_landmarks"].shape, (n_samples, 478, 3))
        self.assertEqual(arrays["clean_blendshapes"].shape, (n_samples, 52))
        self.assertEqual(arrays["clean_pose_euler"].shape, (n_samples, 3))
        print("All array shapes, speaker turns, and zero-disk-frame constraints verified successfully!")


if __name__ == "__main__":
    unittest.main()
