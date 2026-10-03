"""Unit Tests for Distributed Cloud Coordinator, Atomic Locking, Stale Lock Reclamation,
and Multi-Worker Parallel Processing (Colab + Kaggle).
"""

import os
import sys
import json
import time
import shutil
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.storage.cloud_coordinator import CloudCoordinator
from src.storage.cloud_sync import CloudDriveSync


class TestDistributedCloudCoordinator(unittest.TestCase):

    def setUp(self):
        self.test_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_cloud_env"))
        self.drive_dir = os.path.join(self.test_root, "Drive_FaceKeyDataset")
        os.makedirs(self.drive_dir, exist_ok=True)

        self.coordinator_colab = CloudCoordinator(
            drive_folder=self.drive_dir,
            worker_id="colab_worker_test",
            stale_timeout_sec=2  # Short timeout for testing
        )

        self.coordinator_kaggle = CloudCoordinator(
            drive_folder=self.drive_dir,
            worker_id="kaggle_worker_test",
            stale_timeout_sec=2
        )

    def tearDown(self):
        self.coordinator_colab.release_lock("test_vid_1")
        self.coordinator_kaggle.release_lock("test_vid_1")
        if os.path.exists(self.test_root):
            shutil.rmtree(self.test_root, ignore_errors=True)

    def test_parallel_locking_and_duplicate_prevention(self):
        """Worker 1 claims URL; Worker 2 must be prevented from duplicate processing."""
        test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        key = "dQw4w9WgXcQ"

        # Colab worker claims URL
        claimed, reason, k = self.coordinator_colab.try_claim_url(test_url, title="Rick Astley")
        self.assertTrue(claimed)
        self.assertEqual(reason, "claimed")
        self.assertEqual(k, key)

        # Kaggle worker attempts to claim same URL while active
        k_claimed, k_reason, _ = self.coordinator_kaggle.try_claim_url(test_url, title="Rick Astley")
        self.assertFalse(k_claimed)
        self.assertIn("colab_worker_test", k_reason)

    def test_stale_lock_recovery(self):
        """When a worker session crashes/expires, another worker auto-reclaims the abandoned lock."""
        test_url = "https://www.youtube.com/watch?v=APCWOBUZNjc"
        key = "APCWOBUZNjc"

        # Colab claims lock
        claimed, _, _ = self.coordinator_colab.try_claim_url(test_url, title="Test Lecture")
        self.assertTrue(claimed)

        # Stop heartbeat and artificially age the lock file
        self.coordinator_colab._stop_heartbeat_thread()
        lock_file = os.path.join(self.coordinator_colab.locks_dir, f"{key}.lock.json")
        with open(lock_file, "r") as f:
            data = json.load(f)
        data["last_heartbeat"] = time.time() - 100  # Aged past 2-second timeout
        with open(lock_file, "w") as f:
            json.dump(data, f)

        # Kaggle worker should detect stale lock and successfully reclaim it
        k_claimed, k_reason, _ = self.coordinator_kaggle.try_claim_url(test_url, title="Test Lecture")
        self.assertTrue(k_claimed)
        self.assertEqual(k_reason, "claimed")

        # Verify ownership changed to Kaggle
        info = self.coordinator_kaggle.get_lock_info(key)
        self.assertEqual(info["worker_id"], "kaggle_worker_test")

    def test_completion_and_global_deduplication(self):
        """Once completed, neither Colab nor Kaggle can process the URL again."""
        test_url = "https://www.youtube.com/watch?v=MjduMAagEDk"
        key = "MjduMAagEDk"

        claimed, _, _ = self.coordinator_colab.try_claim_url(test_url)
        self.assertTrue(claimed)

        # Mark completed
        self.coordinator_colab.mark_completed(
            key=key,
            canonical_url=test_url,
            title="Completed Stream",
            session_id="session_test_001",
            frame_count=1800,
            duration_seconds=60.0
        )

        # Lock file should be deleted
        self.assertIsNone(self.coordinator_colab.get_lock_info(key))

        # Check global completed status
        self.assertTrue(self.coordinator_colab.is_already_completed(key))
        self.assertTrue(self.coordinator_kaggle.is_already_completed(key))

        # Attempt to claim on Kaggle should immediately return already_completed
        k_claimed, k_reason, _ = self.coordinator_kaggle.try_claim_url(test_url)
        self.assertFalse(k_claimed)
        self.assertEqual(k_reason, "already_completed")

    def test_non_colliding_chunks(self):
        """Parallel workers generating chunks produce distinct worker-tagged filenames."""
        sync = CloudDriveSync(project_root=self.test_root, drive_folder=self.drive_dir)

        # Create dummy session
        s_dir = os.path.join(self.test_root, "sessions", "session_dummy_01")
        os.makedirs(s_dir, exist_ok=True)
        with open(os.path.join(s_dir, "metadata.json"), "w") as f:
            json.dump({"total_video_frames": 100, "duration_seconds": 3.3}, f)

        chunk1 = sync.pack_sessions_into_chunk(
            sessions_dir=os.path.join(self.test_root, "sessions"),
            worker_tag="colab_worker"
        )
        self.assertIsNotNone(chunk1)
        self.assertIn("colab_worker", os.path.basename(chunk1))

        # Second dummy session for Kaggle
        s_dir2 = os.path.join(self.test_root, "sessions", "session_dummy_02")
        os.makedirs(s_dir2, exist_ok=True)
        with open(os.path.join(s_dir2, "metadata.json"), "w") as f:
            json.dump({"total_video_frames": 200, "duration_seconds": 6.6}, f)

        chunk2 = sync.pack_sessions_into_chunk(
            sessions_dir=os.path.join(self.test_root, "sessions"),
            worker_tag="kaggle_worker"
        )
        self.assertIsNotNone(chunk2)
        self.assertIn("kaggle_worker", os.path.basename(chunk2))
        self.assertNotEqual(chunk1, chunk2)


if __name__ == "__main__":
    unittest.main()
