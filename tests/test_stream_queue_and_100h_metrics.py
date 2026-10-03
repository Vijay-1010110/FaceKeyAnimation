"""Verification tests for StreamBatchQueue and 100-Hour training milestone metrics."""

import os
import sys
import tempfile
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.storage.stream_queue import StreamBatchQueue
from src.storage.dataset_tracker import DatasetReadinessTracker

def test_queue_sync_and_resume():
    # Create temporary files to test queue sync without touching user data
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_txt = os.path.join(tmp_dir, "test_urls.txt")
        with open(test_txt, "w", encoding="utf-8") as f:
            f.write("""
            # Initial YouTube Batch List
            https://www.youtube.com/watch?v=MjduMAagEDk
            https://youtu.be/dQw4w9WgXcQ
            """)

        queue = StreamBatchQueue(project_root=".", urls_file=test_txt)
        queue.db_path = os.path.join(tmp_dir, "test_queue.json")
        queue.state = queue._load()

        # 1. Initial Sync
        new_added, completed, pending = queue.sync_from_file()
        print(f"[TEST] Sync 1 -> New: {new_added}, Completed: {completed}, Pending: {pending}")

        # MjduMAagEDk is in registry, so it should be detected as already_recorded!
        # dQw4w9WgXcQ should be pending
        assert "MjduMAagEDk" in queue.state["items"], "MjduMAagEDk must be in queue"
        assert "dQw4w9WgXcQ" in queue.state["items"], "dQw4w9WgXcQ must be in queue"
        assert queue.state["items"]["MjduMAagEDk"]["status"] == "already_recorded", "MjduMAagEDk should be already_recorded"
        assert queue.state["items"]["dQw4w9WgXcQ"]["status"] == "pending", "dQw4w9WgXcQ should be pending"
        print("[PASS] Duplicate detection in batch queue correctly identified registered video!")

        # 2. Get next pending
        next_item = queue.get_next_pending()
        assert next_item is not None and next_item["key"] == "dQw4w9WgXcQ"
        print(f"[PASS] Next pending item correctly retrieved: {next_item['key']}")

        # 3. Simulate processing and interruption
        queue.mark_in_progress("dQw4w9WgXcQ", title="Test Video 1")
        assert queue.state["items"]["dQw4w9WgXcQ"]["status"] == "in_progress"
        
        # Test resume functionality: reset_interrupted_to_pending
        resumed = queue.reset_interrupted_to_pending()
        assert resumed == 1, f"Expected 1 resumed, got {resumed}"
        assert queue.state["items"]["dQw4w9WgXcQ"]["status"] == "pending"
        print("[PASS] Resume mechanism successfully restored in-progress video to pending!")

        # 4. Continuous Ingestion Test: Add another URL to file while running
        with open(test_txt, "a", encoding="utf-8") as f:
            f.write("\nhttps://www.youtube.com/watch?v=9bZkp7q19f0\n")

        new_added2, completed2, pending2 = queue.sync_from_file()
        print(f"[TEST] Sync 2 -> New: {new_added2}, Completed: {completed2}, Pending: {pending2}")
        assert "9bZkp7q19f0" in queue.state["items"], "Newly added URL must be in queue"
        assert new_added2 == 1, "Should detect exactly 1 new URL"
        assert queue.state["items"]["9bZkp7q19f0"]["status"] == "pending"
        print("[PASS] Continuous file addition smoothly picked up new link without resetting queue!")

def test_100h_milestone_and_storage_metrics():
    tracker = DatasetReadinessTracker("sessions")
    report = tracker.compute_cumulative_report()

    print("\n" + "=" * 65)
    print(f"Total Dataset Duration   : {report.total_duration_seconds / 3600.0:.2f} hours ({report.total_duration_seconds / 60.0:.1f} mins)")
    print(f"Total Clean Speech       : {report.total_active_speech_sec / 3600.0:.2f} hours")
    print(f"Phase 1 Target           : {report.phase1_target_hours:.0f} Hours")
    print(f"Phase 1 Progress         : {report.phase1_progress_pct:.2f}%")
    print(f"Scale Target             : {report.scale_target_hours:.0f} Hours")
    print(f"Estimated 100h Size      : {report.estimated_gb_for_100h:.1f} GB")
    print(f"Estimated 1,000h Size    : {report.estimated_gb_for_1000h:.1f} GB")
    print(f"Status Indicator         : {report.readiness_label}")
    print("=" * 65)

    assert report.phase1_target_hours == 100.0
    assert report.scale_target_hours == 1000.0
    assert report.estimated_gb_for_100h > 40.0 and report.estimated_gb_for_100h < 55.0
    assert report.estimated_gb_for_1000h > 400.0 and report.estimated_gb_for_1000h < 550.0
    print("[PASS] 100-Hour and 1,000-Hour milestones and storage estimations verified!")

if __name__ == "__main__":
    test_queue_sync_and_resume()
    test_100h_milestone_and_storage_metrics()
    print("\nALL BATCH QUEUE & 100H METRIC TESTS PASSED!")
