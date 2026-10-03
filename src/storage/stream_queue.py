"""Batch YouTube Stream Queue & Auto-Resume Management System.
Monitors 'sessions/youtubeURLtoProcess.txt', maintains persistent queue state in
'data/youtube_batch_queue.json', prevents duplicate processing, and tracks
progress toward the Phase 1 milestone (100 Hours) and long-term scale (1,000 Hours).
"""

import os
import re
import json
import time
import threading
from typing import Optional, Dict, Any, List, Tuple

from src.storage.stream_registry import StreamRegistry, extract_youtube_id, normalize_stream_url
from src.storage.dataset_tracker import DatasetReadinessTracker


class StreamBatchQueue:
    """Manages continuous ingestion and persistent state for batch YouTube stream processing."""

    def __init__(self, project_root: Optional[str] = None, urls_file: Optional[str] = None):
        self._lock = threading.RLock()
        if project_root is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.project_root = project_root
        self.data_dir = os.path.join(self.project_root, "data")
        os.makedirs(self.data_dir, exist_ok=True)

        self.db_path = os.path.join(self.data_dir, "youtube_batch_queue.json")
        self.urls_file = urls_file or os.path.join(self.project_root, "sessions", "youtubeURLtoProcess.txt")
        os.makedirs(os.path.dirname(self.urls_file), exist_ok=True)
        if not os.path.exists(self.urls_file):
            with open(self.urls_file, "w", encoding="utf-8") as f:
                f.write("# FaceKey Batch YouTube Stream URLs\n# Paste YouTube links here (one per line). New URLs are detected automatically!\n\n")

        self.registry = StreamRegistry(self.project_root)
        self.dataset_tracker = DatasetReadinessTracker(os.path.join(self.project_root, "sessions"))
        self.state = self._load()

    def _load(self) -> Dict[str, Any]:
        """Load persistent queue state from JSON or initialize fresh."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "version": "1.0",
            "last_synced": time.strftime("%Y-%m-%d %H:%M:%S"),
            "items": {},          # key -> item dict
            "order": [],          # list of keys preserving insertion order
            "stats": {
                "total_completed": 0,
                "total_frames": 0,
                "total_duration_sec": 0.0
            }
        }

    def _save(self):
        """Save persistent queue state to JSON atomically."""
        with self._lock:
            self.state["last_synced"] = time.strftime("%Y-%m-%d %H:%M:%S")
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            tmp_path = self.db_path + ".tmp"
            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(self.state, f, indent=2, ensure_ascii=False)
                if os.path.exists(self.db_path):
                    os.replace(tmp_path, self.db_path)
                else:
                    os.rename(tmp_path, self.db_path)
            except Exception as e:
                print(f"[!] Warning saving queue state: {e}")

    def sync_from_file(self) -> Tuple[int, int, int]:
        """Read 'sessions/youtubeURLtoProcess.txt', discover new links, and append to queue.
        Returns: (new_added_count, already_completed_count, pending_count)
        """
        if not os.path.exists(self.urls_file):
            return 0, 0, 0

        try:
            with open(self.urls_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:
            return 0, 0, 0

        new_added = 0
        items = self.state.setdefault("items", {})
        order = self.state.setdefault("order", [])
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")

        for line in lines:
            raw = line.strip()
            if not raw or raw.startswith("#") or raw.startswith("//"):
                continue

            norm_url, yt_id = normalize_stream_url(raw)
            key = yt_id if yt_id else norm_url

            if key not in items:
                # Check if already recorded previously in StreamRegistry
                existing_record = self.registry.find_entry(raw)
                status = "already_recorded" if existing_record else "pending"
                title = existing_record.get("title", "Pending Resolution") if existing_record else "Pending Resolution"

                item = {
                    "key": key,
                    "raw_url": raw,
                    "canonical_url": norm_url,
                    "video_id": yt_id or "N/A",
                    "title": title,
                    "status": status,
                    "added_at": now_str,
                    "started_at": None,
                    "completed_at": existing_record.get("last_recorded_at") if existing_record else None,
                    "session_id": existing_record.get("sessions", [None])[-1] if existing_record else None,
                    "frame_count": existing_record.get("total_frames", 0) if existing_record else 0,
                    "duration_seconds": existing_record.get("total_duration_seconds", 0.0) if existing_record else 0.0,
                    "error_message": None
                }
                items[key] = item
                order.append(key)
                if status == "pending":
                    new_added += 1

        self._save()

        # Count current state
        completed = sum(1 for it in items.values() if it.get("status") in ("completed", "already_recorded"))
        pending = sum(1 for it in items.values() if it.get("status") in ("pending", "in_progress"))
        return new_added, completed, pending

    def reset_interrupted_to_pending(self) -> int:
        """Reset any 'in_progress' items back to 'pending' to allow clean resumption."""
        reset_count = 0
        for it in self.state.get("items", {}).values():
            if it.get("status") == "in_progress":
                it["status"] = "pending"
                reset_count += 1
        if reset_count > 0:
            self._save()
        return reset_count

    def purge_corrupted_completed(self, min_duration_sec: float = 15.0) -> int:
        """Reset any items falsely marked completed/failed with duration < 15s back to pending."""
        reset_count = 0
        for it in self.state.get("items", {}).values():
            status = it.get("status")
            dur = it.get("duration_seconds", 0.0)
            if status in ("completed", "failed") and dur < min_duration_sec:
                it["status"] = "pending"
                it["completed_at"] = None
                it["session_id"] = None
                it["frame_count"] = 0
                it["duration_seconds"] = 0.0
                it["error_message"] = None
                reset_count += 1
        if reset_count > 0:
            self._save()
        return reset_count

    def get_next_pending(self, exclude_keys: Optional[set] = None) -> Optional[Dict[str, Any]]:
        """Return the next pending item in insertion order."""
        with self._lock:
            items = self.state.get("items", {})
            for key in self.state.get("order", []):
                if exclude_keys and key in exclude_keys:
                    continue
                it = items.get(key)
                if it and it.get("status") == "pending":
                    return it
            return None

    def mark_in_progress(self, key: str, title: Optional[str] = None):
        """Mark item as actively processing."""
        item = self.state.get("items", {}).get(key)
        if item:
            item["status"] = "in_progress"
            item["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            if title and title != "Pending Resolution":
                item["title"] = title
            self._save()

    def mark_completed(
        self,
        key: str,
        session_id: str,
        frame_count: int,
        duration_seconds: float,
        title: Optional[str] = None
    ):
        """Mark item as successfully processed and record statistics."""
        item = self.state.get("items", {}).get(key)
        if item:
            item["status"] = "completed"
            item["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            item["session_id"] = session_id
            item["frame_count"] = frame_count
            item["duration_seconds"] = round(duration_seconds, 2)
            if title and title != "Pending Resolution":
                item["title"] = title
            
            stats = self.state.setdefault("stats", {})
            stats["total_completed"] = stats.get("total_completed", 0) + 1
            stats["total_frames"] = stats.get("total_frames", 0) + frame_count
            stats["total_duration_sec"] = round(stats.get("total_duration_sec", 0.0) + duration_seconds, 2)
            self._save()

    def mark_failed(self, key: str, error_message: str):
        """Mark item as failed with error details."""
        item = self.state.get("items", {}).get(key)
        if item:
            item["status"] = "failed"
            item["error_message"] = error_message
            item["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self._save()

    def get_progress_summary(self) -> Dict[str, Any]:
        """Generate comprehensive milestone statistics for 100 Hours Target and Queue."""
        items = self.state.get("items", {})
        total_items = len(items)
        completed = sum(1 for it in items.values() if it.get("status") in ("completed", "already_recorded"))
        pending = sum(1 for it in items.values() if it.get("status") == "pending")
        in_progress = sum(1 for it in items.values() if it.get("status") == "in_progress")
        failed = sum(1 for it in items.values() if it.get("status") == "failed")

        # Global dataset report across all disk sessions
        self.dataset_tracker.scan_sessions()
        rep = self.dataset_tracker.compute_cumulative_report()
        total_dataset_hours = rep.total_duration_seconds / 3600.0
        total_speech_hours = rep.total_active_speech_sec / 3600.0

        phase1_target_hours = 100.0
        phase1_pct = min(100.0, (total_dataset_hours / phase1_target_hours) * 100.0)
        remaining_phase1_hours = max(0.0, phase1_target_hours - total_dataset_hours)

        scale_target_hours = 1000.0
        scale_pct = min(100.0, (total_dataset_hours / scale_target_hours) * 100.0)

        # Storage estimations based on empirical ~490.6 MB/hour rate
        mb_per_hour = 490.6
        est_gb_100h = (mb_per_hour * 100.0) / 1024.0   # ~47.9 GB
        est_gb_1000h = (mb_per_hour * 1000.0) / 1024.0 # ~479.1 GB
        current_gb_used = (rep.total_size_bytes) / (1024.0 * 1024.0 * 1024.0)

        return {
            "queue_total": total_items,
            "queue_completed": completed,
            "queue_pending": pending,
            "queue_in_progress": in_progress,
            "queue_failed": failed,
            "total_dataset_sessions": rep.total_sessions_count,
            "total_dataset_frames": rep.total_samples_frames,
            "total_dataset_hours": round(total_dataset_hours, 2),
            "total_speech_hours": round(total_speech_hours, 2),
            "current_gb_used": round(current_gb_used, 2),
            "phase1_target_hours": phase1_target_hours,
            "phase1_progress_pct": round(phase1_pct, 2),
            "phase1_remaining_hours": round(remaining_phase1_hours, 2),
            "scale_target_hours": scale_target_hours,
            "scale_progress_pct": round(scale_pct, 2),
            "estimated_gb_100h": round(est_gb_100h, 1),
            "estimated_gb_1000h": round(est_gb_1000h, 1),
            "readiness_label": rep.readiness_label,
            "urls_file": self.urls_file
        }
