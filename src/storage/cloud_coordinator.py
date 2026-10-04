"""Distributed Cloud Coordinator & Atomic Locking System for Parallel Multi-Worker Processing.
Enables Google Colab, Kaggle, and Local PC workers to run simultaneous data collection
and training without race conditions, redundant downloads, or lost progress.

Key Features:
  1. Distributed File Locks (Drive/locks/<video_id>.lock.json) to prevent duplicate work.
  2. Heartbeat Monitoring & Auto-Reclaim of Stale Locks when a session expires or crashes.
  3. Shared Global Completed Registry (Drive/data/shared_completed_sources.json).
  4. Worker Tagging & Non-Colliding Dataset Chunks for 5 TB Google Drive storage.
"""

import os
import sys
import json
import time
import socket
import threading
from typing import Optional, Dict, Any, List, Tuple

from src.storage.stream_registry import extract_youtube_id, normalize_stream_url


class CloudCoordinator:
    """Coordinates parallel data collection across multiple cloud environments (Colab, Kaggle, Local).
    Maintains atomic locks and a shared completed database on Google Drive.
    """

    def __init__(
        self,
        drive_folder: str,
        worker_id: Optional[str] = None,
        stale_timeout_sec: int = 1200  # 20 minutes heartbeat expiration
    ):
        self.drive_folder = os.path.abspath(drive_folder)
        self.stale_timeout_sec = stale_timeout_sec

        # Determine worker ID
        if worker_id:
            self.worker_id = worker_id
        else:
            host = socket.gethostname()[:8]
            pid = os.getpid()
            if os.path.exists("/content") and "google.colab" in sys.modules:
                self.worker_id = f"colab_{host}_{pid}"
            elif os.path.exists("/kaggle"):
                self.worker_id = f"kaggle_{host}_{pid}"
            else:
                self.worker_id = f"local_{host}_{pid}"

        # Drive subdirectories
        self.locks_dir = os.path.join(self.drive_folder, "locks")
        self.data_dir = os.path.join(self.drive_folder, "data")
        self.chunks_dir = os.path.join(self.drive_folder, "chunks")
        self.checkpoints_dir = os.path.join(self.drive_folder, "checkpoints")

        os.makedirs(self.locks_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.chunks_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.completed_db_path = os.path.join(self.data_dir, "shared_completed_sources.json")
        self.active_lock_key: Optional[str] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop_heartbeat = threading.Event()

    def get_completed_keys(self) -> Dict[str, Any]:
        """Read central shared completed registry from Google Drive."""
        if os.path.exists(self.completed_db_path):
            try:
                with open(self.completed_db_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"version": "1.0", "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"), "completed": {}}

    def purge_corrupted_completed(self, min_duration_sec: float = 15.0) -> int:
        """Purge any falsely completed entries (e.g. from early crashes with duration < 15s)."""
        db = self.get_completed_keys()
        completed = db.get("completed", {})
        purged = 0
        valid_completed = {}
        for k, v in completed.items():
            dur = v.get("duration_seconds", 0.0)
            frames = v.get("frame_count", 0)
            if dur < min_duration_sec or frames < 50:
                purged += 1
            else:
                valid_completed[k] = v
        if purged > 0:
            db["completed"] = valid_completed
            db["total_completed"] = len(valid_completed)
            db["total_hours"] = round(sum(v.get("duration_seconds", 0.0) for v in valid_completed.values()) / 3600.0, 3)
            db["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            try:
                with open(self.completed_db_path, "w", encoding="utf-8") as f:
                    json.dump(db, f, indent=2, ensure_ascii=False)
                print(f"[+] Purged {purged} falsely marked completed video(s) (< {min_duration_sec}s) from Google Drive registry!")
            except Exception as e:
                print(f"[!] Warning purging corrupted completed entries: {e}")
        return purged

    def is_already_completed(self, key_or_url: str) -> bool:
        """Check if a video key or URL has already been processed by any worker."""
        _, yt_id = normalize_stream_url(key_or_url)
        key = yt_id or key_or_url.strip()

        db = self.get_completed_keys()
        completed = db.get("completed", {})
        if key in completed:
            dur = completed[key].get("duration_seconds", 0.0)
            if dur < 15.0:
                return False
            return True
        for comp_key, comp_val in completed.items():
            if comp_val.get("video_id") == key or comp_val.get("canonical_url") == key_or_url:
                dur = comp_val.get("duration_seconds", 0.0)
                if dur < 15.0:
                    return False
                return True
        return False

    def get_lock_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Read lock file if it exists."""
        lock_file = os.path.join(self.locks_dir, f"{key}.lock.json")
        if os.path.exists(lock_file):
            try:
                with open(lock_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def try_claim_url(
        self,
        url_or_id: str,
        title: Optional[str] = None
    ) -> Tuple[bool, str, Optional[str]]:
        """Attempts to claim a URL exclusively for this worker.
        Returns: (success: bool, reason_or_status: str, key: Optional[str])
        """
        canonical_url, yt_id = normalize_stream_url(url_or_id)
        key = yt_id if yt_id else url_or_id.strip()

        # 1. Check if already completed globally
        if self.is_already_completed(key):
            return False, "already_completed", key

        # 2. Check existing lock
        lock_file = os.path.join(self.locks_dir, f"{key}.lock.json")
        if os.path.exists(lock_file):
            lock_info = self.get_lock_info(key)
            if lock_info:
                other_worker = lock_info.get("worker_id", "unknown")
                last_hb = lock_info.get("last_heartbeat", 0.0)
                age_sec = time.time() - last_hb

                # Check if lock belongs to current worker (resume after local pause)
                if other_worker == self.worker_id:
                    self._start_heartbeat(key)
                    return True, "reclaimed_own_lock", key

                # Check if lock is active
                if age_sec < self.stale_timeout_sec:
                    return False, f"locked_by_{other_worker} (active {age_sec/60.0:.1f}m ago)", key

                # Stale lock detected! The previous worker crashed or timed out
                print(f"\n[!] Stale lock detected for '{key}' by '{other_worker}' (inactive {age_sec/60.0:.1f}m).")
                print(f"[*] Reclaiming abandoned work for worker '{self.worker_id}'...")

        # 3. Write atomic lock
        lock_payload = {
            "key": key,
            "url": canonical_url,
            "video_id": yt_id or "N/A",
            "title": title or "Pending Resolution",
            "worker_id": self.worker_id,
            "claimed_at": time.time(),
            "claimed_at_str": time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_heartbeat": time.time(),
            "last_heartbeat_str": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        tmp_lock = lock_file + f".tmp_{self.worker_id}"
        try:
            with open(tmp_lock, "w", encoding="utf-8") as f:
                json.dump(lock_payload, f, indent=2)
            os.replace(tmp_lock, lock_file)
        except Exception as e:
            return False, f"lock_write_error: {e}", key

        # Verify claim
        verified = self.get_lock_info(key)
        if verified and verified.get("worker_id") == self.worker_id:
            self._start_heartbeat(key)
            return True, "claimed", key
        else:
            return False, "claim_race_lost", key

    def _start_heartbeat(self, key: str):
        """Start background heartbeat thread to keep lock alive while recording."""
        self._stop_heartbeat.set()
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=2.0)

        self.active_lock_key = key
        self._stop_heartbeat.clear()

        def heartbeat_loop():
            lock_file = os.path.join(self.locks_dir, f"{key}.lock.json")
            while not self._stop_heartbeat.is_set():
                time.sleep(30.0)  # Heartbeat every 30 seconds
                if self._stop_heartbeat.is_set():
                    break
                try:
                    if os.path.exists(lock_file):
                        with open(lock_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        data["last_heartbeat"] = time.time()
                        data["last_heartbeat_str"] = time.strftime("%Y-%m-%d %H:%M:%S")
                        tmp = lock_file + ".hb_tmp"
                        with open(tmp, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2)
                        os.replace(tmp, lock_file)
                except Exception:
                    pass

        self._heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    def _stop_heartbeat_thread(self):
        """Stop active heartbeat thread safely."""
        try:
            if hasattr(self, "_stop_heartbeat") and self._stop_heartbeat is not None:
                self._stop_heartbeat.set()
            if getattr(self, "_heartbeat_thread", None) is not None and self._heartbeat_thread.is_alive():
                if self._heartbeat_thread != threading.current_thread():
                    self._heartbeat_thread.join(timeout=1.0)
        except Exception:
            pass
        finally:
            self._heartbeat_thread = None
            self.active_lock_key = None

    def release_lock(self, key: str):
        """Release active lock without marking as completed (e.g., if interrupted or failed)."""
        self._stop_heartbeat_thread()
        lock_file = os.path.join(self.locks_dir, f"{key}.lock.json")
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
            except Exception:
                pass

    def mark_completed(
        self,
        key: str,
        canonical_url: str,
        title: str,
        session_id: str,
        frame_count: int,
        duration_seconds: float
    ):
        """Mark video completed in the shared Drive registry and release lock."""
        self._stop_heartbeat_thread()

        # Update shared completed DB with retry loop
        for attempt in range(5):
            db = self.get_completed_keys()
            completed = db.setdefault("completed", {})
            completed[key] = {
                "key": key,
                "canonical_url": canonical_url,
                "video_id": key,
                "title": title,
                "session_id": session_id,
                "worker_id": self.worker_id,
                "frame_count": frame_count,
                "duration_seconds": round(duration_seconds, 2),
                "duration_minutes": round(duration_seconds / 60.0, 2),
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            db["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            db["total_completed"] = len(completed)
            db["total_hours"] = round(sum(v.get("duration_seconds", 0.0) for v in completed.values()) / 3600.0, 3)

            tmp_path = self.completed_db_path + f".tmp_{self.worker_id}"
            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(db, f, indent=2, ensure_ascii=False)
                os.replace(tmp_path, self.completed_db_path)
                break
            except Exception:
                time.sleep(0.5)

        # Remove lock file
        lock_file = os.path.join(self.locks_dir, f"{key}.lock.json")
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
            except Exception:
                pass

    def clean_stale_locks(self) -> int:
        """Scan locks directory and remove stale locks abandoned by crashed sessions."""
        cleaned = 0
        now = time.time()
        if not os.path.exists(self.locks_dir):
            return 0
        for fname in os.listdir(self.locks_dir):
            if fname.endswith(".lock.json"):
                fpath = os.path.join(self.locks_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    last_hb = data.get("last_heartbeat", 0.0)
                    w_id = data.get("worker_id", "")
                    if (now - last_hb) > self.stale_timeout_sec or (w_id and w_id.startswith(self.worker_id)):
                        os.remove(fpath)
                        cleaned += 1
                        print(f"[!] Cleaned stale lock: {fname} (worker: {w_id}, idle {(now - last_hb)/60:.1f}m)")
                except Exception:
                    pass
        return cleaned
