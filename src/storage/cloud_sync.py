"""Cloud Storage, Google Drive Chunking & Manifest Manager.
Handles packaging recorded facial animation sessions into compressed .tar.gz chunks,
syncing to Google Drive, and extracting chunks to local SSD for ultra-fast GPU training.
"""

import os
import sys
import tarfile
import json
import time
import glob
import shutil
from typing import Optional, Dict, Any, List, Tuple


class CloudDriveSync:
    """Manages chunked dataset persistence and Google Drive integration.
    Prevents local disk exhaustion by packaging sessions into chunks and streaming to Drive.
    """

    def __init__(self, project_root: Optional[str] = None, drive_folder: Optional[str] = None):
        if project_root is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.project_root = os.path.abspath(project_root)

        # Detect Google Colab environment
        self.is_colab = os.path.exists("/content") and "google.colab" in sys.modules

        # Drive root folder path (default: /content/drive/MyDrive/FaceKeyDataset on Colab or local cloud dir)
        if drive_folder:
            self.drive_folder = os.path.abspath(drive_folder)
        elif self.is_colab:
            self.drive_folder = "/content/drive/MyDrive/FaceKeyDataset"
        else:
            self.drive_folder = os.path.join(self.project_root, "cloud_drive_backup")

        self.chunks_dir = os.path.join(self.drive_folder, "chunks")
        self.checkpoints_dir = os.path.join(self.drive_folder, "checkpoints")
        self.manifest_path = os.path.join(self.drive_folder, "dataset_manifest.json")
        self.queue_drive_path = os.path.join(self.drive_folder, "youtubeURLtoProcess.txt")

    def mount_google_drive(self) -> bool:
        """Mount Google Drive inside Google Colab environment."""
        if not self.is_colab:
            os.makedirs(self.chunks_dir, exist_ok=True)
            os.makedirs(self.checkpoints_dir, exist_ok=True)
            return True

        try:
            from google.colab import drive
            print("[*] Mounting Google Drive to '/content/drive'...")
            drive.mount("/content/drive")
            os.makedirs(self.chunks_dir, exist_ok=True)
            os.makedirs(self.checkpoints_dir, exist_ok=True)
            print("[✓] Google Drive successfully mounted!")
            return True
        except Exception as e:
            print(f"[!] Warning: Could not mount Google Drive automatically ({e})")
            return False

    def load_manifest(self) -> Dict[str, Any]:
        """Load manifest of all packaged dataset chunks from Google Drive."""
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "version": "1.0",
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_chunks": 0,
            "total_sessions": 0,
            "total_frames": 0,
            "total_duration_hours": 0.0,
            "total_size_mb": 0.0,
            "chunks": []
        }

    def save_manifest(self, manifest: Dict[str, Any]):
        """Save dataset manifest to Google Drive."""
        manifest["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        os.makedirs(os.path.dirname(self.manifest_path), exist_ok=True)
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        if os.path.exists(self.manifest_path):
            os.replace(tmp, self.manifest_path)
        else:
            os.rename(tmp, self.manifest_path)

    def pack_sessions_into_chunk(
        self,
        sessions_dir: str,
        chunk_name: Optional[str] = None,
        max_sessions_per_chunk: int = 15,
        purge_local_after_pack: bool = False
    ) -> Optional[str]:
        """Packs local session directories into a compressed .tar.gz chunk on Google Drive.
        If purge_local_after_pack is True, removes local session folders to free up disk space!
        """
        os.makedirs(self.chunks_dir, exist_ok=True)
        manifest = self.load_manifest()

        # Find sessions not already listed in manifest
        already_packed_sessions = set()
        for c in manifest.get("chunks", []):
            already_packed_sessions.update(c.get("session_ids", []))

        all_sessions = sorted(glob.glob(os.path.join(sessions_dir, "session_*")))
        unpacked_sessions = [s for s in all_sessions if os.path.basename(s) not in already_packed_sessions]

        if not unpacked_sessions:
            print("[*] No new unpacked sessions found to archive.")
            return None

        # Take batch up to max_sessions_per_chunk
        batch = unpacked_sessions[:max_sessions_per_chunk]
        chunk_idx = manifest.get("total_chunks", 0) + 1
        if not chunk_name:
            chunk_name = f"dataset_chunk_{chunk_idx:04d}.tar.gz"

        chunk_path = os.path.join(self.chunks_dir, chunk_name)
        print(f"\n[*] Creating compressed chunk '{chunk_name}' ({len(batch)} sessions) -> {self.chunks_dir}...")

        chunk_frames = 0
        chunk_duration_sec = 0.0
        session_ids = []

        with tarfile.open(chunk_path, "w:gz") as tar:
            for s_path in batch:
                s_id = os.path.basename(s_path)
                session_ids.append(s_id)
                tar.add(s_path, arcname=s_id)

                # Read metadata for statistics
                meta_file = os.path.join(s_path, "metadata.json")
                if os.path.exists(meta_file):
                    try:
                        with open(meta_file, "r", encoding="utf-8") as mf:
                            m = json.load(mf)
                            chunk_frames += m.get("total_video_frames", 0)
                            chunk_duration_sec += m.get("duration_seconds", 0.0)
                    except Exception:
                        pass

        chunk_size_mb = os.path.getsize(chunk_path) / (1024.0 * 1024.0)

        # Update manifest
        chunk_record = {
            "chunk_name": chunk_name,
            "chunk_path": chunk_path,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_count": len(batch),
            "session_ids": session_ids,
            "frames": chunk_frames,
            "duration_hours": round(chunk_duration_sec / 3600.0, 3),
            "size_mb": round(chunk_size_mb, 2)
        }

        manifest["chunks"].append(chunk_record)
        manifest["total_chunks"] = len(manifest["chunks"])
        manifest["total_sessions"] += len(batch)
        manifest["total_frames"] += chunk_frames
        manifest["total_duration_hours"] = round(manifest.get("total_duration_hours", 0.0) + (chunk_duration_sec / 3600.0), 3)
        manifest["total_size_mb"] = round(manifest.get("total_size_mb", 0.0) + chunk_size_mb, 2)

        self.save_manifest(manifest)
        print(f"[✓] Chunk saved: {chunk_name} ({chunk_size_mb:.1f} MB, {chunk_frames:,} frames, {chunk_duration_sec/60.0:.1f} mins)")

        # Optionally purge local sessions to reclaim precious disk space (keeps disk < 5GB!)
        if purge_local_after_pack:
            print("[*] Reclaiming disk space: Purging local session directories safely...")
            for s_path in batch:
                try:
                    shutil.rmtree(s_path, ignore_errors=True)
                except Exception:
                    pass
            print(f"[✓] Successfully freed local disk space! Data is safely stored in Google Drive.")

        return chunk_path

    def unpack_all_chunks_to_local_ssd(self, target_dir: str) -> int:
        """Unpack all .tar.gz chunks from Google Drive into fast local NVMe / SSD for model training.
        Returns total number of sessions extracted.
        """
        os.makedirs(target_dir, exist_ok=True)
        chunk_files = sorted(glob.glob(os.path.join(self.chunks_dir, "*.tar.gz")))
        if not chunk_files:
            print(f"[!] No chunks found in: {self.chunks_dir}")
            return 0

        print("=" * 75)
        print(" [FAST UNPACK] EXTRACTING GOOGLE DRIVE CHUNKS TO HIGH-SPEED LOCAL SSD")
        print(f" Source Chunks Dir : {self.chunks_dir}")
        print(f" Destination SSD   : {target_dir}")
        print(f" Total Chunks      : {len(chunk_files)}")
        print("=" * 75)

        total_extracted = 0
        t0 = time.perf_counter()

        for idx, cf in enumerate(chunk_files, 1):
            sz_mb = os.path.getsize(cf) / (1024.0 * 1024.0)
            print(f"[{idx}/{len(chunk_files)}] Extracting '{os.path.basename(cf)}' ({sz_mb:.1f} MB)...", end=" ", flush=True)
            with tarfile.open(cf, "r:gz") as tar:
                members = tar.getmembers()
                # Filter to top-level session directories
                session_names = set(m.name.split("/")[0] for m in members if "/" in m.name or m.isdir())
                tar.extractall(path=target_dir)
                total_extracted += len(session_names)
            print("[DONE]")

        elapsed = time.perf_counter() - t0
        print("-" * 75)
        print(f"[✓] Extracted {total_extracted} sessions into '{target_dir}' in {elapsed:.1f}s!")
        print("=" * 75 + "\n")
        return total_extracted
