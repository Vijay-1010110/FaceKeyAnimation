"""Cloud Storage, Google Drive Chunking & Manifest Manager.
Handles packaging recorded facial animation sessions into compressed .tar.gz chunks,
syncing to Google Drive (optimized for large 5 TB storage plans), and extracting chunks
to local SSD for ultra-fast GPU training.

Supports parallel multi-worker clouds:
  - Google Colab (free 16GB T4 GPU)
  - Kaggle (free P100 / 2x T4 GPUs)
  - Local PC / Custom Cloud
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
    Prevents local disk exhaustion (< 5 GB) by packaging sessions into chunks and streaming to Drive.
    """

    def __init__(self, project_root: Optional[str] = None, drive_folder: Optional[str] = None):
        if project_root is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.project_root = os.path.abspath(project_root)

        # Detect environment
        self.is_colab = os.path.exists("/content") and ("google.colab" in sys.modules or os.path.exists("/content/sample_data"))
        self.is_kaggle = os.path.exists("/kaggle")
        self.is_lightning = "LIGHTNING_STUDIO_ID" in os.environ or os.path.exists("/teamspace")

        # Drive root folder path
        if drive_folder:
            self.drive_folder = os.path.abspath(drive_folder)
        elif self.is_colab:
            self.drive_folder = "/content/drive/MyDrive/FaceKeyDataset"
        elif self.is_kaggle:
            # Check common Kaggle Google Drive mount locations
            if os.path.exists("/kaggle/input/google-drive/FaceKeyDataset"):
                self.drive_folder = "/kaggle/input/google-drive/FaceKeyDataset"
            elif os.path.exists("/kaggle/working/google_drive/FaceKeyDataset"):
                self.drive_folder = "/kaggle/working/google_drive/FaceKeyDataset"
            else:
                self.drive_folder = "/kaggle/working/FaceKeyDataset"
        elif self.is_lightning:
            self.drive_folder = "/teamspace/studios/this_studio/FaceKeyDataset" if os.path.exists("/teamspace") else os.path.join(self.project_root, "FaceKeyDataset")
        else:
            self.drive_folder = os.path.join(self.project_root, "cloud_drive_backup")

        self.chunks_dir = os.path.join(self.drive_folder, "chunks")
        self.checkpoints_dir = os.path.join(self.drive_folder, "checkpoints")
        self.locks_dir = os.path.join(self.drive_folder, "locks")
        self.data_dir = os.path.join(self.drive_folder, "data")
        self.manifest_path = os.path.join(self.drive_folder, "dataset_manifest.json")

        # Standard URL queue file paths
        self.colab_urls_path = os.path.join(self.drive_folder, "youtube_urls_colab.txt")
        self.kaggle_urls_path = os.path.join(self.drive_folder, "youtube_urls_kaggle.txt")
        self.lightning_urls_path = os.path.join(self.drive_folder, "youtube_urls_lightning.txt")
        self.shared_urls_path = os.path.join(self.drive_folder, "youtubeURLtoProcess.txt")

    def mount_google_drive(self) -> bool:
        """Mount Google Drive inside Google Colab or verify Drive path in Kaggle/Local."""
        if self.is_colab:
            try:
                from google.colab import drive
                if not os.path.exists("/content/drive/MyDrive"):
                    print("[*] Mounting Google Drive to '/content/drive'...")
                    drive.mount("/content/drive")
                os.makedirs(self.chunks_dir, exist_ok=True)
                os.makedirs(self.checkpoints_dir, exist_ok=True)
                os.makedirs(self.locks_dir, exist_ok=True)
                os.makedirs(self.data_dir, exist_ok=True)
                print("[+] Google Drive successfully mounted at:", self.drive_folder)
                self.setup_drive_queue_files()
                return True
            except Exception as e:
                print(f"[!] Warning: Could not mount Google Drive automatically ({e})")
                return False

        # Kaggle or Local environment
        os.makedirs(self.chunks_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.locks_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        self.setup_drive_queue_files()
        return True

    def setup_drive_queue_files(self):
        """Ensure default URL queue files exist on Google Drive for Colab, Kaggle, and Shared."""
        local_sessions = os.path.join(self.project_root, "sessions")
        local_colab = os.path.join(local_sessions, "youtube_urls_colab.txt")
        local_kaggle = os.path.join(local_sessions, "youtube_urls_kaggle.txt")
        local_lightning = os.path.join(local_sessions, "youtube_urls_lightning.txt")

        # 1. Colab Dedicated URL file
        if not os.path.exists(self.colab_urls_path):
            if os.path.exists(local_colab):
                try:
                    shutil.copy2(local_colab, self.colab_urls_path)
                except Exception:
                    pass

        # 2. Kaggle Dedicated URL file
        if not os.path.exists(self.kaggle_urls_path):
            if os.path.exists(local_kaggle):
                try:
                    shutil.copy2(local_kaggle, self.kaggle_urls_path)
                except Exception:
                    pass

        # 3. Lightning.ai Dedicated URL file
        if not os.path.exists(self.lightning_urls_path):
            if os.path.exists(local_lightning):
                try:
                    shutil.copy2(local_lightning, self.lightning_urls_path)
                except Exception:
                    pass

        # 3. Shared Master URL file
        if not os.path.exists(self.shared_urls_path):
            if os.path.exists(local_shared):
                try:
                    shutil.copy2(local_shared, self.shared_urls_path)
                except Exception:
                    pass
            else:
                with open(self.shared_urls_path, "w", encoding="utf-8") as f:
                    f.write("# FaceKey Master YouTube Stream Queue (Shared Across All Workers)\n\n")

    def load_manifest(self) -> Dict[str, Any]:
        """Load manifest of all packaged dataset chunks from Google Drive."""
        if os.path.exists(self.manifest_path):
            for _ in range(5):
                try:
                    with open(self.manifest_path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    time.sleep(0.2)
        return {
            "version": "1.1",
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "storage_plan": "5 TB Google Drive Plan (18 Months)",
            "total_chunks": 0,
            "total_sessions": 0,
            "total_frames": 0,
            "total_duration_hours": 0.0,
            "total_size_mb": 0.0,
            "chunks": []
        }

    def save_manifest(self, manifest: Dict[str, Any]):
        """Save dataset manifest to Google Drive safely."""
        manifest["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        manifest["storage_plan"] = "5 TB Google Drive Plan (18 Months)"
        os.makedirs(os.path.dirname(self.manifest_path), exist_ok=True)
        tmp = self.manifest_path + f".tmp_{os.getpid()}"
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
        worker_tag: str = "worker",
        max_sessions_per_chunk: int = 10,
        purge_local_after_pack: bool = False
    ) -> Optional[str]:
        """Packs local session directories into a compressed .tar.gz chunk on Google Drive.
        Names chunks with worker tag and timestamp to avoid collisions in parallel execution!
        If purge_local_after_pack is True, removes local session folders to free up disk space (< 5GB).
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
            # Non-colliding unique chunk name across parallel Colab and Kaggle workers
            ts_str = time.strftime("%Y%m%d_%H%M%S")
            chunk_name = f"dataset_chunk_{worker_tag}_{ts_str}_{chunk_idx:04d}.tar.gz"

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

        # Reload latest manifest to ensure atomic addition in multi-worker environment
        manifest = self.load_manifest()

        chunk_record = {
            "chunk_name": chunk_name,
            "chunk_path": chunk_path,
            "worker_tag": worker_tag,
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
        print(f"[+] Chunk saved: {chunk_name} ({chunk_size_mb:.1f} MB, {chunk_frames:,} frames, {chunk_duration_sec/60.0:.1f} mins)")
        print(f"[*] 5 TB Google Drive Total: {manifest['total_duration_hours']:.2f} hrs across {manifest['total_chunks']} chunks ({manifest['total_size_mb']/1024.0:.2f} GB)")

        # Purge local sessions to reclaim disk space (keeps disk < 5GB!)
        if purge_local_after_pack:
            print("[*] Reclaiming disk space: Purging local session directories safely...")
            for s_path in batch:
                try:
                    shutil.rmtree(s_path, ignore_errors=True)
                except Exception:
                    pass
            print(f"[+] Successfully freed local disk space! Data safely stored in Google Drive.")

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
                session_names = set(m.name.split("/")[0] for m in members if "/" in m.name or m.isdir())
                tar.extractall(path=target_dir)
                total_extracted += len(session_names)
            print("[DONE]")

        elapsed = time.perf_counter() - t0
        print("-" * 75)
        print(f"[+] Extracted {total_extracted} sessions into '{target_dir}' in {elapsed:.1f}s!")
        print("=" * 75 + "\n")
        return total_extracted
