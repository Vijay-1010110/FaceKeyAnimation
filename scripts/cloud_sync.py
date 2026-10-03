"""Cloud Sync Manager Bridge & CLI
Exposes CloudDriveSync as CloudSyncManager and provides direct CLI chunk packaging.
"""
import os
import sys
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.storage.cloud_sync import CloudDriveSync

# Backward-compatibility alias
CloudSyncManager = CloudDriveSync

def main():
    parser = argparse.ArgumentParser(description="Cloud Dataset Chunk Packager")
    parser.add_argument("--drive-dir", type=str, default="/kaggle/working/FaceKeyDataset", help="Drive directory root")
    parser.add_argument("--sessions-dir", type=str, default=None, help="Sessions directory path")
    parser.add_argument("--worker-tag", type=str, default="worker", help="Worker name tag")
    parser.add_argument("--purge-local", action="store_true", default=False, help="Delete local sessions after packing")
    args = parser.parse_args()

    s_dir = args.sessions_dir or os.path.join(project_root, "sessions")
    sync = CloudDriveSync(project_root=project_root, drive_folder=args.drive_dir)
    chunk = sync.pack_sessions_into_chunk(
        sessions_dir=s_dir,
        worker_tag=args.worker_tag,
        purge_local_after_pack=args.purge_local
    )
    if chunk:
        print(f"[+] Chunk created: {chunk}")
    else:
        print("[*] No unchunked sessions to pack.")

if __name__ == "__main__":
    main()
