"""Entry point for Facial Key-Animation Acquisition System (GUI & CLI modes)."""

import argparse
import os
import sys
import time


def run_gui(session_dir: str = None):
    """Launch full PySide6 GUI Studio application."""
    from PySide6.QtWidgets import QApplication
    from src.ui.main_window import MainWindow
    from src.ui.session_viewer import SessionViewerWindow

    app = QApplication(sys.argv)
    app.setApplicationName("FacialMotionCaptureStudio")

    if session_dir:
        viewer = SessionViewerWindow(session_dir)
        viewer.show()
    else:
        window = MainWindow()
        window.show()

    sys.exit(app.exec())


def run_cli(args):
    """Execute headless command-line workflows."""
    from src.config import AppConfig
    from src.core.session_manager import SessionManager
    from src.storage.dataset_reader import DatasetReader
    from src.schema import EligibilityLevel

    config = AppConfig()

    if args.inspect:
        print(f"Opening Session Inspector for: {args.inspect}")
        run_gui(session_dir=args.inspect)
        return

    if args.export:
        if not args.output:
            print("Error: --output <file.jsonl> is required when exporting.")
            sys.exit(1)
        reader = DatasetReader(args.export)
        lvl = EligibilityLevel(args.min_level) if args.min_level is not None else EligibilityLevel.LEVEL_3_ANIMATION_QUALITY
        count = reader.export_to_jsonl(args.output, min_eligibility=lvl)
        print(f"Exported {count} samples (>= Level {lvl.value}) to {args.output}")
        return

    manager = SessionManager(config)
    print("Initializing facial perception pipeline (MediaPipe CPU XNNPACK)...")
    manager.initialize()
    print("Pipeline ready.")

    roi = (100, 100, 640, 480)
    if args.roi:
        parts = [int(p.strip()) for p in args.roi.split(",")]
        if len(parts) == 4:
            roi = tuple(parts)

    try:
        if args.mode == "video":
            if not args.video:
                print("Error: --video <path> is required for video mode.")
                sys.exit(1)
            print(f"Processing offline video: {args.video}...")
            t0 = time.time()
            saved_dir = manager.process_video_file(args.video)
            print(f"Finished processing in {time.time()-t0:.2f}s.")
            print(f"Dataset saved cleanly to: {saved_dir}")

        elif args.mode == "manual":
            print(f"Starting manual recording session on ROI {roi}...")
            manager.start_manual_session(roi=roi)
            dur = args.duration if args.duration else 10.0
            print(f"Recording for {dur} seconds (Press Ctrl+C to stop early)...")
            time.sleep(dur)
            saved_dir = manager.stop_session()
            print(f"Session finalized and saved to: {saved_dir}")

        elif args.mode == "watcher":
            print(f"Arming automatic watcher on ROI {roi}...")
            manager.arm_watcher(roi=roi)
            print("Watcher active in background. Press Ctrl+C to disarm.")
            while True:
                time.sleep(1.0)

    except KeyboardInterrupt:
        print("\nInterrupt received. Stopping session...")
        saved_dir = manager.stop_session()
        if saved_dir:
            print(f"Saved dataset: {saved_dir}")
    finally:
        manager.close()


def main():
    parser = argparse.ArgumentParser(description="Facial Key-Animation Data Acquisition System")
    parser.add_argument("--cli", action="store_true", help="Run in headless CLI mode instead of GUI")
    parser.add_argument("--mode", choices=["manual", "watcher", "video"], default="manual", help="Acquisition mode")
    parser.add_argument("--video", type=str, help="Path to input video for Mode C")
    parser.add_argument("--roi", type=str, default="100,100,640,480", help="ROI coordinates x,y,w,h")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration for manual CLI recording")
    parser.add_argument("--inspect", type=str, help="Open session viewer directly for a recorded session folder")
    parser.add_argument("--export", type=str, help="Export session folder to JSONL")
    parser.add_argument("--output", type=str, help="Target JSONL export path")
    parser.add_argument("--min-level", type=int, default=3, help="Minimum eligibility level for export (0 to 4)")

    args = parser.parse_args()

    if args.cli or args.inspect or args.export:
        run_cli(args)
    else:
        run_gui()


if __name__ == "__main__":
    main()
