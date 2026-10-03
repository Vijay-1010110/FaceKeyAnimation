"""Master session manager coordinating lifecycle states, capture modes, and dataset output."""

from enum import Enum
import os
import threading
import time
import uuid
from typing import Callable, Optional, List, Dict, Any, Tuple
import numpy as np

from src.config import AppConfig
from src.schema import TrackedFaceFrame, AudioFrameData, FramePacket
from src.core.capture import ScreenCaptureSource, VideoCaptureSource, AudioCaptureSource
from src.core.ring_buffer import PreRollRingBuffer, BoundedFrameQueue
from src.core.face_pipeline import FacePipeline
from src.core.profiler import SystemProfiler
from src.storage.dataset_writer import DatasetWriter


class SessionState(str, Enum):
    IDLE = "IDLE"
    ARMED = "ARMED"
    RECORDING = "RECORDING"
    FINALIZING = "FINALIZING"
    SAVED = "SAVED"


class SessionManager:
    """Orchestrates capture, inference, ring buffer, and persistence across all modes."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.state = SessionState.IDLE
        self.mode = "manual"  # "watcher", "manual", "uploaded_video"
        self.session_id: str = ""
        self.session_start_time: float = 0.0
        self.session_end_time: float = 0.0

        # Core engines
        self.pipeline = FacePipeline(config)
        self.profiler = SystemProfiler()
        self.writer = DatasetWriter(config.sessions_dir)

        # Buffers
        self.pre_roll = PreRollRingBuffer(
            max_seconds=config.ring_buffer_seconds,
            target_fps=config.target_fps
        )
        self.frame_queue = BoundedFrameQueue(max_size=config.queue_max_size)

        # In-memory session accumulator (cleared upon save)
        self.accumulated_face_frames: List[TrackedFaceFrame] = []
        self.accumulated_audio_frames: List[AudioFrameData] = []

        # Sources
        self.screen_source: Optional[ScreenCaptureSource] = None
        self.audio_source: Optional[AudioCaptureSource] = None
        self.video_source: Optional[VideoCaptureSource] = None

        # Watcher state tracking
        self.watcher_face_seen_start: Optional[float] = None
        self.watcher_last_face_time: Optional[float] = None

        # Worker threads
        self.worker_thread: Optional[threading.Thread] = None
        self.running = False
        self._lock = threading.Lock()

        # Diagnostic callbacks
        self.on_frame_processed: Optional[Callable[[float, List[TrackedFaceFrame], Optional[np.ndarray]], None]] = None
        self.on_state_changed: Optional[Callable[[SessionState], None]] = None
        self.on_progress: Optional[Callable[[float, float], None]] = None

    def _set_state(self, new_state: SessionState):
        with self._lock:
            self.state = new_state
        if self.on_state_changed:
            self.on_state_changed(new_state)

    def initialize(self):
        """Initialize the underlying neural face perception model."""
        self.pipeline.initialize()

    def close(self):
        """Stop all streams and release resources."""
        self.stop_session()
        self.pipeline.close()

    # -------------------------------------------------------------
    # Mode B: Manual Recording
    # -------------------------------------------------------------
    def start_manual_session(
        self,
        roi: Tuple[int, int, int, int] = (100, 100, 640, 480),
        target_hwnd: Optional[int] = None,
        audio_device_idx: Optional[int] = None
    ) -> str:
        """Start an explicit manual recording session."""
        with self._lock:
            if self.state in (SessionState.RECORDING, SessionState.FINALIZING):
                return self.session_id

            self.mode = "manual"
            self.session_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4]
            self.session_start_time = time.perf_counter()
            self.accumulated_face_frames.clear()
            self.accumulated_audio_frames.clear()
            self.frame_queue.clear()
            self.pipeline.reset()

        self._set_state(SessionState.RECORDING)
        self.running = True

        # Start downstream processing worker
        self.worker_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.worker_thread.start()

        # Start screen capture source
        self.screen_source = ScreenCaptureSource(roi=roi, target_fps=self.config.target_fps, target_hwnd=target_hwnd)
        self.screen_source.start(self._on_captured_screen_frame)

        # Start optional audio capture
        self.audio_source = AudioCaptureSource(device_index=audio_device_idx)
        self.audio_source.start(self._on_captured_audio_chunk, start_time=self.session_start_time)

        return self.session_id

    # -------------------------------------------------------------
    # Live Sensor Testing Mode (Zero disk persistence)
    # -------------------------------------------------------------
    def start_live_sensor(
        self,
        roi: Tuple[int, int, int, int] = (100, 100, 640, 480),
        target_hwnd: Optional[int] = None,
        audio_device_idx: Optional[int] = None
    ) -> str:
        """Start real-time facial motion sensor without accumulating dataset frames."""
        with self._lock:
            if self.state in (SessionState.RECORDING, SessionState.FINALIZING):
                return self.session_id

            self.mode = "live_sensor"
            self.session_id = "live_sensor_" + uuid.uuid4().hex[:4]
            self.session_start_time = time.perf_counter()
            self.accumulated_face_frames.clear()
            self.accumulated_audio_frames.clear()
            self.frame_queue.clear()
            self.pipeline.reset()

        self._set_state(SessionState.RECORDING)
        self.running = True

        self.worker_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.worker_thread.start()

        self.screen_source = ScreenCaptureSource(roi=roi, target_fps=self.config.target_fps, target_hwnd=target_hwnd)
        self.screen_source.start(self._on_captured_screen_frame)

        self.audio_source = AudioCaptureSource(device_index=audio_device_idx)
        self.audio_source.start(self._on_captured_audio_chunk, start_time=self.session_start_time)

        return self.session_id

    # -------------------------------------------------------------
    # Mode A: Automatic Watcher / Passive Mode
    # -------------------------------------------------------------
    def arm_watcher(
        self,
        roi: Tuple[int, int, int, int] = (100, 100, 640, 480),
        target_hwnd: Optional[int] = None,
        audio_device_idx: Optional[int] = None
    ):
        """Arm passive background watcher. Consumes minimal resources until face is detected."""
        with self._lock:
            if self.state != SessionState.IDLE:
                return
            self.mode = "watcher"
            self.session_id = time.strftime("%Y%m%d_%H%M%S") + "_watch_" + uuid.uuid4().hex[:4]
            self.accumulated_face_frames.clear()
            self.accumulated_audio_frames.clear()
            self.pre_roll.clear()
            self.frame_queue.clear()
            self.watcher_face_seen_start = None
            self.watcher_last_face_time = None
            self.pipeline.reset()

        self._set_state(SessionState.ARMED)
        self.running = True

        # Start downstream worker
        self.worker_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.worker_thread.start()

        # Start capture source in watcher mode
        self.screen_source = ScreenCaptureSource(roi=roi, target_fps=min(15, self.config.target_fps), target_hwnd=target_hwnd)
        self.screen_source.start(self._on_captured_screen_frame)

        self.audio_source = AudioCaptureSource(device_index=audio_device_idx)
        self.audio_source.start(self._on_captured_audio_chunk)

    # -------------------------------------------------------------
    # Frame Ingestion Callbacks
    # -------------------------------------------------------------
    def _on_captured_screen_frame(self, timestamp: float, frame_rgb: np.ndarray):
        self.profiler.record_capture_event()

        if self.state == SessionState.ARMED:
            # Mode A: Store in RAM pre-roll circular buffer
            self.pre_roll.append(timestamp, frame_rgb, None)
            # Submit for light-weight face verification
            self.frame_queue.put(FramePacket(timestamp=timestamp, roi_image=frame_rgb))

        elif self.state == SessionState.RECORDING:
            # Mode B or active Mode A: queue for immediate processing & disposal
            success = self.frame_queue.put(FramePacket(timestamp=timestamp, roi_image=frame_rgb))
            self.profiler.set_queue_state(self.frame_queue.qsize(), self.frame_queue.dropped_frames)

    def _on_captured_audio_chunk(self, timestamp: float, audio_samples: np.ndarray):
        audio_frame = self.pipeline.speaker_engine.update_audio(audio_samples, timestamp)
        if self.state == SessionState.RECORDING:
            self.accumulated_audio_frames.append(audio_frame)

    # -------------------------------------------------------------
    # Mode C: Direct Uploaded Video Processing
    # -------------------------------------------------------------
    def process_video_file(self, video_path: str) -> str:
        """Process an offline uploaded video sequentially without temporary disk files."""
        self.mode = "uploaded_video"
        self.session_id = time.strftime("%Y%m%d_%H%M%S") + "_vid_" + uuid.uuid4().hex[:4]
        self.session_start_time = time.perf_counter()
        self.accumulated_face_frames.clear()
        self.accumulated_audio_frames.clear()
        self.pipeline.reset()

        self._set_state(SessionState.RECORDING)
        self.running = True

        self.video_source = VideoCaptureSource(video_path)

        def frame_cb(ts: float, frame_rgb: np.ndarray, audio_samples: Optional[np.ndarray]):
            if not self.running:
                return
            t0 = time.perf_counter()
            audio_frame = None
            if audio_samples is not None:
                audio_frame = self.pipeline.speaker_engine.update_audio(audio_samples, ts)
                self.accumulated_audio_frames.append(audio_frame)

            tracked_faces = self.pipeline.process_frame(frame_rgb, ts, audio_frame)
            latency = (time.perf_counter() - t0) * 1000.0
            self.profiler.record_inference_latency(latency)

            # STRICT SINGLE-FACE VERIFIED-SPEAKING GATE:
            # Only record/accumulate frames when exactly 1 face is present and actively speaking
            if self.pipeline.speaker_engine.is_gate_active:
                self.accumulated_face_frames.extend(tracked_faces)
            if self.on_frame_processed:
                latest_audio = audio_frame or getattr(self.pipeline.speaker_engine, "latest_audio", None)
                self.on_frame_processed(ts, tracked_faces, frame_rgb, latest_audio)

        def prog_cb(pct: float, cur_ts: float):
            if self.on_progress:
                self.on_progress(pct, cur_ts)

        try:
            self.video_source.process(frame_cb, prog_cb)
        finally:
            self.running = False
            return self.stop_session()

    # -------------------------------------------------------------
    # Downstream Inference Worker
    # -------------------------------------------------------------
    def _processing_worker(self):
        while self.running:
            packet: Optional[FramePacket] = self.frame_queue.get(timeout=0.1)
            if packet is None or packet.roi_image is None:
                continue

            frame_rgb = packet.roi_image
            ts = packet.timestamp

            t0 = time.perf_counter()
            try:
                tracked_faces = self.pipeline.process_frame(frame_rgb, ts, packet.audio_frame)
            except Exception:
                tracked_faces = []
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self.profiler.record_inference_latency(latency_ms)

            # Check Watcher Hysteresis Trigger in ARMED mode
            if self.state == SessionState.ARMED:
                has_usable_face = (
                    len(tracked_faces) == 1
                    and self.pipeline.speaker_engine.is_gate_active
                    and any(f.quality.is_valid for f in tracked_faces)
                )
                if has_usable_face:
                    if self.watcher_face_seen_start is None:
                        self.watcher_face_seen_start = ts
                    elif (ts - self.watcher_face_seen_start) >= self.config.hysteresis.watcher_trigger_sec:
                        # Single speaking face observed continuously -> activate RECORDING!
                        self._activate_watcher_recording(ts)
                else:
                    self.watcher_face_seen_start = None

            elif self.state == SessionState.RECORDING:
                # Mode A Watcher Loss Hysteresis
                if self.mode == "watcher":
                    has_usable_face = (
                        len(tracked_faces) == 1
                        and any(f.quality.is_valid for f in tracked_faces)
                    )
                    if has_usable_face:
                        self.watcher_last_face_time = ts
                    elif self.watcher_last_face_time is not None:
                        if (ts - self.watcher_last_face_time) >= self.config.hysteresis.watcher_loss_sec:
                            # Face lost for duration -> stop watcher session
                            threading.Thread(target=self.stop_session, daemon=True).start()
                            break

                # STRICT GATE: Only accumulate into dataset when 1 face + verified speaking
                if self.mode != "live_sensor" and self.pipeline.speaker_engine.is_gate_active:
                    self.accumulated_face_frames.extend(tracked_faces)

            # Broadcast to diagnostic visualizer (if hooked)
            if self.on_frame_processed:
                latest_audio = getattr(self.pipeline.speaker_engine, "latest_audio", None)
                self.on_frame_processed(ts, tracked_faces, frame_rgb, latest_audio)

            # Release the image buffer immediately to keep RAM strictly bounded
            packet.roi_image = None
            del frame_rgb

    def _activate_watcher_recording(self, current_ts: float):
        """Transition watcher from ARMED to RECORDING, flushing RAM pre-roll buffer."""
        self._set_state(SessionState.RECORDING)
        self.session_start_time = time.perf_counter()
        self.watcher_last_face_time = current_ts

        # Flush pre-roll frames into active pipeline
        pre_roll_frames = self.pre_roll.pop_pre_roll(lookback_seconds=self.config.ring_buffer_seconds)
        for ts, pr_rgb, pr_audio in pre_roll_frames:
            try:
                pr_faces = self.pipeline.process_frame(pr_rgb, ts, None)
                if self.mode != "live_sensor":
                    self.accumulated_face_frames.extend(pr_faces)
            except Exception:
                pass
            del pr_rgb

    # -------------------------------------------------------------
    # Finalization and Cleanup
    # -------------------------------------------------------------
    def stop_session(self) -> str:
        """Stop capture, finalize structured data, and write dataset to disk."""
        if self.state in (SessionState.IDLE, SessionState.FINALIZING, SessionState.SAVED):
            return ""

        self._set_state(SessionState.FINALIZING)
        self.running = False
        self.session_end_time = time.perf_counter()

        # Stop hardware capture streams
        if self.screen_source:
            self.screen_source.stop()
            self.screen_source = None
        if self.audio_source:
            self.audio_source.stop()
            self.audio_source = None
        if self.video_source:
            self.video_source.cancel()
            self.video_source = None

        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
            self.worker_thread = None

        # Write dataset to disk (structured data only, zero frame images)
        snapshot = self.profiler.get_snapshot()
        session_dir = ""
        if self.mode != "live_sensor" and self.accumulated_face_frames:
            session_dir = self.writer.write_session(
                session_id=self.session_id,
                mode=self.mode,
                source_description="ROI Screen Capture" if self.mode != "uploaded_video" else "Direct Video File",
                face_frames=self.accumulated_face_frames,
                audio_frames=self.accumulated_audio_frames,
                session_start_time=self.session_start_time,
                session_end_time=self.session_end_time,
                capture_fps=snapshot["capture_fps"],
                inference_fps=snapshot["inference_fps"],
                telemetry=snapshot
            )

        self._set_state(SessionState.SAVED)
        self._set_state(SessionState.IDLE)
        return session_dir
