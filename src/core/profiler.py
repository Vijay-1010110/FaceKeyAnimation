"""Telemetry and performance profiler measuring CPU, GPU, VRAM, RAM, and FPS."""

import collections
import os
import subprocess
import threading
import time
from typing import Dict, Any, Optional
import psutil

# Windows flag to suppress console window creation
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


class SystemProfiler:
    """Monitors real-time CPU, GPU, RAM, VRAM, and processing pipeline throughput."""

    def __init__(self, process_id: Optional[int] = None):
        self.pid = process_id or os.getpid()
        self.process = psutil.Process(self.pid)
        self.lock = threading.RLock()

        # Rolling statistics
        self.capture_timestamps: collections.deque = collections.deque(maxlen=60)
        self.inference_latencies: collections.deque = collections.deque(maxlen=60)
        self.dropped_frames: int = 0
        self.queue_depth: int = 0

        # Cached hardware query results
        self.last_gpu_query: float = 0.0
        self.cached_gpu_info: Dict[str, Any] = {"gpu_util_pct": 0, "vram_used_mb": 0, "vram_total_mb": 4096}

    def _query_gpu(self) -> Dict[str, Any]:
        """Query GPU metrics lazily with rate limiting to prevent driver overhead."""
        now = time.perf_counter()
        if now - self.last_gpu_query < 2.0:
            return self.cached_gpu_info

        self.last_gpu_query = now
        try:
            cmd = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]
            kwargs = {"capture_output": True, "text": True, "timeout": 0.5}
            if os.name == "nt":
                kwargs["creationflags"] = CREATE_NO_WINDOW

            result = subprocess.run(cmd, **kwargs)
            if result.returncode == 0 and result.stdout.strip():
                parts = [p.strip() for p in result.stdout.strip().split(",")]
                if len(parts) >= 3:
                    self.cached_gpu_info = {
                        "gpu_util_pct": int(parts[0]),
                        "vram_used_mb": int(parts[1]),
                        "vram_total_mb": int(parts[2])
                    }
        except Exception:
            pass
        return self.cached_gpu_info

    def record_capture_event(self):
        """Record timestamp of a captured frame for FPS calculation."""
        with self.lock:
            self.capture_timestamps.append(time.perf_counter())

    def record_inference_latency(self, latency_ms: float):
        """Record inference time for a frame."""
        with self.lock:
            self.inference_latencies.append(latency_ms)

    def set_queue_state(self, depth: int, drops: int):
        with self.lock:
            self.queue_depth = depth
            self.dropped_frames = drops

    def get_capture_fps(self) -> float:
        with self.lock:
            if len(self.capture_timestamps) < 2:
                return 0.0
            dt = self.capture_timestamps[-1] - self.capture_timestamps[0]
            if dt <= 0:
                return 0.0
            return (len(self.capture_timestamps) - 1) / dt

    def get_avg_inference_latency(self) -> float:
        with self.lock:
            if not self.inference_latencies:
                return 0.0
            return float(sum(self.inference_latencies) / len(self.inference_latencies))

    def get_snapshot(self) -> Dict[str, Any]:
        """Get instantaneous system snapshot."""
        try:
            cpu_pct = self.process.cpu_percent(interval=None)
            mem_info = self.process.memory_info()
            ram_mb = mem_info.rss / (1024 * 1024)
        except Exception:
            cpu_pct = 0.0
            ram_mb = 0.0

        gpu_info = self._query_gpu()

        with self.lock:
            capture_fps = self.get_capture_fps()
            avg_latency = self.get_avg_inference_latency()
            inf_fps = (1000.0 / avg_latency) if avg_latency > 0 else 0.0
            drops = self.dropped_frames
            q_depth = self.queue_depth

        return {
            "cpu_percent": round(cpu_pct, 1),
            "ram_rss_mb": round(ram_mb, 1),
            "gpu_util_pct": gpu_info.get("gpu_util_pct", 0),
            "vram_used_mb": gpu_info.get("vram_used_mb", 0),
            "vram_total_mb": gpu_info.get("vram_total_mb", 4096),
            "capture_fps": round(capture_fps, 1),
            "inference_fps": round(inf_fps, 1),
            "avg_latency_ms": round(avg_latency, 2),
            "queue_depth": q_depth,
            "dropped_frames": drops
        }

    def close(self):
        pass
