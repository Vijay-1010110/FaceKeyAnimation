"""In-memory thread-safe ring buffer and bounded packet queue."""

import collections
import threading
import time
from typing import List, Optional, Tuple, Any
import numpy as np


class PreRollRingBuffer:
    """Fixed-capacity in-memory ring buffer holding recent raw frames.
    Used for pre-roll in Mode A (Watcher) so that when a face appears, the
    preceding 1-2 seconds of video are preserved for the start of the session.
    Frames are stored strictly in RAM and overwritten when capacity is reached.
    """

    def __init__(self, max_seconds: float = 2.0, target_fps: int = 30):
        self.capacity = max(10, int(max_seconds * target_fps))
        self.buffer = collections.deque(maxlen=self.capacity)
        self.lock = threading.Lock()

    def append(self, timestamp: float, frame_rgb: np.ndarray, audio_data: Optional[Any] = None):
        """Append a frame copy or reference to the circular buffer."""
        with self.lock:
            self.buffer.append((timestamp, frame_rgb.copy(), audio_data))

    def pop_pre_roll(self, lookback_seconds: float = 1.5) -> List[Tuple[float, np.ndarray, Any]]:
        """Retrieve and drain frames matching the lookback duration.
        Returns frames in chronological order.
        """
        with self.lock:
            if not self.buffer:
                return []
            latest_time = self.buffer[-1][0]
            cutoff_time = latest_time - lookback_seconds
            results = [item for item in self.buffer if item[0] >= cutoff_time]
            self.buffer.clear()
            return results

    def clear(self):
        """Clear all buffered frames."""
        with self.lock:
            self.buffer.clear()

    def __len__(self):
        with self.lock:
            return len(self.buffer)


class BoundedFrameQueue:
    """Bounded thread-safe queue with explicit backpressure/drop policy.
    If the consumer (e.g. neural network inference) falls behind, oldest
    frames are dropped instead of allowing RAM usage to grow indefinitely.
    """

    def __init__(self, max_size: int = 60):
        self.max_size = max_size
        self.queue = collections.deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)
        self.dropped_frames: int = 0

    def put(self, item: Any) -> bool:
        """Push an item into the queue.
        If queue is at capacity, drops the oldest frame.
        """
        with self.lock:
            dropped = False
            if len(self.queue) >= self.max_size:
                self.queue.popleft()
                self.dropped_frames += 1
                dropped = True
            self.queue.append(item)
            self.not_empty.notify()
            return not dropped

    def get(self, timeout: Optional[float] = None) -> Optional[Any]:
        """Pop an item from the queue, blocking up to timeout seconds."""
        with self.not_empty:
            start_time = time.monotonic()
            while len(self.queue) == 0:
                if timeout is None:
                    self.not_empty.wait()
                else:
                    elapsed = time.monotonic() - start_time
                    remaining = timeout - elapsed
                    if remaining <= 0:
                        return None
                    self.not_empty.wait(remaining)
            return self.queue.popleft()

    def qsize(self) -> int:
        with self.lock:
            return len(self.queue)

    def clear(self):
        with self.lock:
            self.queue.clear()
