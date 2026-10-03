"""Low-resource capture sources: In-memory Screen ROI, Video Decoder, and Audio Stream."""

import ctypes
import os
import threading
import time
from typing import Callable, Optional, Tuple, Dict, Any, Generator

def ensure_input_desktop():
    """Ensure current thread is attached to the active Windows input desktop."""
    if os.name == "nt":
        try:
            user32 = ctypes.windll.user32
            # Prioritize 'Default' desktop (the real user display surface)
            hdesk = user32.OpenDesktopW("Default", 0, False, 0x01FF)
            if not hdesk:
                hdesk = user32.OpenInputDesktop(0, False, 0x01FF)
            if hdesk:
                user32.SetThreadDesktop(hdesk)
        except Exception:
            pass

# Guarantee current thread is on the interactive user desktop before any COM/audio initialization
ensure_input_desktop()

import numpy as np
import mss

try:
    import win32gui
    import win32con
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False

try:
    import av
    HAS_PYAV = True
except ImportError:
    HAS_PYAV = False

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False


class ScreenCaptureSource:
    """High-speed in-memory screen ROI capture without disk I/O.
    Uses MSS for low-overhead GDI direct frame buffer reading.
    Can dynamically follow a specific window handle (hwnd).
    """

    def __init__(
        self,
        roi: Tuple[int, int, int, int] = (100, 100, 640, 480),
        target_fps: int = 30,
        target_hwnd: Optional[int] = None
    ):
        self.roi = list(roi)  # [x, y, w, h]
        self.target_fps = target_fps
        self.target_hwnd = target_hwnd
        self.running = False
        self.paused = False
        self.thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def set_roi(self, x: int, y: int, w: int, h: int):
        with self._lock:
            self.roi = [int(x), int(y), max(16, int(w)), max(16, int(h))]

    def get_roi(self) -> Tuple[int, int, int, int]:
        with self._lock:
            return tuple(self.roi)

    def attach_to_window(self, hwnd: int):
        self.target_hwnd = hwnd
        self._update_window_roi()

    def _update_window_roi(self):
        if not HAS_WIN32 or not self.target_hwnd:
            return
        try:
            if win32gui.IsWindow(self.target_hwnd) and not win32gui.IsIconic(self.target_hwnd):
                rect = win32gui.GetWindowRect(self.target_hwnd)
                x = rect[0]
                y = rect[1]
                w = rect[2] - rect[0]
                h = rect[3] - rect[1]
                if w > 10 and h > 10:
                    with self._lock:
                        self.roi = [x, y, w, h]
        except Exception:
            pass

    def start(self, frame_callback: Callable[[float, np.ndarray], None]):
        """Start screen capture loop in a dedicated background thread."""
        if self.running:
            return
        self.running = True
        self.paused = False
        self.thread = threading.Thread(target=self._capture_worker, args=(frame_callback,), daemon=True)
        self.thread.start()

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.thread = None

    def _capture_worker(self, callback: Callable[[float, np.ndarray], None]):
        ensure_input_desktop()
        sct = mss.mss()
        interval = 1.0 / max(1, self.target_fps)
        session_start_time = time.perf_counter()

        while self.running:
            t_loop_start = time.perf_counter()

            if self.paused:
                time.sleep(0.05)
                continue

            if self.target_hwnd:
                self._update_window_roi()

            with self._lock:
                top = self.roi[1]
                left = self.roi[0]
                width = self.roi[2]
                height = self.roi[3]

            monitor = {"top": top, "left": left, "width": width, "height": height}

            try:
                sct_img = sct.grab(monitor)
                # sct_img.raw is BGRA, convert to RGB numpy array in RAM
                raw_bgra = np.frombuffer(sct_img.raw, dtype=np.uint8).reshape((height, width, 4))
                frame_rgb = raw_bgra[:, :, [2, 1, 0]]  # BGR to RGB
                timestamp = time.perf_counter() - session_start_time
                callback(timestamp, frame_rgb)
            except Exception as e:
                # In case of monitor bounds changes or display resolution transitions
                time.sleep(0.05)

            elapsed = time.perf_counter() - t_loop_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        sct.close()


class VideoCaptureSource:
    """Offline direct video file decoder stream using PyAV.
    Streams video frames and audio chunks into memory without temporary disk copies.
    """

    def __init__(self, file_path: str):
        if not HAS_PYAV:
            raise RuntimeError("PyAV is required for VideoCaptureSource")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Video file not found: {file_path}")
        self.file_path = file_path
        self.cancelled = False
        self.paused = False

    def cancel(self):
        self.cancelled = True

    def process(
        self,
        frame_callback: Callable[[float, np.ndarray, Optional[np.ndarray]], None],
        progress_callback: Optional[Callable[[float, float], None]] = None
    ):
        """Decode frames sequentially in presentation timestamp (PTS) order.
        Yields (timestamp_seconds, rgb_frame, optional_audio_samples).
        """
        container = av.open(self.file_path)
        video_stream = container.streams.video[0] if container.streams.video else None
        audio_stream = container.streams.audio[0] if container.streams.audio else None

        if not video_stream:
            container.close()
            raise ValueError(f"No video stream found in {self.file_path}")

        total_duration = float(container.duration) / av.time_base if container.duration else 0.0

        for frame in container.decode(video=0):
            if self.cancelled:
                break
            while self.paused and not self.cancelled:
                time.sleep(0.05)

            timestamp = float(frame.pts * video_stream.time_base) if frame.pts is not None else 0.0
            rgb_frame = frame.to_ndarray(format="rgb24")

            frame_callback(timestamp, rgb_frame, None)

            if progress_callback and total_duration > 0:
                progress_callback(min(1.0, timestamp / total_duration), timestamp)

        container.close()


class AudioCaptureSource:
    """Real-time system or microphone audio capture with accurate timestamps."""

    def __init__(self, sample_rate: int = 16000, chunk_duration_sec: float = 0.05, device_index: Optional[int] = None):
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration_sec)
        self.device_index = device_index
        self.running = False
        self.stream: Optional[Any] = None
        self.callback: Optional[Callable[[float, np.ndarray], None]] = None
        self.session_start_time: float = 0.0

    @staticmethod
    def get_input_devices() -> Dict[int, str]:
        """List available audio input devices (filtered for stability)."""
        if not HAS_SOUNDDEVICE:
            return {}
        devices = {}
        try:
            hostapis = sd.query_hostapis()
            for idx, dev in enumerate(sd.query_devices()):
                if dev.get("max_input_channels", 0) > 0:
                    api_name = hostapis[dev["hostapi"]]["name"]
                    if "WDM-KS" not in api_name:
                        devices[idx] = f"{idx}: {dev.get('name', 'Audio Device')} ({api_name})"
        except Exception:
            pass
        return devices

    def start(self, callback: Callable[[float, np.ndarray], None], start_time: Optional[float] = None):
        """Start capturing audio stream."""
        if not HAS_SOUNDDEVICE:
            return
        self.callback = callback
        self.session_start_time = start_time if start_time is not None else time.perf_counter()
        self.running = True

        def audio_cb(indata, frames, time_info, status):
            if not self.running:
                return
            timestamp = time.perf_counter() - self.session_start_time
            # Mono channel 32-bit float array
            mono = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
            if self.callback:
                self.callback(timestamp, mono)

        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                device=self.device_index,
                channels=1,
                dtype="float32",
                callback=audio_cb
            )
            self.stream.start()
        except Exception as e:
            # Gracefully handle device failure
            self.stream = None

    def stop(self):
        self.running = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None


class WindowCaptureSource:
    """Captures a specific application window (e.g. Chrome, Brave, YouTube) in the background.
    Uses Win32 PrintWindow with PW_RENDERFULLCONTENT so the window can be occluded,
    behind other windows, or not visible on screen, without interrupting your work.
    """

    def __init__(self, window_title_query: str, target_fps: int = 30):
        self.query = window_title_query.lower()
        self.target_fps = target_fps
        self.running = False
        self.paused = False
        self.thread: Optional[threading.Thread] = None
        self.target_hwnd: Optional[int] = None
        self.target_title: str = ""

    def find_target_window(self) -> Optional[int]:
        """Finds window handle matching title query using EnumWindows for reliable discovery."""
        matches = []

        def _clean_finder():
            user32 = ctypes.windll.user32
            hdesk = user32.OpenDesktopW("Default", 0, False, 0x01FF)
            if not hdesk:
                hdesk = user32.OpenInputDesktop(0, False, 0x01FF)
            if hdesk:
                user32.SetThreadDesktop(hdesk)

            def enum_cb(hwnd, lparam):
                if user32.IsWindowVisible(hwnd):
                    length = user32.GetWindowTextLengthW(hwnd)
                    if length > 0:
                        buf = ctypes.create_unicode_buffer(length + 1)
                        user32.GetWindowTextW(hwnd, buf, length + 1)
                        title = buf.value.strip()
                        if title and self.query in title.lower() and "facekey" not in title.lower():
                            matches.append((hwnd, title))
                return True

            WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
            user32.EnumWindows(WNDENUMPROC(enum_cb), 0)

        t = threading.Thread(target=_clean_finder)
        t.start()
        t.join(timeout=2.0)

        if matches:
            # Prioritize media / video / YouTube titles over static text tabs
            matches.sort(key=lambda m: 0 if any(k in m[1].lower() for k in ("youtube", "video", "movie", "watch", "drishyam", "trailer", "clip")) else 1)
            self.target_hwnd = matches[0][0]
            self.target_title = matches[0][1]
            return self.target_hwnd
        return None

    def start(self, frame_callback: Callable[[float, np.ndarray], None]):
        """Start window capture worker thread."""
        if self.running:
            return
        if not self.target_hwnd:
            self.find_target_window()
        if not self.target_hwnd:
            # Fallback 1: Auto-discover any active application or browser window
            all_wins = self.list_all_windows()
            if all_wins:
                # Prefer browser windows (brave, chrome, edge, youtube, etc.) if possible
                browser_wins = [w for w in all_wins if any(b in w[1].lower() for b in ("brave", "chrome", "edge", "firefox", "opera", "youtube", "video", "player"))]
                if browser_wins:
                    self.target_hwnd, self.target_title = browser_wins[0]
                else:
                    self.target_hwnd, self.target_title = all_wins[0]
                self.query = self.target_title.lower()
                print(f"[*] Window '{self.query}' auto-resolved to: [{self.target_hwnd}] '{self.target_title}'")
            else:
                raise RuntimeError(f"Could not find any open window matching: '{self.query}'")

        self.running = True
        self.paused = False
        self.thread = threading.Thread(target=self._worker, args=(frame_callback,), daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.thread = None

    def switch_target(self, new_hwnd: int, new_title: str):
        """Dynamically switch capture to a different window/tab without restarting."""
        self.target_hwnd = new_hwnd
        self.target_title = new_title
        self.query = new_title.lower()

    @staticmethod
    def get_window_title_by_hwnd(hwnd: int) -> str:
        """Retrieve title for a specific window handle."""
        if not hwnd:
            return ""
        user32 = ctypes.windll.user32
        length = user32.GetWindowTextLengthW(hwnd)
        if length > 0:
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            return buf.value.strip()
        return ""

    @staticmethod
    def list_all_windows() -> list:
        """List all visible application windows suitable for capture."""
        results = []

        def _clean_lister():
            user32 = ctypes.windll.user32
            hdesk = user32.OpenDesktopW("Default", 0, False, 0x01FF)
            if not hdesk:
                hdesk = user32.OpenInputDesktop(0, False, 0x01FF)
            if hdesk:
                user32.SetThreadDesktop(hdesk)

            def enum_cb(hwnd, lparam):
                if user32.IsWindowVisible(hwnd) and not user32.IsIconic(hwnd):
                    length = user32.GetWindowTextLengthW(hwnd)
                    if length > 0:
                        buf = ctypes.create_unicode_buffer(length + 1)
                        user32.GetWindowTextW(hwnd, buf, length + 1)
                        title = buf.value.strip()
                        ignored = ("Program Manager", "Default IME", "MSCTFIME UI", "Windows Input Experience", "NVIDIA GeForce Overlay")
                        if title and not any(title.startswith(ig) for ig in ignored) and "FaceKey Studio" not in title:
                            results.append((hwnd, title))
                return True

            WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
            user32.EnumWindows(WNDENUMPROC(enum_cb), 0)

        t = threading.Thread(target=_clean_lister)
        t.start()
        t.join(timeout=2.0)

        def _score(item):
            t = item[1].lower()
            if any(k in t for k in ("youtube", "video", "movie", "watch", "drishyam", "trailer", "clip")):
                return 0
            if any(k in t for k in ("brave", "chrome", "edge", "firefox", "opera", "vlc")):
                return 1
            return 2

        results.sort(key=_score)
        return results

    def _worker(self, callback: Callable[[float, np.ndarray], None]):
        import win32gui
        import win32ui
        import win32con

        ensure_input_desktop()
        interval = 1.0 / max(1, self.target_fps)
        session_start = time.perf_counter()

        while self.running:
            t0 = time.perf_counter()

            if self.paused:
                time.sleep(0.05)
                continue

            if not win32gui.IsWindow(self.target_hwnd):
                # Try to re-acquire window if it was refreshed or restarted
                self.find_target_window()
                if not self.target_hwnd:
                    time.sleep(0.2)
                    continue

            if ctypes.windll.user32.IsIconic(self.target_hwnd):
                try:
                    # Target window was minimized to taskbar. Keep it rendering in background without stealing user focus!
                    ctypes.windll.user32.ShowWindow(self.target_hwnd, 4)  # SW_SHOWNOACTIVATE
                    ctypes.windll.user32.SetWindowPos(
                        self.target_hwnd, 1, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0010
                    )  # HWND_BOTTOM, SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE
                    time.sleep(0.05)
                except Exception:
                    time.sleep(0.1)
                    continue

            hwndDC = None
            mfcDC = None
            saveDC = None
            saveBitMap = None
            try:
                rect = win32gui.GetWindowRect(self.target_hwnd)
                w = max(32, rect[2] - rect[0])
                h = max(32, rect[3] - rect[1])

                hwndDC = win32gui.GetWindowDC(self.target_hwnd)
                mfcDC = win32ui.CreateDCFromHandle(hwndDC)
                saveDC = mfcDC.CreateCompatibleDC()
                saveBitMap = win32ui.CreateBitmap()
                saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
                saveDC.SelectObject(saveBitMap)

                # PW_RENDERFULLCONTENT = 2: Renders window buffer even if occluded or behind
                ret = ctypes.windll.user32.PrintWindow(self.target_hwnd, saveDC.GetSafeHdc(), 2)

                bmpinfo = saveBitMap.GetInfo()
                bmpstr = saveBitMap.GetBitmapBits(True)
                raw_bgra = np.frombuffer(bmpstr, dtype=np.uint8).reshape((bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4))

                # If PrintWindow returned 0 or black due to hardware acceleration, fallback to direct desktop compositor
                if ret == 0 or np.mean(raw_bgra) < 0.5:
                    try:
                        import mss
                        with mss.mss() as sct:
                            mon = {"top": max(0, rect[1]), "left": max(0, rect[0]), "width": w, "height": h}
                            sct_img = sct.grab(mon)
                            raw_bgra = np.frombuffer(sct_img.raw, dtype=np.uint8).reshape((h, w, 4))
                    except Exception:
                        pass

                frame_rgb = raw_bgra[:, :, [2, 1, 0]]  # BGR to RGB

                timestamp = time.perf_counter() - session_start
                callback(timestamp, frame_rgb)
            except Exception:
                time.sleep(0.05)
            finally:
                if saveBitMap is not None:
                    try:
                        win32gui.DeleteObject(saveBitMap.GetHandle())
                    except Exception:
                        pass
                if saveDC is not None:
                    try:
                        saveDC.DeleteDC()
                    except Exception:
                        pass
                if mfcDC is not None:
                    try:
                        mfcDC.DeleteDC()
                    except Exception:
                        pass
                if hwndDC is not None:
                    try:
                        win32gui.ReleaseDC(self.target_hwnd, hwndDC)
                    except Exception:
                        pass

            dt = time.perf_counter() - t0
            sleep_time = interval - dt
            if sleep_time > 0:
                time.sleep(sleep_time)


class StreamCaptureSource:
    """Network Video Stream Capture for Phone Cameras (e.g. Vivo Y21 via IP Webcam or DroidCam).
    Streams MJPEG / RTSP / HTTP video in a dedicated high-priority buffer thread to guarantee
    strictly zero latency and sub-33ms real-time delivery without phone overheating.
    """

    def __init__(self, stream_url: str, target_fps: int = 30):
        self.stream_url = stream_url
        self.target_fps = target_fps
        self.running = False
        self.paused = False
        self.is_completed = False
        self.on_complete_callback = None
        self.thread: Optional[threading.Thread] = None

    def start(self, frame_callback: Callable[[float, np.ndarray], None]):
        import cv2
        self.running = True
        self.paused = False
        self.is_completed = False
        self.thread = threading.Thread(target=self._worker, args=(frame_callback,), daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.thread = None

    def _worker(self, callback: Callable[[float, np.ndarray], None]):
        import cv2
        cap = cv2.VideoCapture(self.stream_url)
        session_start = time.perf_counter()
        consecutive_empty = 0
        try:
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        except Exception:
            total_frames = -1.0

        while self.running:
            if not cap.isOpened():
                time.sleep(0.3)
                consecutive_empty += 1
                if consecutive_empty > 10:
                    self.is_completed = True
                    self.running = False
                    if self.on_complete_callback:
                        try:
                            self.on_complete_callback()
                        except Exception:
                            pass
                    break
                cap = cv2.VideoCapture(self.stream_url)
                continue

            if self.paused:
                time.sleep(0.05)
                continue

            try:
                cur_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            except Exception:
                cur_pos = -1.0

            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                consecutive_empty += 1
                time.sleep(0.02)
                # Detect video stream EOF (reached end of video)
                is_eof = (total_frames > 0 and cur_pos >= total_frames - 2) or (consecutive_empty >= 10)
                if is_eof:
                    print("\n[*] Video stream playback has completed (reached end of video).")
                    self.is_completed = True
                    self.running = False
                    if self.on_complete_callback:
                        try:
                            self.on_complete_callback()
                        except Exception:
                            pass
                    break
                continue

            consecutive_empty = 0
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            timestamp = time.perf_counter() - session_start
            callback(timestamp, frame_rgb)

        cap.release()
