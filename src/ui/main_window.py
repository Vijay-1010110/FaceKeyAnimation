"""Main PySide6 Studio Control Window with On-Screen HUD Overlay & Live Testing."""

import math
import os
import sys
import threading
import time
from typing import Optional, List, Tuple
import cv2
import numpy as np

from PySide6.QtCore import Qt, QTimer, Signal, Slot, QRect
from PySide6.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QBrush, QPen
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTabWidget, QGroupBox, QComboBox, QCheckBox,
    QProgressBar, QFileDialog, QMessageBox, QFrame, QSplitter
)

from src.config import AppConfig
from src.schema import TrackedFaceFrame, SessionMetadata, EligibilityLevel, FaceRole, AudioFrameData
from src.core.session_manager import SessionManager, SessionState
from src.core.capture import AudioCaptureSource
from src.ui.roi_selector import ROISelectorOverlay
from src.ui.screen_overlay import ScreenMotionOverlay
from src.ui.session_viewer import SessionViewerWindow

try:
    import win32gui
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False

# Feature wireframe loops for OpenCV drawing
CV_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
CV_LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33]
CV_RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 362]
CV_LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
CV_RIGHT_BROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
CV_NOSE = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2]


class AudioLevelMeter(QWidget):
    """Real-time horizontal audio energy & speech VAD activity meter."""

    def __init__(self):
        super().__init__()
        self.setFixedHeight(24)
        self.energy: float = 0.0
        self.is_speech: bool = False

    def set_audio_state(self, energy: float, is_speech: bool):
        self.energy = energy
        self.is_speech = is_speech
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        painter.fillRect(self.rect(), QColor(15, 23, 42))

        # 1. Speech LED
        led_color = QColor(34, 197, 94) if self.is_speech else QColor(71, 85, 105)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(led_color))
        painter.drawEllipse(6, 6, 12, 12)

        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(led_color)
        status_text = "VOICE ACTIVE" if self.is_speech else "AUDIO IDLE"
        painter.drawText(24, 16, status_text)

        # 2. Audio Energy Level Meter
        meter_x = 130
        meter_w = w - meter_x - 10
        if meter_w > 20:
            bar_rect = QRect(meter_x, 6, meter_w, 12)
            painter.setBrush(QBrush(QColor(30, 41, 59)))
            painter.drawRoundedRect(bar_rect, 3, 3)

            # Normalized level (log scaled for sensitivity)
            level = min(1.0, max(0.0, self.energy / 0.06))
            fill_w = int(meter_w * level)
            if fill_w > 0:
                fill_rect = QRect(meter_x, 6, fill_w, 12)
                color = QColor(34, 197, 94) if self.is_speech else QColor(6, 182, 212)
                painter.setBrush(QBrush(color))
                painter.drawRoundedRect(fill_rect, 3, 3)


class MainWindow(QMainWindow):
    """Studio control center with real-time on-screen HUD overlay for YouTube testing."""

    frame_update_signal = Signal(float, list, object, object)
    state_change_signal = Signal(object)
    progress_signal = Signal(float, float)

    def __init__(self, config: Optional[AppConfig] = None):
        super().__init__()
        self.config = config or AppConfig()
        self.session_manager = SessionManager(self.config)

        self.setWindowTitle("Facial Key-Animation Acquisition Studio")
        self.resize(1200, 800)
        self._apply_dark_theme()

        self.roi = (100, 100, 640, 480)
        self.roi_overlay = ROISelectorOverlay()
        self.roi_overlay.roi_selected.connect(self._on_roi_selected)

        # Floating on-screen transparent HUD overlay
        self.screen_overlay = ScreenMotionOverlay()
        self.screen_overlay.update_roi(*self.roi)

        self.session_viewer: Optional[SessionViewerWindow] = None
        self.last_saved_session_dir: str = ""

        # Connect session manager signals
        self.session_manager.on_frame_processed = lambda ts, faces, frame, audio: self.frame_update_signal.emit(ts, faces, frame, audio)
        self.session_manager.on_state_changed = lambda state: self.state_change_signal.emit(state)
        self.session_manager.on_progress = lambda pct, ts: self.progress_signal.emit(pct, ts)

        self.frame_update_signal.connect(self._on_frame_update)
        self.state_change_signal.connect(self._on_state_changed)
        self.progress_signal.connect(self._on_progress_update)

        self._build_ui()

        # Telemetry refresh timer
        self.telemetry_timer = QTimer(self)
        self.telemetry_timer.timeout.connect(self._update_telemetry)
        self.telemetry_timer.start(500)

        # Initialize neural pipeline in background thread
        threading.Thread(target=self._init_pipeline, daemon=True).start()

    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #0b0f19; color: #f8fafc; }
            QLabel { color: #f8fafc; font-family: 'Segoe UI'; }
            QTabWidget::pane { border: 1px solid #1e293b; background-color: #0f172a; border-radius: 8px; }
            QTabBar::tab {
                background: #1e293b; color: #94a3b8; padding: 10px 22px;
                border-top-left-radius: 6px; border-top-right-radius: 6px;
                font-weight: bold; margin-right: 4px;
            }
            QTabBar::tab:selected { background: #0f172a; color: #38bdf8; border-bottom: 2px solid #38bdf8; }
            QPushButton {
                background-color: #1e293b; color: #f8fafc;
                border: 1px solid #334155; border-radius: 6px;
                padding: 8px 16px; font-weight: bold; font-family: 'Segoe UI';
            }
            QPushButton:hover { background-color: #334155; }
            QPushButton:disabled { background-color: #0f172a; color: #475569; border-color: #1e293b; }
            QGroupBox {
                border: 1px solid #1e293b; border-radius: 8px;
                margin-top: 10px; font-weight: bold; color: #94a3b8;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QComboBox {
                background-color: #1e293b; color: #f8fafc;
                border: 1px solid #334155; border-radius: 6px; padding: 6px 12px;
            }
            QProgressBar {
                border: 1px solid #334155; border-radius: 6px; text-align: center;
                background-color: #1e293b; color: #f8fafc; font-weight: bold;
            }
            QProgressBar::chunk { background-color: #06b6d4; border-radius: 5px; }
        """)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # Header Bar
        header = QHBoxLayout()
        title_box = QVBoxLayout()
        lbl_title = QLabel("Facial Key-Animation Acquisition Studio")
        lbl_title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        lbl_subtitle = QLabel("Real-time low-resource motion capture & speaker attribution")
        lbl_subtitle.setStyleSheet("color: #64748b; font-size: 12px;")
        title_box.addWidget(lbl_title)
        title_box.addWidget(lbl_subtitle)
        header.addLayout(title_box, stretch=1)

        # Toggle Button for On-Screen Floating Overlay
        self.btn_toggle_overlay = QPushButton("🌟 Float On-Screen Overlay (On YouTube)")
        self.btn_toggle_overlay.setCheckable(True)
        self.btn_toggle_overlay.setStyleSheet("""
            QPushButton { background-color: #0369a1; color: white; border: 1px solid #0284c7; padding: 8px 16px; border-radius: 6px; font-weight: bold; }
            QPushButton:checked { background-color: #0891b2; border: 2px solid #22d3ee; }
        """)
        self.btn_toggle_overlay.toggled.connect(self._on_toggle_screen_overlay)
        header.addWidget(self.btn_toggle_overlay)

        self.lbl_system_state = QLabel("STATUS: INITIALIZING")
        self.lbl_system_state.setStyleSheet("""
            background-color: #1e293b; color: #f59e0b;
            padding: 8px 16px; border-radius: 6px; font-weight: bold; font-size: 13px;
        """)
        header.addWidget(self.lbl_system_state)

        self.btn_open_viewer = QPushButton("🔍 Dataset Inspector")
        self.btn_open_viewer.clicked.connect(self._open_session_viewer)
        header.addWidget(self.btn_open_viewer)

        main_layout.addLayout(header)

        # Center Splitter: Left Controls & Mode Tabs, Right Live HUD & Telemetry
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ----------------- Left Panel: Controls & Modes -----------------
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # ROI & Source Selection Box
        roi_box = QGroupBox("Capture Source & Target Region")
        roi_layout = QVBoxLayout(roi_box)

        roi_btn_row = QHBoxLayout()
        self.btn_select_roi = QPushButton("📐 Select Screen ROI (Click & Drag)")
        self.btn_select_roi.clicked.connect(self._start_roi_selection)
        roi_btn_row.addWidget(self.btn_select_roi)

        self.lbl_roi_coords = QLabel(f"ROI: {self.roi[2]}x{self.roi[3]} at ({self.roi[0]}, {self.roi[1]})")
        self.lbl_roi_coords.setStyleSheet("color: #38bdf8; font-weight: bold;")
        roi_btn_row.addWidget(self.lbl_roi_coords, stretch=1)
        roi_layout.addLayout(roi_btn_row)

        # Window picker row
        win_row = QHBoxLayout()
        win_row.addWidget(QLabel("Target Window:"))
        self.cmb_windows = QComboBox()
        self.cmb_windows.addItem("None (Use Screen Coordinates)", None)
        self._populate_windows()
        self.cmb_windows.currentIndexChanged.connect(self._on_window_selected)
        win_row.addWidget(self.cmb_windows, stretch=1)
        self.btn_refresh_wins = QPushButton("🔄")
        self.btn_refresh_wins.setFixedWidth(36)
        self.btn_refresh_wins.clicked.connect(self._populate_windows)
        win_row.addWidget(self.btn_refresh_wins)
        roi_layout.addLayout(win_row)

        # Audio source picker row
        audio_row = QHBoxLayout()
        audio_row.addWidget(QLabel("Audio Device:"))
        self.cmb_audio = QComboBox()
        self.cmb_audio.addItem("Default Audio Input (Mic/System)", None)
        devices = AudioCaptureSource.get_input_devices()
        for idx, name in devices.items():
            self.cmb_audio.addItem(name, idx)
        audio_row.addWidget(self.cmb_audio, stretch=1)
        roi_layout.addLayout(audio_row)

        # Live Audio Energy & Speech Meter
        self.audio_meter = AudioLevelMeter()
        roi_layout.addWidget(self.audio_meter)

        left_layout.addWidget(roi_box)

        # Mode Selection Tabs
        self.tabs = QTabWidget()

        # Tab 1: Live Sensor Testing & Manual Recording
        tab_manual = QWidget()
        t1_layout = QVBoxLayout(tab_manual)
        t1_layout.setContentsMargins(12, 16, 12, 12)

        lbl_manual_info = QLabel("Interactive testing on YouTube / Podcasts:\nRun live sensor to watch motion dots & speaker attribution in real-time.")
        lbl_manual_info.setWordWrap(True)
        lbl_manual_info.setStyleSheet("color: #94a3b8; font-size: 12px; margin-bottom: 8px;")
        t1_layout.addWidget(lbl_manual_info)

        # Live Sensor Test Buttons (Zero Disk Writes)
        test_btn_row = QHBoxLayout()
        self.btn_live_sensor = QPushButton("▶ START LIVE SENSOR (Test on YouTube)")
        self.btn_live_sensor.setStyleSheet("background-color: #0284c7; color: white; font-size: 13px; padding: 10px 16px;")
        self.btn_live_sensor.clicked.connect(self._on_start_live_sensor)
        test_btn_row.addWidget(self.btn_live_sensor)

        self.btn_stop_sensor = QPushButton("⏹ STOP SENSOR")
        self.btn_stop_sensor.setEnabled(False)
        self.btn_stop_sensor.clicked.connect(self._on_stop_session)
        test_btn_row.addWidget(self.btn_stop_sensor)
        t1_layout.addLayout(test_btn_row)

        # Dataset Recording Buttons
        rec_box = QGroupBox("Record Dataset (Save Structured Motion)")
        rec_layout = QVBoxLayout(rec_box)
        rec_btns = QHBoxLayout()
        self.btn_record = QPushButton("● START RECORDING SESSION")
        self.btn_record.setStyleSheet("background-color: #dc2626; color: white; font-size: 13px; padding: 10px 16px;")
        self.btn_record.clicked.connect(self._on_start_manual)
        rec_btns.addWidget(self.btn_record)

        self.btn_stop_rec = QPushButton("■ STOP & SAVE")
        self.btn_stop_rec.setEnabled(False)
        self.btn_stop_rec.clicked.connect(self._on_stop_session)
        rec_btns.addWidget(self.btn_stop_rec)
        rec_layout.addLayout(rec_btns)
        t1_layout.addWidget(rec_box)

        self.lbl_session_duration = QLabel("Duration: 00:00.00 | Captured: 0 samples")
        self.lbl_session_duration.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        self.lbl_session_duration.setStyleSheet("color: #34d399; margin-top: 6px;")
        t1_layout.addWidget(self.lbl_session_duration)
        t1_layout.addStretch()
        self.tabs.addTab(tab_manual, "Live Sensor & Record")

        # Tab 2: Mode A - Automatic Watcher
        tab_watcher = QWidget()
        t2_layout = QVBoxLayout(tab_watcher)
        t2_layout.setContentsMargins(12, 16, 12, 12)

        lbl_watcher_info = QLabel("Mode A: Passive background watcher. Automatically starts recording with pre-roll buffer when a face is detected.")
        lbl_watcher_info.setWordWrap(True)
        lbl_watcher_info.setStyleSheet("color: #94a3b8; font-size: 12px; margin-bottom: 8px;")
        t2_layout.addWidget(lbl_watcher_info)

        watcher_btns = QHBoxLayout()
        self.btn_arm_watcher = QPushButton("🛡️ ARM WATCHER")
        self.btn_arm_watcher.setStyleSheet("background-color: #0284c7; color: white; font-size: 13px; padding: 10px 18px;")
        self.btn_arm_watcher.clicked.connect(self._on_arm_watcher)
        watcher_btns.addWidget(self.btn_arm_watcher)

        self.btn_disarm_watcher = QPushButton("⏹ DISARM")
        self.btn_disarm_watcher.setEnabled(False)
        self.btn_disarm_watcher.clicked.connect(self._on_stop_session)
        watcher_btns.addWidget(self.btn_disarm_watcher)
        t2_layout.addLayout(watcher_btns)

        self.lbl_watcher_status = QLabel("Watcher State: IDLE")
        self.lbl_watcher_status.setStyleSheet("color: #94a3b8; margin-top: 8px;")
        t2_layout.addWidget(self.lbl_watcher_status)
        t2_layout.addStretch()
        self.tabs.addTab(tab_watcher, "Mode A: Watcher")

        # Tab 3: Mode C - Uploaded Video
        tab_video = QWidget()
        t3_layout = QVBoxLayout(tab_video)
        t3_layout.setContentsMargins(12, 16, 12, 12)

        lbl_video_info = QLabel("Mode C: Process video file in memory without intermediate disk writes.")
        lbl_video_info.setStyleSheet("color: #94a3b8; font-size: 12px; margin-bottom: 8px;")
        t3_layout.addWidget(lbl_video_info)

        v_file_row = QHBoxLayout()
        self.btn_select_video = QPushButton("📁 Choose Video File...")
        self.btn_select_video.clicked.connect(self._on_select_video)
        v_file_row.addWidget(self.btn_select_video)
        self.lbl_selected_video = QLabel("No video selected")
        self.lbl_selected_video.setStyleSheet("color: #e2e8f0;")
        v_file_row.addWidget(self.lbl_selected_video, stretch=1)
        t3_layout.addLayout(v_file_row)

        self.btn_process_video = QPushButton("⚡ Process Video Offline")
        self.btn_process_video.setEnabled(False)
        self.btn_process_video.setStyleSheet("background-color: #059669; color: white; font-size: 13px; padding: 10px 18px;")
        self.btn_process_video.clicked.connect(self._on_start_video_processing)
        t3_layout.addWidget(self.btn_process_video)

        self.video_progress = QProgressBar()
        self.video_progress.setValue(0)
        t3_layout.addWidget(self.video_progress)
        t3_layout.addStretch()
        self.tabs.addTab(tab_video, "Mode C: Upload Video")

        left_layout.addWidget(self.tabs)
        splitter.addWidget(left_panel)

        # ----------------- Right Panel: Live HUD & Telemetry -----------------
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Live HUD Diagnostic Monitor
        hud_group = QGroupBox("Live Tracking Viewport (Studio Feed)")
        hud_layout = QVBoxLayout(hud_group)

        self.chk_debug_mode = QCheckBox("Enable Viewport Rendering (Uncheck to minimize CPU)")
        self.chk_debug_mode.setChecked(True)
        hud_layout.addWidget(self.chk_debug_mode)

        self.lbl_video_feed = QLabel()
        self.lbl_video_feed.setMinimumSize(540, 360)
        self.lbl_video_feed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_video_feed.setStyleSheet("background-color: #020617; border-radius: 6px;")
        self.lbl_video_feed.setText("Live Feed Ready\nClick 'START LIVE SENSOR' or 'START RECORDING'")
        hud_layout.addWidget(self.lbl_video_feed, stretch=1)

        self.lbl_hud_diagnostics = QLabel("Faces: 0 | Active Speaker: None")
        self.lbl_hud_diagnostics.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        self.lbl_hud_diagnostics.setStyleSheet("color: #38bdf8;")
        hud_layout.addWidget(self.lbl_hud_diagnostics)

        right_layout.addWidget(hud_group)

        # Telemetry & Resource Monitor
        telemetry_group = QGroupBox("Resource Instrumentation")
        t_layout = QVBoxLayout(telemetry_group)

        self.lbl_telemetry_cpu_ram = QLabel("CPU: 0.0%  |  RAM: 0 MB  |  GPU: 0%  |  VRAM: 0 MB")
        self.lbl_telemetry_cpu_ram.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        t_layout.addWidget(self.lbl_telemetry_cpu_ram)

        self.lbl_telemetry_fps_latency = QLabel("Capture: 0.0 FPS  |  Inference: 0.0 FPS  |  Latency: 0.0 ms  |  Drops: 0")
        self.lbl_telemetry_fps_latency.setFont(QFont("Consolas", 10))
        self.lbl_telemetry_fps_latency.setStyleSheet("color: #34d399;")
        t_layout.addWidget(self.lbl_telemetry_fps_latency)

        right_layout.addWidget(telemetry_group)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        main_layout.addWidget(splitter, stretch=1)

    def _init_pipeline(self):
        try:
            self.session_manager.initialize()
            self._set_ui_status("READY", "#10b981")
        except Exception as e:
            self._set_ui_status("MODEL ERROR", "#ef4444")

    def _set_ui_status(self, text: str, color: str):
        self.lbl_system_state.setText(f"STATUS: {text}")
        self.lbl_system_state.setStyleSheet(f"""
            background-color: #1e293b; color: {color};
            padding: 8px 16px; border-radius: 6px; font-weight: bold; font-size: 13px;
        """)

    def _populate_windows(self):
        self.cmb_windows.clear()
        self.cmb_windows.addItem("None (Use Screen Coordinates)", None)
        if not HAS_WIN32:
            return

        def enum_cb(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd).strip()
                if title and len(title) > 2 and title != "Program Manager":
                    self.cmb_windows.addItem(f"{title[:45]} (hwnd:{hwnd})", hwnd)

        try:
            win32gui.EnumWindows(enum_cb, None)
        except Exception:
            pass

    def _on_window_selected(self, idx: int):
        hwnd = self.cmb_windows.currentData()
        if hwnd and HAS_WIN32:
            try:
                rect = win32gui.GetWindowRect(hwnd)
                x, y = rect[0], rect[1]
                w = max(64, rect[2] - rect[0])
                h = max(64, rect[3] - rect[1])
                self.roi = (x, y, w, h)
                self.lbl_roi_coords.setText(f"ROI: {w}x{h} at ({x}, {y})")
                self.screen_overlay.update_roi(x, y, w, h)
            except Exception:
                pass

    def _start_roi_selection(self):
        self.roi_overlay.start_selection()

    def _on_roi_selected(self, x: int, y: int, w: int, h: int):
        self.roi = (x, y, w, h)
        self.lbl_roi_coords.setText(f"ROI: {w}x{h} at ({x}, {y})")
        self.screen_overlay.update_roi(x, y, w, h)

    def _on_toggle_screen_overlay(self, checked: bool):
        if checked:
            self.screen_overlay.update_roi(*self.roi)
            self.screen_overlay.show()
        else:
            self.screen_overlay.hide()

    def _on_start_live_sensor(self):
        # Auto-enable floating HUD overlay if not already active
        if not self.btn_toggle_overlay.isChecked():
            self.btn_toggle_overlay.setChecked(True)

        hwnd = self.cmb_windows.currentData()
        audio_idx = self.cmb_audio.currentData()
        self.session_manager.start_live_sensor(roi=self.roi, target_hwnd=hwnd, audio_device_idx=audio_idx)
        self.btn_live_sensor.setEnabled(False)
        self.btn_stop_sensor.setEnabled(True)
        self.btn_record.setEnabled(False)

    def _on_start_manual(self):
        hwnd = self.cmb_windows.currentData()
        audio_idx = self.cmb_audio.currentData()
        self.session_manager.start_manual_session(roi=self.roi, target_hwnd=hwnd, audio_device_idx=audio_idx)
        self.btn_record.setEnabled(False)
        self.btn_stop_rec.setEnabled(True)
        self.btn_live_sensor.setEnabled(False)

    def _on_arm_watcher(self):
        hwnd = self.cmb_windows.currentData()
        audio_idx = self.cmb_audio.currentData()
        self.session_manager.arm_watcher(roi=self.roi, target_hwnd=hwnd, audio_device_idx=audio_idx)
        self.btn_arm_watcher.setEnabled(False)
        self.btn_disarm_watcher.setEnabled(True)

    def _on_stop_session(self):
        saved_dir = self.session_manager.stop_session()
        self.btn_live_sensor.setEnabled(True)
        self.btn_stop_sensor.setEnabled(False)
        self.btn_record.setEnabled(True)
        self.btn_stop_rec.setEnabled(False)
        self.btn_arm_watcher.setEnabled(True)
        self.btn_disarm_watcher.setEnabled(False)

        if saved_dir:
            self.last_saved_session_dir = saved_dir
            QMessageBox.information(
                self,
                "Session Finalized & Saved",
                f"Session dataset saved cleanly with zero disk video frames to:\n{saved_dir}\n\nYou can inspect it in Dataset Inspector."
            )

    def _on_select_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.mkv *.avi *.mov *.webm)")
        if path:
            self.selected_video_path = path
            self.lbl_selected_video.setText(os.path.basename(path))
            self.btn_process_video.setEnabled(True)

    def _on_start_video_processing(self):
        if not hasattr(self, "selected_video_path") or not self.selected_video_path:
            return
        self.btn_process_video.setEnabled(False)
        self.video_progress.setValue(0)

        def run_video():
            saved_dir = self.session_manager.process_video_file(self.selected_video_path)
            if saved_dir:
                self.last_saved_session_dir = saved_dir

        threading.Thread(target=run_video, daemon=True).start()

    @Slot(object)
    def _on_state_changed(self, state: SessionState):
        color_map = {
            SessionState.IDLE: ("IDLE", "#94a3b8"),
            SessionState.ARMED: ("ARMED (WATCHER)", "#0284c7"),
            SessionState.RECORDING: ("RECORDING" if self.session_manager.mode != "live_sensor" else "LIVE TRACKING", "#dc2626" if self.session_manager.mode != "live_sensor" else "#0284c7"),
            SessionState.FINALIZING: ("FINALIZING", "#f59e0b"),
            SessionState.SAVED: ("SAVED", "#10b981")
        }
        name, color = color_map.get(state, ("UNKNOWN", "#94a3b8"))
        self._set_ui_status(name, color)

    @Slot(float, float)
    def _on_progress_update(self, pct: float, cur_ts: float):
        self.video_progress.setValue(int(pct * 100))

    @Slot(float, list, object, object)
    def _on_frame_update(
        self,
        ts: float,
        faces: List[TrackedFaceFrame],
        frame_rgb: Optional[np.ndarray],
        audio_frame: Optional[AudioFrameData]
    ):
        mins, secs = divmod(ts, 60.0)
        mode_label = "Live Sensor" if self.session_manager.mode == "live_sensor" else "Recording"
        self.lbl_session_duration.setText(
            f"{mode_label}: {int(mins):02d}:{secs:05.2f} | Captured: {len(self.session_manager.accumulated_face_frames)} samples"
        )

        is_speech = audio_frame.is_speech if audio_frame else False
        energy = audio_frame.energy_rms if audio_frame else 0.0
        self.audio_meter.set_audio_state(energy, is_speech)

        fps = self.session_manager.profiler.get_snapshot().get("inference_fps", 0.0)

        # 1. Update on-screen floating HUD overlay
        if self.screen_overlay.isVisible():
            self.screen_overlay.update_tracking_data(faces, is_speech, energy, fps)

        # 2. Render inside Studio Viewport
        if self.chk_debug_mode.isChecked() and frame_rgb is not None:
            disp_img = frame_rgb.copy()
            h, w = disp_img.shape[:2]

            hud_text = f"Faces: {len(faces)}"
            for face in faces:
                x, y, fw, fh = face.bbox
                is_speaker = (face.role == FaceRole.SPEAKER)
                is_listener = (face.role == FaceRole.LISTENER)

                # Theme color (RGB)
                color = (34, 197, 94) if is_speaker else ((56, 189, 248) if is_listener else (148, 163, 184))

                # Draw bounding box
                cv2.rectangle(disp_img, (x, y), (x + fw, y + fh), color, 2 if is_speaker else 1)

                # Draw role tag
                tag = f"Face {face.face_id} [{face.role.value}]"
                if is_speaker:
                    tag += f" ({int(face.speaker_probability * 100)}%)"
                cv2.putText(disp_img, tag, (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2 if is_speaker else 1, cv2.LINE_AA)

                lms = face.clean.landmarks
                if len(lms) >= 468:
                    # Draw feature loops
                    for loop in [CV_LIPS, CV_LEFT_EYE, CV_RIGHT_EYE, CV_LEFT_BROW, CV_RIGHT_BROW, CV_NOSE]:
                        pts = [(int(lms[idx][0] * w), int(lms[idx][1] * h)) for idx in loop if idx < len(lms)]
                        for i in range(len(pts) - 1):
                            cv2.line(disp_img, pts[i], pts[i+1], color, 1, cv2.LINE_AA)

                    # Iris points (Gold)
                    for iris_idx in [468, 473]:
                        if iris_idx < len(lms):
                            cv2.circle(disp_img, (int(lms[iris_idx][0] * w), int(lms[iris_idx][1] * h)), 3, (234, 179, 8), -1)

                    # Subsampled dots
                    for pt in lms[::6]:
                        px = int(pt[0] * w)
                        py = int(pt[1] * h)
                        cv2.circle(disp_img, (px, py), 2, (255, 255, 255), -1)

                    # Head pose orientation
                    nose = (int(lms[1][0] * w), int(lms[1][1] * h))
                    pitch, yaw, roll = face.clean.head_pose_euler
                    rad = math.pi / 180.0
                    p_r, y_r, r_r = pitch * rad, yaw * rad, roll * rad
                    axis_len = 30.0

                    dx_x = int(axis_len * math.cos(y_r) * math.cos(r_r))
                    dy_x = int(axis_len * (math.sin(p_r) * math.sin(y_r) * math.cos(r_r) - math.cos(p_r) * math.sin(r_r)))
                    cv2.line(disp_img, nose, (nose[0] + dx_x, nose[1] + dy_x), (239, 68, 68), 2)  # Red X

                    dx_y = int(-axis_len * math.cos(y_r) * math.sin(r_r))
                    dy_y = int(-axis_len * (math.sin(p_r) * math.sin(y_r) * math.sin(r_r) + math.cos(p_r) * math.cos(r_r)))
                    cv2.line(disp_img, nose, (nose[0] + dx_y, nose[1] + dy_y), (34, 197, 94), 2)  # Green Y

                    dx_z = int(-axis_len * math.sin(y_r))
                    dy_z = int(axis_len * math.sin(p_r) * math.cos(y_r))
                    cv2.line(disp_img, nose, (nose[0] + dx_z, nose[1] + dy_z), (59, 130, 246), 2)  # Blue Z

                hud_text += f" | F{face.face_id}: {face.role.value}"

            self.lbl_hud_diagnostics.setText(hud_text)

            # Convert to QPixmap and display
            bytes_per_line = 3 * w
            qimg = QImage(disp_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(
                self.lbl_video_feed.width(),
                self.lbl_video_feed.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.lbl_video_feed.setPixmap(pix)

    def _update_telemetry(self):
        snap = self.session_manager.profiler.get_snapshot()
        self.lbl_telemetry_cpu_ram.setText(
            f"CPU: {snap['cpu_percent']}%  |  RAM (RSS): {snap['ram_rss_mb']} MB  |  GPU: {snap['gpu_util_pct']}%  |  VRAM: {snap['vram_used_mb']}/{snap['vram_total_mb']} MB"
        )
        self.lbl_telemetry_fps_latency.setText(
            f"Capture: {snap['capture_fps']} FPS  |  Inference: {snap['inference_fps']} FPS  |  Latency: {snap['avg_latency_ms']} ms  |  Drops: {snap['dropped_frames']}"
        )

    def _open_session_viewer(self):
        target_dir = self.last_saved_session_dir if self.last_saved_session_dir else "sessions"
        self.session_viewer = SessionViewerWindow(target_dir if os.path.exists(target_dir) else None)
        self.session_viewer.show()

    def closeEvent(self, event):
        self.screen_overlay.close()
        self.session_manager.close()
        event.accept()
