"""Interactive Facial Motion Dataset Inspector and Timeline Scrubber."""

import math
import os
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from PySide6.QtCore import Qt, QTimer, QRect, QPointF
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPolygonF
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSlider, QComboBox, QFileDialog, QSplitter,
    QProgressBar, QGroupBox, QCheckBox, QMessageBox, QTabWidget
)

from src.storage.dataset_reader import DatasetReader
from src.schema import EligibilityLevel, FaceRole


# Standard MediaPipe Face Mesh landmark connection loops
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 362]
LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
NOSE_CONTOUR = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]


class FaceMeshCanvas(QWidget):
    """Canvas rendering 2D/3D facial landmarks, mesh wireframe, and head pose orientation."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 400)
        self.raw_landmarks: Optional[np.ndarray] = None
        self.clean_landmarks: Optional[np.ndarray] = None
        self.head_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.show_raw: bool = False
        self.show_clean: bool = True
        self.show_wireframe: bool = True
        self.show_pose_axes: bool = True

    def set_data(self, raw_lms: Optional[np.ndarray], clean_lms: Optional[np.ndarray], pose: Tuple[float, float, float]):
        self.raw_landmarks = raw_lms
        self.clean_landmarks = clean_lms
        self.head_pose = pose
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Dark sleek canvas background
        painter.fillRect(self.rect(), QColor(15, 23, 42))

        w, h = self.width(), self.height()
        cx, cy = w / 2.0, h / 2.0
        scale = min(w, h) * 0.75

        # Grid lines
        painter.setPen(QPen(QColor(30, 41, 59), 1))
        painter.drawLine(int(cx), 0, int(cx), h)
        painter.drawLine(0, int(cy), w, int(cy))

        lms = self.clean_landmarks if self.show_clean and self.clean_landmarks is not None else self.raw_landmarks
        if lms is None or len(lms) < 468:
            painter.setFont(QFont("Segoe UI", 12))
            painter.setPen(QColor(100, 116, 139))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Facial Motion Data")
            return

        # Helper mapping normalized [0, 1] landmark to canvas pixel
        def to_canvas(pt: np.ndarray) -> QPointF:
            # pt[0] is x in [0, 1], pt[1] is y in [0, 1]
            px = cx + (pt[0] - 0.5) * scale
            py = cy + (pt[1] - 0.5) * scale
            return QPointF(px, py)

        # 1. Draw Mesh Wireframe Loops
        if self.show_wireframe:
            wire_pen = QPen(QColor(56, 189, 248, 120), 1.5)
            painter.setPen(wire_pen)
            for loop in [FACE_OVAL, LIPS_OUTER, LIPS_INNER, LEFT_EYE, RIGHT_EYE, LEFT_EYEBROW, RIGHT_EYEBROW, NOSE_CONTOUR]:
                poly = QPolygonF()
                for idx in loop:
                    if idx < len(lms):
                        poly.append(to_canvas(lms[idx]))
                painter.drawPolyline(poly)

        # 2. Draw Raw Landmarks (Orange/Red) if enabled
        if self.show_raw and self.raw_landmarks is not None:
            raw_pen = QPen(QColor(249, 115, 22, 180), 3)
            painter.setPen(raw_pen)
            for pt in self.raw_landmarks[::2]:  # Subsample for rendering speed
                painter.drawPoint(to_canvas(pt))

        # 3. Draw Clean Filtered Landmarks (Neon Green/Cyan)
        if self.show_clean and self.clean_landmarks is not None:
            clean_pen = QPen(QColor(52, 211, 153), 3)
            painter.setPen(clean_pen)
            for pt in self.clean_landmarks[::2]:
                painter.drawPoint(to_canvas(pt))

            # Highlight irises
            painter.setPen(QPen(QColor(234, 179, 8), 5))
            for iris_idx in [468, 473]:
                if iris_idx < len(self.clean_landmarks):
                    painter.drawPoint(to_canvas(self.clean_landmarks[iris_idx]))

        # 4. Draw 3D Head Pose Axes
        if self.show_pose_axes and len(lms) > 1:
            nose_pt = to_canvas(lms[1])  # Landmark 1 is nose tip
            pitch, yaw, roll = self.head_pose
            rad = math.pi / 180.0
            p_rad, y_rad, r_rad = pitch * rad, yaw * rad, roll * rad

            axis_len = scale * 0.25
            # Simplified projection of Euler angles
            # X axis (Pitch/Roll - Red)
            dx_x = axis_len * math.cos(y_rad) * math.cos(r_rad)
            dy_x = axis_len * (math.sin(p_rad) * math.sin(y_rad) * math.cos(r_rad) - math.cos(p_rad) * math.sin(r_rad))
            painter.setPen(QPen(QColor(239, 68, 68), 3))
            painter.drawLine(nose_pt, QPointF(nose_pt.x() + dx_x, nose_pt.y() + dy_x))

            # Y axis (Yaw/Pitch - Green)
            dx_y = -axis_len * math.cos(y_rad) * math.sin(r_rad)
            dy_y = -axis_len * (math.sin(p_rad) * math.sin(y_rad) * math.sin(r_rad) + math.cos(p_rad) * math.cos(r_rad))
            painter.setPen(QPen(QColor(34, 197, 94), 3))
            painter.drawLine(nose_pt, QPointF(nose_pt.x() + dx_y, nose_pt.y() + dy_y))

            # Z axis (Forward gaze - Blue)
            dx_z = -axis_len * math.sin(y_rad)
            dy_z = axis_len * math.sin(p_rad) * math.cos(y_rad)
            painter.setPen(QPen(QColor(59, 130, 246), 3))
            painter.drawLine(nose_pt, QPointF(nose_pt.x() + dx_z, nose_pt.y() + dy_z))


class BlendshapeMeterWidget(QWidget):
    """Visualizes live blendshape activation values as animated horizontal bars."""

    TRACKED_SHAPES = [
        "jawOpen", "mouthSmileLeft", "mouthSmileRight",
        "eyeBlinkLeft", "eyeBlinkRight", "browInnerUp",
        "browDownLeft", "browDownRight", "mouthPucker", "mouthFunnel"
    ]

    def __init__(self):
        super().__init__()
        self.setMinimumWidth(260)
        self.values: Dict[str, float] = {}

    def set_blendshapes(self, bs_dict: Dict[str, float]):
        self.values = bs_dict
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(15, 23, 42))

        bar_h = 16
        spacing = 10
        y = 15
        w = self.width() - 30

        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))

        for name in self.TRACKED_SHAPES:
            val = max(0.0, min(1.0, self.values.get(name, 0.0)))

            # Label text & value
            painter.setPen(QColor(226, 232, 240))
            painter.drawText(15, y + 12, f"{name}: {val:.2f}")

            # Background bar
            bar_rect = QRect(15, y + 18, w, bar_h)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(30, 41, 59)))
            painter.drawRoundedRect(bar_rect, 4, 4)

            # Active fill bar
            fill_w = int(w * val)
            if fill_w > 0:
                fill_rect = QRect(15, y + 18, fill_w, bar_h)
                # Gradient color: cyan to green
                color = QColor(6, 182, 212) if val < 0.5 else QColor(16, 185, 129)
                painter.setBrush(QBrush(color))
                painter.drawRoundedRect(fill_rect, 4, 4)

            y += bar_h + spacing + 18


class SessionViewerWindow(QMainWindow):
    """Full-featured interactive session inspector and dataset viewer."""

    def __init__(self, session_dir: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("Facial Motion Dataset Inspector")
        self.resize(1100, 750)
        self.setStyleSheet("""
            QMainWindow { background-color: #0b0f19; color: #f8fafc; }
            QLabel { color: #f8fafc; font-family: 'Segoe UI'; }
            QPushButton {
                background-color: #1e293b; color: #f8fafc;
                border: 1px solid #334155; border-radius: 6px;
                padding: 6px 14px; font-weight: bold;
            }
            QPushButton:hover { background-color: #334155; }
            QSlider::groove:horizontal { height: 6px; background: #334155; border-radius: 3px; }
            QSlider::sub-page:horizontal { background: #06b6d4; border-radius: 3px; }
            QSlider::handle:horizontal {
                background: #f8fafc; width: 16px; margin-top: -5px;
                margin-bottom: -5px; border-radius: 8px;
            }
            QGroupBox {
                border: 1px solid #1e293b; border-radius: 8px;
                margin-top: 10px; font-weight: bold; color: #94a3b8;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
        """)

        self.reader: Optional[DatasetReader] = None
        self.current_index: int = 0
        self.is_playing: bool = False

        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self._on_playback_step)

        self._build_ui()

        if session_dir and os.path.exists(session_dir):
            self.load_session(session_dir)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(12)

        # Top Bar: Session Info & Load / Export
        top_bar = QHBoxLayout()
        self.btn_load = QPushButton("📂 Open Session Folder")
        self.btn_load.clicked.connect(self._on_browse_session)
        top_bar.addWidget(self.btn_load)

        self.lbl_session_name = QLabel("No Session Loaded")
        self.lbl_session_name.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        top_bar.addWidget(self.lbl_session_name, stretch=1)

        self.btn_export = QPushButton("💾 Export Dataset (JSONL)")
        self.btn_export.clicked.connect(self._on_export_dataset)
        top_bar.addWidget(self.btn_export)

        main_layout.addLayout(top_bar)

        # Main Splitter: Canvas (Left) + Blendshapes & Metadata (Right)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left Column: Canvas + View Options
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = FaceMeshCanvas()
        left_layout.addWidget(self.canvas, stretch=1)

        # Checkbox toggles
        opts_layout = QHBoxLayout()
        self.chk_clean = QCheckBox("Clean (One Euro)")
        self.chk_clean.setChecked(True)
        self.chk_clean.stateChanged.connect(self._update_canvas_flags)
        opts_layout.addWidget(self.chk_clean)

        self.chk_raw = QCheckBox("Raw Observed")
        self.chk_raw.setChecked(False)
        self.chk_raw.stateChanged.connect(self._update_canvas_flags)
        opts_layout.addWidget(self.chk_raw)

        self.chk_wire = QCheckBox("Wireframe")
        self.chk_wire.setChecked(True)
        self.chk_wire.stateChanged.connect(self._update_canvas_flags)
        opts_layout.addWidget(self.chk_wire)

        self.chk_pose = QCheckBox("Head Axes")
        self.chk_pose.setChecked(True)
        self.chk_pose.stateChanged.connect(self._update_canvas_flags)
        opts_layout.addWidget(self.chk_pose)
        left_layout.addLayout(opts_layout)

        splitter.addWidget(left_widget)

        # Right Column: Live Blendshapes & Diagnostic Cards
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Diagnostics Card
        diag_group = QGroupBox("Observation & Speaker Provenance")
        diag_layout = QVBoxLayout(diag_group)
        self.lbl_role_badge = QLabel("Role: UNKNOWN | Speaker Prob: 0.00")
        self.lbl_role_badge.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 13px;")
        diag_layout.addWidget(self.lbl_role_badge)

        self.lbl_level_badge = QLabel("Eligibility: LEVEL 0 (Detected)")
        self.lbl_level_badge.setStyleSheet("color: #a78bfa; font-weight: bold;")
        diag_layout.addWidget(self.lbl_level_badge)

        self.lbl_quality_stats = QLabel("Quality: 0.00 | Blur: 0.0 | Jitter: 0.0")
        diag_layout.addWidget(self.lbl_quality_stats)

        self.lbl_pose_stats = QLabel("Head Pose: Yaw=0.0°, Pitch=0.0°, Roll=0.0°")
        diag_layout.addWidget(self.lbl_pose_stats)
        right_layout.addWidget(diag_group)

        # Blendshapes Group
        bs_group = QGroupBox("Key Blendshape Activations")
        bs_layout = QVBoxLayout(bs_group)
        self.bs_meter = BlendshapeMeterWidget()
        bs_layout.addWidget(self.bs_meter)
        right_layout.addWidget(bs_group, stretch=1)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        main_layout.addWidget(splitter, stretch=1)

        # Timeline Scrubber & Playback Controls
        bottom_box = QGroupBox("Session Master Timeline")
        bottom_layout = QVBoxLayout(bottom_box)

        # Timeline slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self._on_slider_moved)
        bottom_layout.addWidget(self.slider)

        # Playback buttons and time readout
        ctrls_layout = QHBoxLayout()
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.clicked.connect(self._toggle_playback)
        ctrls_layout.addWidget(self.btn_play)

        self.btn_prev = QPushButton("⏮ Prev")
        self.btn_prev.clicked.connect(self._step_prev)
        ctrls_layout.addWidget(self.btn_prev)

        self.btn_next = QPushButton("⏭ Next")
        self.btn_next.clicked.connect(self._step_next)
        ctrls_layout.addWidget(self.btn_next)

        self.lbl_time = QLabel("00:00.00 / 00:00.00 (Frame 0/0)")
        self.lbl_time.setFont(QFont("Consolas", 10))
        ctrls_layout.addWidget(self.lbl_time, stretch=1)

        # Eligibility Filter Combobox
        ctrls_layout.addWidget(QLabel("Filter:"))
        self.cmb_filter = QComboBox()
        self.cmb_filter.addItems(["All Frames", "Level 2+ (Usable)", "Level 3+ (Animation Quality)", "Level 4 (Speaker Paired)"])
        self.cmb_filter.currentIndexChanged.connect(self._on_filter_changed)
        ctrls_layout.addWidget(self.cmb_filter)

        bottom_layout.addLayout(ctrls_layout)
        main_layout.addWidget(bottom_box)

    def _update_canvas_flags(self):
        self.canvas.show_clean = self.chk_clean.isChecked()
        self.canvas.show_raw = self.chk_raw.isChecked()
        self.canvas.show_wireframe = self.chk_wire.isChecked()
        self.canvas.show_pose_axes = self.chk_pose.isChecked()
        self.canvas.update()

    def _on_browse_session(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Recorded Session Folder", "sessions")
        if dir_path:
            self.load_session(dir_path)

    def load_session(self, session_dir: str):
        try:
            self.reader = DatasetReader(session_dir)
            total = self.reader.total_samples
            if total == 0:
                QMessageBox.warning(self, "Empty Session", "No motion frames found in this session.")
                return

            self.slider.setRange(0, total - 1)
            self.current_index = 0
            self.lbl_session_name.setText(f"Session: {os.path.basename(session_dir)} ({total} frames)")
            self._display_sample(0)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load session: {str(e)}")

    def _display_sample(self, index: int):
        if not self.reader or index < 0 or index >= self.reader.total_samples:
            return

        sample = self.reader.get_sample_at(index)
        ts = sample["timestamp"]
        total_ts = self.reader.arrays["timestamps"][-1] if "timestamps" in self.reader.arrays else ts

        # Update labels
        mins, secs = divmod(ts, 60.0)
        t_mins, t_secs = divmod(total_ts, 60.0)
        self.lbl_time.setText(f"{int(mins):02d}:{secs:05.2f} / {int(t_mins):02d}:{t_secs:05.2f} (Frame {index + 1}/{self.reader.total_samples})")

        # Update Canvas
        pose_euler = tuple(sample["clean_pose_euler"])
        self.canvas.set_data(sample["raw_landmarks"], sample["clean_landmarks"], pose_euler)

        # Update Blendshapes
        self.bs_meter.set_blendshapes(sample["clean_blendshapes"])

        # Update Diagnostics
        role = sample["role"]
        prob = sample["speaker_probability"]
        self.lbl_role_badge.setText(f"Face ID {sample['face_id']} | Role: {role} | Speaker Prob: {prob:.2f}")

        lvl_names = {0: "Level 0: Detected", 1: "Level 1: Trackable", 2: "Level 2: Geometrically Usable", 3: "Level 3: Animation Quality", 4: "Level 4: Speaker Paired"}
        self.lbl_level_badge.setText(f"Eligibility: {lvl_names.get(sample['eligibility_level'], 'Level Unknown')}")

        q_score = sample["quality_score"]
        blur = sample["blur_score"]
        self.lbl_quality_stats.setText(f"Quality: {q_score:.2f} | Sharpness (Laplacian): {blur:.1f} | BBox: {sample['bbox']}")
        self.lbl_pose_stats.setText(f"Head Pose: Pitch={pose_euler[0]:.1f}°, Yaw={pose_euler[1]:.1f}°, Roll={pose_euler[2]:.1f}°")

    def _on_slider_moved(self, val: int):
        if val != self.current_index:
            self.current_index = val
            self._display_sample(val)

    def _toggle_playback(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.setText("⏸ Pause")
            self.playback_timer.start(33)  # ~30 FPS
        else:
            self.btn_play.setText("▶ Play")
            self.playback_timer.stop()

    def _on_playback_step(self):
        if not self.reader:
            return
        next_idx = self.current_index + 1
        if next_idx >= self.reader.total_samples:
            next_idx = 0
        self.current_index = next_idx
        self.slider.setValue(next_idx)

    def _step_prev(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.slider.setValue(self.current_index)

    def _step_next(self):
        if self.reader and self.current_index < self.reader.total_samples - 1:
            self.current_index += 1
            self.slider.setValue(self.current_index)

    def _on_filter_changed(self, idx: int):
        # Could jump to next matching sample or filter list
        pass

    def _on_export_dataset(self):
        if not self.reader:
            QMessageBox.information(self, "No Session", "Please load a session first.")
            return

        save_path, _ = QFileDialog.getSaveFileName(self, "Export JSONL Dataset", "dataset_export.jsonl", "JSON Lines (*.jsonl)")
        if save_path:
            cnt = self.reader.export_to_jsonl(save_path, min_eligibility=EligibilityLevel.LEVEL_3_ANIMATION_QUALITY)
            QMessageBox.information(self, "Export Complete", f"Successfully exported {cnt} animation-quality samples to:\n{save_path}")
