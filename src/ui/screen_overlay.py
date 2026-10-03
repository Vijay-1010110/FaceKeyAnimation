"""Transparent click-through on-screen HUD overlay for live motion capture tracking."""

import math
from typing import List, Optional, Tuple, Dict
import numpy as np

from PySide6.QtCore import Qt, QRect, QPointF, QTimer
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPolygonF
from PySide6.QtWidgets import QWidget

from src.schema import TrackedFaceFrame, FaceRole, EligibilityLevel

# Standard facial feature loops
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 362]
LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
NOSE_BRIDGE = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2]


class ScreenMotionOverlay(QWidget):
    """Transparent click-through on-screen HUD floating directly over the captured screen ROI.
    Renders facial landmark dots, head orientation axes, and speaker attribution tags
    directly on top of the playing YouTube video in real time.
    """

    def __init__(self):
        super().__init__()
        # Set click-through, frameless, and top-most window flags
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool |
            Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        self.roi: Tuple[int, int, int, int] = (100, 100, 640, 480)
        self.tracked_faces: List[TrackedFaceFrame] = []
        self.is_speech_active: bool = False
        self.audio_energy: float = 0.0
        self.fps: float = 0.0
        self.pulse_phase: float = 0.0

        # Animation timer for pulsing speaker aura
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._on_anim_tick)
        self.anim_timer.start(50)  # 20 FPS pulse animation

    def _on_anim_tick(self):
        self.pulse_phase += 0.15
        if self.pulse_phase > 2 * math.pi:
            self.pulse_phase = 0.0
        if self.isVisible():
            self.update()

    def update_roi(self, x: int, y: int, w: int, h: int):
        """Reposition the overlay window to match the exact screen ROI coordinates."""
        self.roi = (x, y, w, h)
        self.setGeometry(x, y, w, h)
        self.update()

    def update_tracking_data(
        self,
        faces: List[TrackedFaceFrame],
        is_speech: bool = False,
        energy: float = 0.0,
        fps: float = 0.0
    ):
        """Update live motion tracking data for on-screen overlay rendering."""
        self.tracked_faces = faces
        self.is_speech_active = is_speech
        self.audio_energy = energy
        self.fps = fps
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()

        # 1. Subtle futuristic border around the monitored ROI
        border_pen = QPen(QColor(56, 189, 248, 140), 1.5, Qt.PenStyle.DashLine)
        painter.setPen(border_pen)
        painter.drawRect(1, 1, w - 2, h - 2)

        # Corner brackets
        corner_pen = QPen(QColor(6, 182, 212), 3)
        painter.setPen(corner_pen)
        cl = 16
        painter.drawLine(0, 0, cl, 0); painter.drawLine(0, 0, 0, cl)
        painter.drawLine(w, 0, w - cl, 0); painter.drawLine(w, 0, w, cl)
        painter.drawLine(0, h, cl, h); painter.drawLine(0, h, 0, h - cl)
        painter.drawLine(w, h, w - cl, h); painter.drawLine(w, h, w, h - cl)

        # 2. Render each tracked face
        for face in self.tracked_faces:
            self._render_face(painter, face, w, h)

        # 3. Top HUD Status Bar
        self._render_top_hud(painter, w)

    def _render_face(self, painter: QPainter, face: TrackedFaceFrame, roi_w: int, roi_h: int):
        bx, by, bw, bh = face.bbox
        is_speaker = (face.role == FaceRole.SPEAKER)
        is_listener = (face.role == FaceRole.LISTENER)

        # Determine theme color
        if is_speaker:
            # Pulsating neon green
            alpha = int(190 + 65 * math.sin(self.pulse_phase))
            theme_color = QColor(34, 197, 94, alpha)
            glow_color = QColor(34, 197, 94, 60)
            tag_text = f"🎙️ FACE {face.face_id}: SPEAKER ({int(face.speaker_probability * 100)}%)"
        elif is_listener:
            if getattr(face, "is_narration", False):
                # Listening to off-screen narration
                theme_color = QColor(245, 158, 11, 230)
                glow_color = QColor(245, 158, 11, 35)
                tag_text = f"👁️ FACE {face.face_id}: LISTENER [NARRATION]"
            else:
                # Cool neon cyan/blue
                theme_color = QColor(56, 189, 248, 220)
                glow_color = QColor(56, 189, 248, 30)
                tag_text = f"👁️ FACE {face.face_id}: LISTENER"
        else:
            # Amber / gray for unknown
            theme_color = QColor(148, 163, 184, 180)
            glow_color = QColor(148, 163, 184, 20)
            tag_text = f"FACE {face.face_id}: UNKNOWN"

        # 1. Bounding Box & Corner Highlights
        painter.setPen(QPen(theme_color, 2 if is_speaker else 1.5))
        painter.setBrush(QBrush(glow_color))
        painter.drawRoundedRect(bx, by, bw, bh, 6, 6)

        # 2. Floating Speaker / Identity Tag
        tag_font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        painter.setFont(tag_font)
        metrics = painter.fontMetrics()
        tag_w = metrics.horizontalAdvance(tag_text) + 16
        tag_h = metrics.height() + 8
        tag_y = max(8, by - tag_h - 4)

        # Tag background
        painter.setPen(Qt.PenStyle.NoPen)
        tag_bg = QColor(15, 23, 42, 235)
        painter.setBrush(QBrush(tag_bg))
        painter.drawRoundedRect(bx, tag_y, tag_w, tag_h, 4, 4)

        # Tag border & text
        painter.setPen(QPen(theme_color, 1))
        painter.drawRoundedRect(bx, tag_y, tag_w, tag_h, 4, 4)
        painter.setPen(theme_color)
        painter.drawText(QRect(bx + 8, tag_y, tag_w - 16, tag_h), Qt.AlignmentFlag.AlignVCenter, tag_text)

        # 3. Mini Motion Metrics Chip (Jaw, Smile, Quality)
        bs = face.clean.blendshapes
        jaw_val = bs.get("jawOpen", 0.0)
        smile_val = (bs.get("mouthSmileLeft", 0.0) + bs.get("mouthSmileRight", 0.0)) / 2.0
        q_val = face.quality.composite_score

        metrics_text = f"Jaw:{jaw_val:.2f} | Smile:{smile_val:.2f} | Q:{q_val:.2f}"
        chip_y = by + bh + 4
        if chip_y + 20 < roi_h:
            painter.setFont(QFont("Consolas", 8, QFont.Weight.DemiBold))
            c_metrics = painter.fontMetrics()
            chip_w = c_metrics.horizontalAdvance(metrics_text) + 12
            chip_h = c_metrics.height() + 4

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(15, 23, 42, 210)))
            painter.drawRoundedRect(bx, chip_y, chip_w, chip_h, 3, 3)
            painter.setPen(QColor(226, 232, 240))
            painter.drawText(QRect(bx + 6, chip_y, chip_w - 12, chip_h), Qt.AlignmentFlag.AlignVCenter, metrics_text)

        # 4. Landmark Mesh Dots
        lms = face.clean.landmarks
        if len(lms) >= 468:
            # Feature lines
            wire_pen = QPen(QColor(theme_color.red(), theme_color.green(), theme_color.blue(), 100), 1)
            painter.setPen(wire_pen)
            for loop in [LIPS_OUTER, LIPS_INNER, LEFT_EYE, RIGHT_EYE, LEFT_EYEBROW, RIGHT_EYEBROW, NOSE_BRIDGE]:
                poly = QPolygonF()
                for idx in loop:
                    if idx < len(lms):
                        poly.append(QPointF(lms[idx][0] * roi_w, lms[idx][1] * roi_h))
                painter.drawPolyline(poly)

            # Iris dots (Gold)
            painter.setPen(QPen(QColor(234, 179, 8), 4))
            for iris_idx in [468, 473]:
                if iris_idx < len(lms):
                    painter.drawPoint(QPointF(lms[iris_idx][0] * roi_w, lms[iris_idx][1] * roi_h))

            # Landmark dots
            dot_color = QColor(255, 255, 255, 200) if is_speaker else QColor(56, 189, 248, 160)
            painter.setPen(QPen(dot_color, 2.5))
            # Subsample dots for crystal clear overlay
            for pt in lms[::4]:
                painter.drawPoint(QPointF(pt[0] * roi_w, pt[1] * roi_h))

            # 5. 3D Head Orientation Axes
            nose_pt = QPointF(lms[1][0] * roi_w, lms[1][1] * roi_h)
            pitch, yaw, roll = face.clean.head_pose_euler
            rad = math.pi / 180.0
            p_r, y_r, r_r = pitch * rad, yaw * rad, roll * rad
            axis_len = 35.0

            # Red: Pitch/X
            dx_x = axis_len * math.cos(y_r) * math.cos(r_r)
            dy_x = axis_len * (math.sin(p_r) * math.sin(y_r) * math.cos(r_r) - math.cos(p_r) * math.sin(r_r))
            painter.setPen(QPen(QColor(239, 68, 68), 2.5))
            painter.drawLine(nose_pt, QPointF(nose_pt.x() + dx_x, nose_pt.y() + dy_x))

            # Green: Yaw/Y
            dx_y = -axis_len * math.cos(y_r) * math.sin(r_r)
            dy_y = -axis_len * (math.sin(p_r) * math.sin(y_r) * math.sin(r_r) + math.cos(p_r) * math.cos(r_r))
            painter.setPen(QPen(QColor(34, 197, 94), 2.5))
            painter.drawLine(nose_pt, QPointF(nose_pt.x() + dx_y, nose_pt.y() + dy_y))

            # Blue: Forward gaze/Z
            dx_z = -axis_len * math.sin(y_r)
            dy_z = axis_len * math.sin(p_r) * math.cos(y_r)
            painter.setPen(QPen(QColor(59, 130, 246), 2.5))
            painter.drawLine(nose_pt, QPointF(nose_pt.x() + dx_z, nose_pt.y() + dy_z))

    def _render_top_hud(self, painter: QPainter, roi_w: int):
        hud_h = 24
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(15, 23, 42, 220)))
        painter.drawRoundedRect(6, 6, roi_w - 12, hud_h, 4, 4)

        painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))

        # 1. Tracking state & FPS
        fps_text = f"LIVE TRACKER: {self.fps:.1f} FPS"
        painter.setPen(QColor(56, 189, 248))
        painter.drawText(14, 22, fps_text)

        # 2. Audio Speech Activity LED
        led_x = 180
        led_color = QColor(34, 197, 94) if self.is_speech_active else QColor(100, 116, 139)
        painter.setBrush(QBrush(led_color))
        painter.drawEllipse(led_x, 13, 10, 10)

        speech_text = "SPEECH ACTIVE" if self.is_speech_active else "AUDIO IDLE"
        painter.setPen(led_color)
        painter.drawText(led_x + 16, 22, speech_text)

        # 3. Active Speaker / Narration Status
        spk_faces = [f.face_id for f in self.tracked_faces if f.role == FaceRole.SPEAKER]
        is_narr = any(getattr(f, "is_narration", False) for f in self.tracked_faces)
        if spk_faces:
            spk_text = f"🎙️ SPEAKER: Face {spk_faces[0]}"
            painter.setPen(QColor(34, 197, 94))
        elif is_narr or (self.is_speech_active and not spk_faces):
            spk_text = "📢 NARRATION / VOICEOVER (Off-Screen)"
            painter.setPen(QColor(245, 158, 11))
        elif self.is_speech_active:
            spk_text = "🎙️ SPEECH ACTIVE"
            painter.setPen(QColor(34, 197, 94))
        else:
            spk_text = "SILENCE / LISTENING"
            painter.setPen(QColor(148, 163, 184))
        painter.drawText(310, 22, spk_text)

        # Hint on right
        painter.setPen(QColor(148, 163, 184))
        painter.drawText(roi_w - 145, 22, "Transparent Click-Through")
