"""Transparent interactive screen ROI selector overlay using PySide6."""

from typing import Optional, Tuple
from PySide6.QtCore import Qt, QRect, QPoint, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QGuiApplication
from PySide6.QtWidgets import QWidget


class ROISelectorOverlay(QWidget):
    """Full-screen translucent overlay allowing click-and-drag selection of a screen region."""

    roi_selected = Signal(int, int, int, int)  # (x, y, w, h) in global screen coordinates

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setCursor(Qt.CursorShape.CrossCursor)

        self.start_pos: Optional[QPoint] = None
        self.current_pos: Optional[QPoint] = None
        self.is_selecting = False

    def start_selection(self):
        """Cover all virtual screens and show overlay."""
        # Calculate bounding rect of all available screens
        virtual_rect = QRect()
        for screen in QGuiApplication.screens():
            virtual_rect = virtual_rect.united(screen.geometry())

        self.setGeometry(virtual_rect)
        self.start_pos = None
        self.current_pos = None
        self.is_selecting = False
        self.show()
        self.activateWindow()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.start_pos = event.globalPosition().toPoint()
            self.current_pos = self.start_pos
            self.is_selecting = True
            self.update()

    def mouseMoveEvent(self, event):
        if self.is_selecting:
            self.current_pos = event.globalPosition().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.is_selecting:
            self.is_selecting = False
            if self.start_pos and self.current_pos:
                rect = QRect(self.start_pos, self.current_pos).normalized()
                # Ensure minimum selection size
                if rect.width() >= 32 and rect.height() >= 32:
                    self.roi_selected.emit(rect.x(), rect.y(), rect.width(), rect.height())
            self.close()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 1. Dark translucent background mask
        painter.fillRect(self.rect(), QColor(0, 0, 0, 110))

        if self.start_pos and self.current_pos:
            # Map global positions to local widget coordinates
            p1 = self.mapFromGlobal(self.start_pos)
            p2 = self.mapFromGlobal(self.current_pos)
            selection_rect = QRect(p1, p2).normalized()

            # 2. Clear out selection rectangle
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.fillRect(selection_rect, Qt.GlobalColor.transparent)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

            # 3. Bright neon border around ROI
            pen = QPen(QColor(0, 220, 255), 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawRect(selection_rect)

            # 4. Dimension label tooltip
            label_text = f"ROI: {selection_rect.width()} × {selection_rect.height()} px  ({selection_rect.x()}, {selection_rect.y()})"
            painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(15, 23, 42, 220)))

            metrics = painter.fontMetrics()
            text_w = metrics.horizontalAdvance(label_text) + 20
            text_h = metrics.height() + 10
            label_rect = QRect(selection_rect.x(), max(10, selection_rect.y() - text_h - 6), text_w, text_h)
            painter.drawRoundedRect(label_rect, 4, 4)

            painter.setPen(QColor(255, 255, 255))
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, label_text)
        else:
            # Instructions hint in center
            hint = "Click and drag to select screen ROI for capture. Press ESC to cancel."
            painter.setFont(QFont("Segoe UI", 13, QFont.Weight.DemiBold))
            painter.setPen(QColor(255, 255, 255, 220))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, hint)
