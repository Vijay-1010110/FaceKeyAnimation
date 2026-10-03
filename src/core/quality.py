"""Multi-metric face quality evaluation with threshold hysteresis."""

import cv2
import math
from typing import Dict, Tuple, Optional, List
import numpy as np

from src.schema import QualityMetrics, EligibilityLevel
from src.config import FaceThresholds, HysteresisConfig


class QualityEvaluator:
    """Evaluates geometric, optical, and temporal reliability of facial observations."""

    def __init__(self, thresholds: FaceThresholds, hysteresis: HysteresisConfig):
        self.thresh = thresholds
        self.hysteresis = hysteresis
        # Per-face temporal tracking state: face_id -> {prev_landmarks, prev_time, state, state_time}
        self.face_histories: Dict[int, Dict] = {}

    def reset(self):
        self.face_histories.clear()

    @staticmethod
    def compute_blur_laplacian(image_rgb: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """Compute Laplacian variance over the face bounding box crop."""
        x, y, w, h = bbox
        img_h, img_w = image_rgb.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)

        if x2 - x1 < 10 or y2 - y1 < 10:
            return 0.0

        face_crop = image_rgb[y1:y2, x1:x2]
        gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = float(laplacian.var())
        return variance

    def evaluate(
        self,
        face_id: int,
        image_rgb: np.ndarray,
        bbox: Tuple[int, int, int, int],
        landmarks: np.ndarray,
        head_pose_euler: Tuple[float, float, float],
        timestamp: float,
        detection_confidence: float = 1.0,
        tracking_confidence: float = 1.0
    ) -> QualityMetrics:
        """Calculate comprehensive quality metrics and eligibility for a face observation."""
        img_h, img_w = image_rgb.shape[:2]
        roi_area = max(1, img_w * img_h)
        x, y, w, h = bbox
        area_ratio = (w * h) / roi_area

        # 1. Blur / Sharpness Metric
        blur_score = self.compute_blur_laplacian(image_rgb, bbox)
        is_sharp = blur_score >= self.thresh.min_blur_laplacian
        # Normalize sharpness score to [0, 1]
        sharp_norm = min(1.0, blur_score / 150.0)

        # 2. Geometric Size Metric
        size_valid = (w >= self.thresh.min_width_px) and (h >= self.thresh.min_height_px) and (area_ratio >= self.thresh.min_area_ratio)
        size_norm = min(1.0, area_ratio / 0.15)

        # 3. Head Pose Orientation Quality
        pitch, yaw, roll = head_pose_euler
        yaw_penalty = max(0.0, 1.0 - (abs(yaw) / max(1.0, self.thresh.max_abs_yaw_deg)))
        pitch_penalty = max(0.0, 1.0 - (abs(pitch) / max(1.0, self.thresh.max_abs_pitch_deg)))
        roll_penalty = max(0.0, 1.0 - (abs(roll) / max(1.0, self.thresh.max_abs_roll_deg)))
        pose_quality = (yaw_penalty * 0.5) + (pitch_penalty * 0.3) + (roll_penalty * 0.2)
        pose_valid = (abs(yaw) <= self.thresh.max_abs_yaw_deg) and (abs(pitch) <= self.thresh.max_abs_pitch_deg)

        # 4. Temporal Jitter / Displacement
        history = self.face_histories.setdefault(face_id, {
            "prev_landmarks": None,
            "prev_time": None,
            "state_is_good": False,
            "state_change_time": timestamp
        })

        temporal_jitter = 0.0
        jitter_valid = True
        if history["prev_landmarks"] is not None and history["prev_time"] is not None:
            dt = max(1e-4, timestamp - history["prev_time"])
            # Use nose tip and eye landmarks (indices 1, 33, 263) for displacement
            indices = [1, 33, 263, 61, 291]
            diff = landmarks[indices, :2] - history["prev_landmarks"][indices, :2]
            dist = np.mean(np.linalg.norm(diff, axis=1))
            temporal_jitter = float(dist / dt)  # displacement per second
            if temporal_jitter > self.thresh.max_jitter_velocity:
                jitter_valid = False

        history["prev_landmarks"] = landmarks.copy()
        history["prev_time"] = timestamp

        # 5. Composite Score Calculation
        confidence_norm = (detection_confidence + tracking_confidence) / 2.0
        jitter_norm = max(0.0, 1.0 - min(1.0, temporal_jitter / 1.5))

        composite_score = (
            confidence_norm * 0.25 +
            size_norm * 0.20 +
            sharp_norm * 0.20 +
            pose_quality * 0.20 +
            jitter_norm * 0.15
        )

        # 6. Rejection Reason Diagnosis
        rejection_reason = None
        if not size_valid:
            rejection_reason = f"face_too_small ({w}x{h}px, area {area_ratio:.3f})"
        elif not is_sharp:
            rejection_reason = f"motion_blur (laplacian {blur_score:.1f} < {self.thresh.min_blur_laplacian})"
        elif not pose_valid:
            rejection_reason = f"extreme_pose (yaw={yaw:.1f}, pitch={pitch:.1f})"
        elif not jitter_valid:
            rejection_reason = f"tracking_jitter (velocity {temporal_jitter:.2f})"
        elif confidence_norm < self.thresh.min_detection_confidence:
            rejection_reason = f"low_confidence ({confidence_norm:.2f})"

        # 7. Threshold Hysteresis
        current_state = history["state_is_good"]
        if not current_state:
            # Entering high quality requires passing enter threshold and all validations
            if composite_score >= self.hysteresis.quality_enter_threshold and rejection_reason is None:
                history["state_is_good"] = True
                history["state_change_time"] = timestamp
        else:
            # Exiting high quality occurs if score drops below exit threshold or critical failure
            if composite_score < self.hysteresis.quality_exit_threshold or rejection_reason is not None:
                if (timestamp - history["state_change_time"]) >= self.hysteresis.quality_min_duration_sec:
                    history["state_is_good"] = False
                    history["state_change_time"] = timestamp

        is_valid = history["state_is_good"]

        return QualityMetrics(
            composite_score=round(float(composite_score), 4),
            blur_score=round(float(blur_score), 2),
            is_sharp=is_sharp,
            face_width_px=w,
            face_height_px=h,
            area_ratio=round(float(area_ratio), 4),
            pose_quality=round(float(pose_quality), 4),
            temporal_jitter=round(float(temporal_jitter), 4),
            is_valid=is_valid,
            rejection_reason=rejection_reason
        )
