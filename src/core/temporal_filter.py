"""Adaptive One Euro Filter for landmarks, blendshapes, and head pose."""

import math
from typing import Dict, Tuple, Optional
import numpy as np

from src.schema import RawObservation, CleanObservation
from src.config import TemporalFilterConfig


class LowPassFilter:
    """Standard 1st-order low-pass filter with exponential smoothing."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.prev_val: Optional[np.ndarray] = None

    def filter(self, val: np.ndarray, alpha: Optional[float] = None) -> np.ndarray:
        if alpha is not None:
            self.alpha = alpha
        if self.prev_val is None:
            self.prev_val = np.array(val, dtype=float, copy=True)
            return self.prev_val
        filtered = self.alpha * val + (1.0 - self.alpha) * self.prev_val
        self.prev_val = filtered
        return filtered

    def reset(self):
        self.prev_val = None


class OneEuroFilterArray:
    """One Euro Filter for N-dimensional numpy arrays (e.g. landmarks or blendshapes).
    Dynamically adapts cutoff frequency based on movement velocity.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.prev_time: Optional[float] = None

    @staticmethod
    def _alpha(rate: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / rate
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: np.ndarray, timestamp: float) -> np.ndarray:
        if self.prev_time is None:
            self.prev_time = timestamp
            return self.x_filter.filter(x, alpha=1.0)

        dt = max(1e-4, timestamp - self.prev_time)
        rate = 1.0 / dt
        self.prev_time = timestamp

        # 1. Estimate derivative (speed of change)
        prev_x = self.x_filter.prev_val if self.x_filter.prev_val is not None else x
        dx = (x - prev_x) * rate
        edx = self.dx_filter.filter(dx, self._alpha(rate, self.d_cutoff))

        # 2. Compute dynamic cutoff frequency
        cutoff = self.min_cutoff + self.beta * np.abs(edx)
        alpha = self._alpha(rate, cutoff)

        return self.x_filter.filter(x, alpha)

    def reset(self):
        self.prev_time = None
        self.x_filter.reset()
        self.dx_filter.reset()


class FaceTemporalFilter:
    """Maintains One Euro filter instances per tracked face identity."""

    def __init__(self, config: TemporalFilterConfig):
        self.config = config
        # face_id -> {landmarks_filter, blendshapes_filter, pose_filter}
        self.face_filters: Dict[int, Dict[str, OneEuroFilterArray]] = {}

    def reset(self):
        self.face_filters.clear()

    def filter_observation(
        self,
        face_id: int,
        raw: RawObservation,
        timestamp: float
    ) -> CleanObservation:
        """Filter raw observation using One Euro Filter without modifying the raw data."""
        if face_id not in self.face_filters:
            self.face_filters[face_id] = {
                "landmarks": OneEuroFilterArray(self.config.min_cutoff, self.config.beta, self.config.d_cutoff),
                "blendshapes": OneEuroFilterArray(self.config.min_cutoff, self.config.beta, self.config.d_cutoff),
                "pose": OneEuroFilterArray(self.config.min_cutoff, self.config.beta, self.config.d_cutoff),
            }

        filters = self.face_filters[face_id]

        # 1. Filter 3D Landmarks (478, 3)
        clean_landmarks = filters["landmarks"].filter(raw.landmarks, timestamp)

        # 2. Filter Blendshapes (dict converted to array and back)
        bs_keys = sorted(raw.blendshapes.keys())
        bs_values = np.array([raw.blendshapes[k] for k in bs_keys], dtype=float)
        clean_bs_values = filters["blendshapes"].filter(bs_values, timestamp)
        # Clip blendshapes to valid physical range [0.0, 1.0]
        clean_bs_values = np.clip(clean_bs_values, 0.0, 1.0)
        clean_blendshapes = {k: round(float(v), 5) for k, v in zip(bs_keys, clean_bs_values)}

        # 3. Filter Head Pose Euler (Pitch, Yaw, Roll) & Translation
        pose_arr = np.array([
            raw.head_pose_euler[0], raw.head_pose_euler[1], raw.head_pose_euler[2],
            raw.head_translation[0], raw.head_translation[1], raw.head_translation[2]
        ], dtype=float)
        clean_pose_arr = filters["pose"].filter(pose_arr, timestamp)

        clean_euler = (round(float(clean_pose_arr[0]), 2), round(float(clean_pose_arr[1]), 2), round(float(clean_pose_arr[2]), 2))
        clean_trans = (round(float(clean_pose_arr[3]), 3), round(float(clean_pose_arr[4]), 3), round(float(clean_pose_arr[5]), 3))

        return CleanObservation(
            landmarks=clean_landmarks,
            blendshapes=clean_blendshapes,
            head_pose_euler=clean_euler,
            head_translation=clean_trans,
            is_observed=True,
            is_reconstructed=False
        )
