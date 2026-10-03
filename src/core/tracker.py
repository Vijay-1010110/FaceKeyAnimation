"""Spatial-temporal multi-face identity tracker with persistent IDs."""

import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np


@dataclass
class FaceTrack:
    """Internal state for a tracked face identity."""
    face_id: int
    bbox: Tuple[int, int, int, int]      # (x, y, w, h)
    centroid: Tuple[float, float]        # (cx, cy) normalized [0, 1]
    last_timestamp: float
    hit_count: int = 1
    missed_frames: int = 0
    is_primary: bool = False
    area_ratio: float = 0.0


class MultiFaceTracker:
    """Assigns and maintains persistent face_id identities across frames.
    Uses centroid distance and bounding-box overlap matching to prevent ID switching.
    """

    def __init__(self, max_missed_frames: int = 15, max_match_distance: float = 0.30):
        self.max_missed_frames = max_missed_frames
        self.max_match_distance = max_match_distance
        self.next_face_id: int = 0
        self.tracks: Dict[int, FaceTrack] = {}

    def reset(self):
        """Reset all tracking states."""
        self.next_face_id = 0
        self.tracks.clear()

    @staticmethod
    def bbox_from_landmarks(landmarks_norm: np.ndarray, image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Convert normalized (478, 3) landmarks into pixel bounding box (x, y, w, h)."""
        h, w = image_shape[:2]
        xs = landmarks_norm[:, 0] * w
        ys = landmarks_norm[:, 1] * h
        x_min = max(0, int(np.min(xs)))
        y_min = max(0, int(np.min(ys)))
        x_max = min(w, int(np.max(xs)))
        y_max = min(h, int(np.max(ys)))
        return (x_min, y_min, max(1, x_max - x_min), max(1, y_max - y_min))

    def update(
        self,
        detected_landmarks_list: List[np.ndarray],
        timestamp: float,
        image_shape: Tuple[int, int]
    ) -> List[Tuple[int, bool, Tuple[int, int, int, int]]]:
        """Update tracker with detections from current frame.
        Returns list of (face_id, is_primary, bbox) matching the order of input landmarks.
        """
        img_h, img_w = image_shape[:2]
        roi_area = max(1, img_w * img_h)

        detections = []
        for lms in detected_landmarks_list:
            bbox = self.bbox_from_landmarks(lms, image_shape)
            cx = (bbox[0] + bbox[2] / 2.0) / img_w
            cy = (bbox[1] + bbox[3] / 2.0) / img_h
            area = (bbox[2] * bbox[3]) / roi_area
            detections.append({"bbox": bbox, "centroid": (cx, cy), "area_ratio": area})

        # Match detections to existing active tracks
        active_track_ids = [tid for tid, trk in self.tracks.items() if trk.missed_frames <= self.max_missed_frames]
        assigned_track_ids = [None] * len(detections)
        matched_tracks = set()

        if active_track_ids and detections:
            # Build cost matrix based on Euclidean centroid distance
            cost_matrix = np.zeros((len(detections), len(active_track_ids)), dtype=float)
            for i, det in enumerate(detections):
                dcx, dcy = det["centroid"]
                for j, tid in enumerate(active_track_ids):
                    tcx, tcy = self.tracks[tid].centroid
                    dist = math.sqrt((dcx - tcx) ** 2 + (dcy - tcy) ** 2)
                    cost_matrix[i, j] = dist

            # Greedy matching for lowest distance below threshold
            while True:
                min_val = np.min(cost_matrix)
                if min_val > self.max_match_distance:
                    break
                i, j = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
                cost_matrix[i, :] = 1e9
                cost_matrix[:, j] = 1e9

                matched_tid = active_track_ids[j]
                assigned_track_ids[i] = matched_tid
                matched_tracks.add(matched_tid)

        # Update matched tracks & create new tracks for unassigned detections
        result = []
        for i, det in enumerate(detections):
            tid = assigned_track_ids[i]
            if tid is not None:
                track = self.tracks[tid]
                track.bbox = det["bbox"]
                track.centroid = det["centroid"]
                track.area_ratio = det["area_ratio"]
                track.last_timestamp = timestamp
                track.hit_count += 1
                track.missed_frames = 0
            else:
                tid = self.next_face_id
                self.next_face_id += 1
                track = FaceTrack(
                    face_id=tid,
                    bbox=det["bbox"],
                    centroid=det["centroid"],
                    last_timestamp=timestamp,
                    area_ratio=det["area_ratio"]
                )
                self.tracks[tid] = track

            result.append(tid)

        # Increment missed frames on unmatched active tracks
        for tid in active_track_ids:
            if tid not in matched_tracks:
                self.tracks[tid].missed_frames += 1

        # Determine Primary Face
        # Criteria: persistence (hit_count), size (area_ratio), centrality
        primary_tid = None
        best_score = -1.0
        for tid in result:
            trk = self.tracks[tid]
            cx, cy = trk.centroid
            center_dist = math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)
            centrality = max(0.0, 1.0 - (center_dist / 0.707))
            score = (trk.area_ratio * 0.5) + (centrality * 0.3) + (min(trk.hit_count, 30) / 30.0 * 0.2)
            if score > best_score:
                best_score = score
                primary_tid = tid

        # Build output list
        output = []
        for i, tid in enumerate(result):
            is_primary = (tid == primary_tid)
            self.tracks[tid].is_primary = is_primary
            output.append((tid, is_primary, detections[i]["bbox"]))

        return output
