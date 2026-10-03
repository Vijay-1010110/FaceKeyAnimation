"""Dataset reader for loading, filtering, and exporting acquired facial motion sessions."""

import json
import os
from typing import Dict, Any, Optional, List, Generator
import numpy as np

from src.schema import EligibilityLevel, FaceRole


class DatasetReader:
    """Reads structured session data from disk with multi-level eligibility filtering."""

    def __init__(self, session_dir: str):
        if not os.path.exists(session_dir):
            raise FileNotFoundError(f"Session directory does not exist: {session_dir}")
        self.session_dir = session_dir
        self.metadata: Dict[str, Any] = {}
        self.arrays: Dict[str, np.ndarray] = {}
        self.audio_features: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        meta_path = os.path.join(self.session_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        npz_path = os.path.join(self.session_dir, "face_motion.npz")
        if os.path.exists(npz_path):
            data = np.load(npz_path, allow_pickle=True)
            for k in data.files:
                self.arrays[k] = data[k]

        audio_path = os.path.join(self.session_dir, "audio_features.json")
        if os.path.exists(audio_path):
            with open(audio_path, "r", encoding="utf-8") as f:
                self.audio_features = json.load(f)

    @property
    def total_samples(self) -> int:
        return len(self.arrays.get("timestamps", []))

    def filter_indices(
        self,
        min_eligibility: Optional[EligibilityLevel] = None,
        face_id: Optional[int] = None,
        role: Optional[FaceRole] = None,
        primary_only: bool = False
    ) -> np.ndarray:
        """Get indices of samples meeting specific filtering criteria."""
        mask = np.ones(self.total_samples, dtype=bool)

        if min_eligibility is not None and "eligibility_levels" in self.arrays:
            mask &= (self.arrays["eligibility_levels"] >= int(min_eligibility))

        if face_id is not None and "face_ids" in self.arrays:
            mask &= (self.arrays["face_ids"] == face_id)

        if primary_only and "is_primary" in self.arrays:
            mask &= self.arrays["is_primary"]

        if role is not None and "roles" in self.arrays:
            target_role = role.value if hasattr(role, "value") else str(role)
            mask &= (self.arrays["roles"] == target_role)

        return np.where(mask)[0]

    def get_sample_at(self, index: int) -> Dict[str, Any]:
        """Retrieve a complete structured dictionary for a specific sample index."""
        bs_names = list(self.arrays.get("blendshape_names", []))
        raw_bs_row = self.arrays["raw_blendshapes"][index] if "raw_blendshapes" in self.arrays else []
        clean_bs_row = self.arrays["clean_blendshapes"][index] if "clean_blendshapes" in self.arrays else []

        raw_bs_dict = {str(name): float(val) for name, val in zip(bs_names, raw_bs_row)}
        clean_bs_dict = {str(name): float(val) for name, val in zip(bs_names, clean_bs_row)}

        return {
            "timestamp": float(self.arrays["timestamps"][index]),
            "face_id": int(self.arrays["face_ids"][index]),
            "is_primary": bool(self.arrays["is_primary"][index]),
            "bbox": [int(x) for x in self.arrays["bboxes"][index]],
            "raw_landmarks": self.arrays["raw_landmarks"][index],
            "clean_landmarks": self.arrays["clean_landmarks"][index],
            "raw_blendshapes": raw_bs_dict,
            "clean_blendshapes": clean_bs_dict,
            "raw_pose_euler": [float(x) for x in self.arrays["raw_pose_euler"][index]],
            "clean_pose_euler": [float(x) for x in self.arrays["clean_pose_euler"][index]],
            "quality_score": float(self.arrays["quality_scores"][index]),
            "blur_score": float(self.arrays["blur_scores"][index]),
            "is_valid": bool(self.arrays["is_valid"][index]),
            "eligibility_level": int(self.arrays["eligibility_levels"][index]),
            "role": str(self.arrays["roles"][index]),
            "speaker_probability": float(self.arrays["speaker_probabilities"][index])
        }

    def export_to_jsonl(self, output_path: str, min_eligibility: Optional[EligibilityLevel] = None) -> int:
        """Export filtered samples to a newline-delimited JSON file for ML model training."""
        indices = self.filter_indices(min_eligibility=min_eligibility)
        count = 0
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for idx in indices:
                sample = self.get_sample_at(idx)
                # Convert numpy landmarks to serializable list
                sample["raw_landmarks"] = sample["raw_landmarks"].tolist()
                sample["clean_landmarks"] = sample["clean_landmarks"].tolist()
                f.write(json.dumps(sample) + "\n")
                count += 1
        return count


class NormalizedTrainingDataset:
    """Reader for normalized multi-subject training datasets.
    Provides batch indexing, PyTorch/TensorFlow compatibility, and train/val splits.
    """

    def __init__(self, npz_path: str = "sessions/normalized_training_dataset.npz"):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Normalized dataset not found: {npz_path}")
        self.npz_path = npz_path
        self._data = np.load(npz_path, allow_pickle=True)
        self.split_mask = self._data["train_split_mask"]

    def __len__(self) -> int:
        return len(self._data["timestamps_ns"])

    def get_split(self, split: str = "train") -> Dict[str, np.ndarray]:
        """Returns arrays filtered by split ('train', 'val', or 'all')."""
        if split == "train":
            mask = self.split_mask
        elif split in ("val", "validation"):
            mask = ~self.split_mask
        else:
            mask = np.ones(len(self), dtype=bool)

        out = {}
        for k in self._data.files:
            arr = self._data[k]
            if len(arr) == len(self):
                out[k] = arr[mask]
            else:
                out[k] = arr
        return out

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get sample dictionary by index."""
        sample = {
            "canonical_landmarks": self._data["canonical_landmarks"][idx],
            "expression_deltas": self._data["expression_deltas"][idx],
            "symmetric_landmarks": self._data["symmetric_landmarks"][idx],
            "asymmetric_residuals": self._data["asymmetric_residuals"][idx],
            "dental_features": self._data["dental_features"][idx],
            "pose_deltas": self._data["pose_deltas"][idx],
            "timestamp_ns": int(self._data["timestamps_ns"][idx]),
            "session_id": str(self._data["session_ids"][idx]),
            "role": str(self._data["roles"][idx]),
            "audio_energy": float(self._data["audio_energy"][idx]),
            "audio_speech_prob": float(self._data["audio_speech_prob"][idx]),
            "is_train": bool(self.split_mask[idx])
        }
        if "blendshapes" in self._data:
            sample["blendshapes"] = self._data["blendshapes"][idx]
        return sample
