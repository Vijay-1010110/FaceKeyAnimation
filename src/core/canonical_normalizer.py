"""Canonical Facial Normalization Module for Deep Learning Training.
Decouples skull morphology, head pose, camera distance, and structural asymmetry from pure speech kinematics.
Projects all subjects onto an Invariant Universal Canonical Reference Face.
"""

import math
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


# ---------------------------------------------------------------------------
# Invariant Cranial Bone Anchors (Zero deformation during speech/jaw opening)
# ---------------------------------------------------------------------------
CRANIAL_BONE_ANCHORS = [
    168,  # Sellion / Nasion (nasal bridge between eyes)
    6,    # Upper nasal bone
    197,  # Mid nasal bone
    133,  # Left inner eye canthus (ethmoid/lacrimal bone anchor)
    362,  # Right inner eye canthus
    33,   # Left outer eye canthus (zygomaticofrontal suture)
    263,  # Right outer eye canthus
    10,   # Superior mid-forehead (frontal bone)
    151,  # Mid-glabella
    9,    # Lower glabella
    234,  # Left skull temple / tragus root (temporal bone)
    454   # Right skull temple / tragus root
]

# ---------------------------------------------------------------------------
# Anatomical Region Landmarks
# ---------------------------------------------------------------------------
SELLION_INDEX = 168
CHIN_INDEX = 152
NOSE_TIP_INDEX = 1
SUB_NASALE_INDEX = 2

LEFT_PUPIL_INDEX = 468
RIGHT_PUPIL_INDEX = 473

MOUTH_LEFT_CORNER = 61
MOUTH_RIGHT_CORNER = 291
UPPER_LIP_CENTER_OUTER = 0
LOWER_LIP_CENTER_OUTER = 17
UPPER_LIP_CENTER_INNER = 13
LOWER_LIP_CENTER_INNER = 14

LIP_LANDMARKS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
    95, 88, 178, 87, 14, 317, 402, 318, 324
]

NOSE_LANDMARKS = [
    168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 97, 326, 327
]

EYE_LANDMARKS = [
    33, 160, 158, 133, 153, 144, 468, 469, 470, 471, 472,
    362, 385, 387, 263, 373, 380, 473, 474, 475, 476, 477
]

JAW_LANDMARKS = [
    172, 136, 149, 150, 176, 152, 400, 378, 379, 365, 397
]

# ---------------------------------------------------------------------------
# Bilateral Symmetry Mapping Table (Left <-> Right anatomical pairs)
# ---------------------------------------------------------------------------
# Midline landmarks: invariant under reflection
MIDLINE_LANDMARKS = [
    10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2,
    0, 11, 12, 13, 14, 15, 16, 17, 18, 200, 199, 175, 152
]

# Key Bilateral Symmetry Pairs (Left Index <-> Right Index)
BILATERAL_PAIRS = [
    # Eyes & Canthi
    (33, 263), (133, 362), (160, 385), (158, 387), (153, 373), (144, 380),
    (159, 386), (145, 374), (246, 466), (7, 249),
    # Pupils & Irises
    (468, 473), (469, 474), (470, 475), (471, 476), (472, 477),
    # Eyebrows
    (70, 300), (63, 293), (105, 334), (66, 296), (107, 336),
    (55, 285), (65, 295), (52, 282), (53, 283), (46, 276),
    # Mouth & Vermilion Borders
    (61, 291), (185, 409), (40, 270), (39, 269), (37, 267),
    (146, 375), (91, 321), (181, 405), (84, 314),
    # Inner Lips
    (78, 308), (191, 415), (80, 310), (81, 311), (82, 312),
    (95, 324), (88, 318), (178, 402), (87, 317),
    # Nose & Alar Wings
    (98, 327), (97, 326), (129, 358), (203, 423),
    # Cheeks & Zygomatic
    (116, 345), (123, 352), (147, 376), (213, 433), (138, 367),
    (234, 454), (127, 356), (93, 323), (132, 361), (58, 288),
    # Jawline
    (172, 397), (136, 365), (149, 378), (150, 379), (176, 400),
    (58, 288), (21, 251), (54, 284), (103, 332), (67, 297), (109, 338)
]


def build_symmetry_permutation_index(num_landmarks: int = 478) -> np.ndarray:
    """Build a complete permutation array perm such that mirrored_point[i] corresponds to perm[i]."""
    perm = np.arange(num_landmarks, dtype=np.int32)
    for l_idx, r_idx in BILATERAL_PAIRS:
        if l_idx < num_landmarks and r_idx < num_landmarks:
            perm[l_idx] = r_idx
            perm[r_idx] = l_idx
    for mid in MIDLINE_LANDMARKS:
        if mid < num_landmarks:
            perm[mid] = mid
    return perm


# Precomputed symmetry mapping
SYMMETRY_PERMUTATION = build_symmetry_permutation_index(478)


# ---------------------------------------------------------------------------
# 3D Rigid Procrustes / Umeyama Alignment
# ---------------------------------------------------------------------------
def umeyama_rigid_alignment(
    source_pts: np.ndarray,
    target_pts: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Computes the closed-form Umeyama (1991) 3D similarity transformation.
    Finds optimal scale s, rotation R in SO(3), and translation t such that:
        || target_pts - (s * source_pts @ R.T + t) ||^2 is minimized.
    
    Args:
        source_pts: (K, 3) coordinates in source space
        target_pts: (K, 3) coordinates in target/canonical space
        
    Returns:
        s: float isotropic scale factor
        R: (3, 3) orthogonal rotation matrix with det(R) == +1.0
        t: (3,) translation vector
    """
    assert source_pts.shape == target_pts.shape, "Source and target point sets must have identical shapes"
    k = len(source_pts)
    
    mu_src = np.mean(source_pts, axis=0)
    mu_tgt = np.mean(target_pts, axis=0)
    
    src_c = source_pts - mu_src
    tgt_c = target_pts - mu_tgt
    
    var_src = np.mean(np.sum(src_c ** 2, axis=1))
    if var_src < 1e-12:
        return 1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
    
    # Covariance matrix H = tgt_c.T @ src_c / k
    H = (tgt_c.T @ src_c) / k
    
    # SVD
    U, D, Vt = np.linalg.svd(H)
    
    # Reflection check to enforce strictly det(R) == +1.0
    det_check = np.linalg.det(U) * np.linalg.det(Vt)
    d = 1.0 if det_check >= 0 else -1.0
    
    S = np.diag([1.0, 1.0, d])
    R = U @ S @ Vt
    
    s = float(np.sum(D * np.diag(S)) / var_src)
    t = mu_tgt - s * (R @ mu_src)
    
    return s, R.astype(np.float32), t.astype(np.float32)


def apply_similarity_transform(
    pts: np.ndarray,
    s: float,
    R: np.ndarray,
    t: np.ndarray
) -> np.ndarray:
    """Applies s * pts @ R.T + t to an array of points of shape (..., 3)."""
    return (s * (pts @ R.T)) + t


# ---------------------------------------------------------------------------
# Universal Canonical Reference Face Template
# ---------------------------------------------------------------------------
class UniversalCanonicalFace:
    """Universal Canonical Reference Face template centered at the nasal sellion (0,0,0).
    Coordinates are defined in metric-equivalent canonical space with:
    - X-axis: Left (-) to Right (+) horizontal vector
    - Y-axis: Superior (-) to Inferior (+) vertical vector
    - Z-axis: Posterior (-) to Anterior (+) forward depth vector
    - Inter-Pupillary Distance (IPD): Standardized to 1.0 canonical units
    """
    _instance: Optional[np.ndarray] = None

    @classmethod
    def get_template(cls) -> np.ndarray:
        if cls._instance is None:
            cls._instance = cls._generate_canonical_template()
        return cls._instance.copy()

    @classmethod
    def _generate_canonical_template(cls) -> np.ndarray:
        """Constructs an idealized anthropometric canonical face from MediaPipe topology."""
        template = np.zeros((478, 3), dtype=np.float32)
        
        # In a normalized canonical face with IPD = 1.0:
        # Left pupil at x = -0.5, Right pupil at x = +0.5, y = 0, z = 0
        template[LEFT_PUPIL_INDEX] = [-0.50, 0.00, 0.00]
        template[RIGHT_PUPIL_INDEX] = [+0.50, 0.00, 0.00]
        
        # Nasal sellion at origin
        template[SELLION_INDEX] = [0.00, 0.00, 0.00]
        
        # Eye canthi
        template[133] = [-0.25, 0.00, 0.02]  # Left inner canthus
        template[362] = [+0.25, 0.00, 0.02]  # Right inner canthus
        template[33]  = [-0.60, 0.00, -0.05] # Left outer canthus
        template[263] = [+0.60, 0.00, -0.05] # Right outer canthus
        
        # Nose bridge & tip
        template[6]   = [0.00, 0.15, 0.04]
        template[197] = [0.00, 0.30, 0.08]
        template[1]   = [0.00, 0.55, 0.22]  # Nose tip (Pronasale)
        template[2]   = [0.00, 0.70, 0.12]  # Subnasale
        template[98]  = [-0.20, 0.60, 0.08] # Left alar base
        template[327] = [+0.20, 0.60, 0.08] # Right alar base
        
        # Lips & Mouth
        template[MOUTH_LEFT_CORNER]  = [-0.38, 0.95, 0.06]
        template[MOUTH_RIGHT_CORNER] = [+0.38, 0.95, 0.06]
        template[UPPER_LIP_CENTER_OUTER] = [0.00, 0.88, 0.14]
        template[LOWER_LIP_CENTER_OUTER] = [0.00, 1.05, 0.12]
        template[UPPER_LIP_CENTER_INNER] = [0.00, 0.93, 0.10]
        template[LOWER_LIP_CENTER_INNER] = [0.00, 0.98, 0.09]
        
        # Chin
        template[CHIN_INDEX] = [0.00, 1.45, 0.04]
        
        # Cranium / Glabella / Forehead
        template[168] = [0.00, 0.00, 0.00]
        template[9]   = [0.00, -0.15, -0.01]
        template[151] = [0.00, -0.30, -0.03]
        template[10]  = [0.00, -0.65, -0.08]
        
        # Skull temples
        template[234] = [-0.95, -0.05, -0.45]
        template[454] = [+0.95, -0.05, -0.45]
        
        # Enforce exact bilateral symmetry across x=0
        for l_idx, r_idx in BILATERAL_PAIRS:
            if l_idx < 478 and r_idx < 478:
                if np.all(template[l_idx] == 0) and not np.all(template[r_idx] == 0):
                    template[l_idx] = template[r_idx] * np.array([-1.0, 1.0, 1.0], dtype=np.float32)
                elif np.all(template[r_idx] == 0) and not np.all(template[l_idx] == 0):
                    template[r_idx] = template[l_idx] * np.array([-1.0, 1.0, 1.0], dtype=np.float32)
                else:
                    # Symmetrize
                    sym_x = 0.5 * (abs(template[l_idx, 0]) + abs(template[r_idx, 0]))
                    sym_y = 0.5 * (template[l_idx, 1] + template[r_idx, 1])
                    sym_z = 0.5 * (template[l_idx, 2] + template[r_idx, 2])
                    template[l_idx] = [-sym_x, sym_y, sym_z]
                    template[r_idx] = [+sym_x, sym_y, sym_z]
        
        for mid in MIDLINE_LANDMARKS:
            template[mid, 0] = 0.0
            
        return template


# ---------------------------------------------------------------------------
# Subject Neutral Rest Face Estimator
# ---------------------------------------------------------------------------
class SubjectNeutralFaceEstimator:
    """Estimates subject baseline resting morphology from pause and low-activation frames."""

    @staticmethod
    def estimate_neutral_face(
        session_landmarks: np.ndarray,
        session_blendshapes: Optional[np.ndarray] = None,
        blendshape_names: Optional[List[str]] = None,
        session_roles: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Finds resting neutral frames and computes the median cranial-aligned face."""
        n_frames = len(session_landmarks)
        if n_frames == 0:
            return UniversalCanonicalFace.get_template()

        rest_indices = []

        # Find jawOpen and mouthSmile blendshape columns if available
        jaw_open_col = -1
        smile_left_col = -1
        smile_right_col = -1
        if session_blendshapes is not None and blendshape_names is not None:
            bs_map = {name: i for i, name in enumerate(blendshape_names)}
            jaw_open_col = bs_map.get("jawOpen", -1)
            smile_left_col = bs_map.get("mouthSmileLeft", -1)
            smile_right_col = bs_map.get("mouthSmileRight", -1)

        for i in range(n_frames):
            is_candidate = True
            
            # Check pause role
            if session_roles is not None and len(session_roles) > i:
                role = str(session_roles[i])
                if "PAUSE" in role or "LISTENER" in role:
                    rest_indices.append(i)
                    continue

            # Check low blendshape activation
            if session_blendshapes is not None and len(session_blendshapes) > i:
                bs = session_blendshapes[i]
                if jaw_open_col >= 0 and bs[jaw_open_col] > 0.08:
                    is_candidate = False
                if smile_left_col >= 0 and bs[smile_left_col] > 0.12:
                    is_candidate = False
                if smile_right_col >= 0 and bs[smile_right_col] > 0.12:
                    is_candidate = False

            if is_candidate:
                rest_indices.append(i)

        # Fallback to all frames if no strict rest frames identified
        if len(rest_indices) < 5:
            rest_indices = list(range(n_frames))

        selected_landmarks = session_landmarks[rest_indices]
        neutral_face = np.median(selected_landmarks, axis=0).astype(np.float32)
        return neutral_face


# ---------------------------------------------------------------------------
# Bilateral Symmetry Normalizer
# ---------------------------------------------------------------------------
class BilateralSymmetryNormalizer:
    """Decomposes 3D facial motion into symmetric speech dynamics and asymmetric residuals."""

    @staticmethod
    def flip_horizontal(landmarks: np.ndarray) -> np.ndarray:
        """Horizontally mirrors landmarks across the sagittal (x=0) symmetry plane.
        Flips x coordinate sign and permutes left-right anatomical index correspondences.
        """
        flipped = landmarks.copy()
        # Flip X coordinate
        flipped[..., 0] *= -1.0
        # Permute left-right landmark correspondences
        return flipped[..., SYMMETRY_PERMUTATION, :]

    @classmethod
    def decompose(cls, landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decomposes landmarks into:
            sym_landmarks: Perfectly bilateral symmetric speech motion
            asym_residual: Subject-specific anatomical/expressive asymmetry
        """
        mirrored = cls.flip_horizontal(landmarks)
        sym_landmarks = 0.5 * (landmarks + mirrored)
        asym_residual = 0.5 * (landmarks - mirrored)
        return sym_landmarks.astype(np.float32), asym_residual.astype(np.float32)


# ---------------------------------------------------------------------------
# High-Level Canonical Normalizer Pipeline
# ---------------------------------------------------------------------------
class CanonicalFaceNormalizer:
    """Complete multi-subject facial normalization engine for AI model training.
    
    Transforms raw facial landmarks into:
    1. Canonical Pose-Invariant Space (via Umeyama cranial Procrustes alignment).
    2. Morphological Identity-Decoupled Deltas (relative to subject neutral rest face).
    3. Regional Anthropometric Scaled Kinematics (lips, nose, eyes normalized).
    4. Bilateral Symmetrized Speech Targets (zero asymmetry bias).
    """

    def __init__(self, canonical_template: Optional[np.ndarray] = None):
        if canonical_template is not None:
            self.canonical_template = canonical_template.copy()
        else:
            self.canonical_template = UniversalCanonicalFace.get_template()

        self.cranial_anchors_canonical = self.canonical_template[CRANIAL_BONE_ANCHORS]
        
        # Anthropometric dimensions of canonical reference face
        self.canon_mouth_width = float(np.linalg.norm(
            self.canonical_template[MOUTH_LEFT_CORNER] - self.canonical_template[MOUTH_RIGHT_CORNER]
        ))
        self.canon_ipd = float(np.linalg.norm(
            self.canonical_template[LEFT_PUPIL_INDEX] - self.canonical_template[RIGHT_PUPIL_INDEX]
        ))
        self.canon_nose_height = float(np.linalg.norm(
            self.canonical_template[SELLION_INDEX] - self.canonical_template[SUB_NASALE_INDEX]
        ))

    def compute_subject_neutral_reference(
        self,
        landmarks_seq: np.ndarray,
        blendshapes_seq: Optional[np.ndarray] = None,
        blendshape_names: Optional[List[str]] = None,
        roles_seq: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Estimates and rigidly aligns the subject's neutral rest face to canonical space."""
        raw_neutral = SubjectNeutralFaceEstimator.estimate_neutral_face(
            landmarks_seq, blendshapes_seq, blendshape_names, roles_seq
        )
        # Rigidly align subject raw neutral face to canonical cranial anchors
        s, R, t = umeyama_rigid_alignment(raw_neutral[CRANIAL_BONE_ANCHORS], self.cranial_anchors_canonical)
        aligned_neutral = apply_similarity_transform(raw_neutral, s, R, t)
        return aligned_neutral

    def normalize_frame(
        self,
        landmarks: np.ndarray,
        subject_neutral_face: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Normalizes a single (478, 3) landmark observation.
        
        Returns:
            canonical_landmarks: (478, 3) rigidly aligned to canonical face
            expression_deltas: (478, 3) articulation displacements on canonical template
            symmetric_landmarks: (478, 3) perfectly symmetric speech target
            asymmetric_residual: (478, 3) anatomical asymmetry component
            transform_params: {'scale': s, 'rotation': R, 'translation': t}
        """
        # 1. 3D Rigid Umeyama Alignment to Invariant Cranial Anchors
        s, R, t = umeyama_rigid_alignment(landmarks[CRANIAL_BONE_ANCHORS], self.cranial_anchors_canonical)
        aligned_landmarks = apply_similarity_transform(landmarks, s, R, t)

        # 2. Expression Delta relative to subject's aligned neutral rest face
        raw_deltas = aligned_landmarks - subject_neutral_face

        # 3. Regional Anthropometric Scaling
        # Compute subject neutral dimensions
        subj_mouth_width = max(1e-4, float(np.linalg.norm(
            subject_neutral_face[MOUTH_LEFT_CORNER] - subject_neutral_face[MOUTH_RIGHT_CORNER]
        )))
        subj_ipd = max(1e-4, float(np.linalg.norm(
            subject_neutral_face[LEFT_PUPIL_INDEX] - subject_neutral_face[RIGHT_PUPIL_INDEX]
        )))
        subj_nose_height = max(1e-4, float(np.linalg.norm(
            subject_neutral_face[SELLION_INDEX] - subject_neutral_face[SUB_NASALE_INDEX]
        )))

        scale_mouth = self.canon_mouth_width / subj_mouth_width
        scale_ipd = self.canon_ipd / subj_ipd
        scale_nose = self.canon_nose_height / subj_nose_height

        scaled_deltas = raw_deltas.copy()
        scaled_deltas[LIP_LANDMARKS] *= scale_mouth
        scaled_deltas[EYE_LANDMARKS] *= scale_ipd
        scaled_deltas[NOSE_LANDMARKS] *= scale_nose

        # Cranial bone anchors are strictly zero displacement
        scaled_deltas[CRANIAL_BONE_ANCHORS] = 0.0

        # 4. Project onto Canonical Reference Face
        canonical_landmarks = self.canonical_template + scaled_deltas

        # 5. Bilateral Symmetry Decomposition
        sym_landmarks, asym_residual = BilateralSymmetryNormalizer.decompose(canonical_landmarks)

        return {
            "canonical_landmarks": canonical_landmarks.astype(np.float32),
            "expression_deltas": scaled_deltas.astype(np.float32),
            "symmetric_landmarks": sym_landmarks.astype(np.float32),
            "asymmetric_residual": asym_residual.astype(np.float32),
            "transform_scale": s,
            "transform_rotation": R,
            "transform_translation": t
        }

    def normalize_sequence(
        self,
        landmarks_seq: np.ndarray,
        subject_neutral_face: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Batch-normalizes an entire sequence of landmarks (N, 478, 3)."""
        n_frames = len(landmarks_seq)
        canon_out = np.zeros_like(landmarks_seq, dtype=np.float32)
        deltas_out = np.zeros_like(landmarks_seq, dtype=np.float32)
        sym_out = np.zeros_like(landmarks_seq, dtype=np.float32)
        asym_out = np.zeros_like(landmarks_seq, dtype=np.float32)

        for i in range(n_frames):
            res = self.normalize_frame(landmarks_seq[i], subject_neutral_face)
            canon_out[i] = res["canonical_landmarks"]
            deltas_out[i] = res["expression_deltas"]
            sym_out[i] = res["symmetric_landmarks"]
            asym_out[i] = res["asymmetric_residual"]

        return {
            "canonical_landmarks": canon_out,
            "expression_deltas": deltas_out,
            "symmetric_landmarks": sym_out,
            "asymmetric_residual": asym_out
        }
