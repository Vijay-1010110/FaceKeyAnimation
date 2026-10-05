"""FaceKey Studio: Face Skeleton Speech-Reaction Video Generator.
Renders an animated 3D facial skeleton / wireframe driven directly by speech acoustics.
Maps 52 ARKit blendshapes, dental dynamics, and head pose onto a canonical facial wireframe.
Outputs:
  - face_skeleton_reaction.mp4 (Smooth 30 FPS video with HUD audio visualizer)
  - face_skeleton_reaction.gif (Looping animated GIF for instant preview)
"""

import os
import sys
import math
import json
import argparse
from typing import Tuple, Dict, List, Optional

# Enable UTF-8 encoding on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cv2
import torch
import torch.nn as nn

# Ensure root workspace in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.face_animator_model import SpeechToFaceAnimator

# ---------------------------------------------------------------------------
# Canonical Neutral Face Skeleton Template (Normalized [x, y, z] coordinates)
# Center of face at (0, 0, 0), span roughly -1.0 to 1.0
# ---------------------------------------------------------------------------
NEUTRAL_CHIN = np.array([0.0, 0.78, -0.05], dtype=np.float32)
NEUTRAL_JAW_LEFT = np.array([-0.55, 0.40, -0.15], dtype=np.float32)
NEUTRAL_JAW_RIGHT = np.array([0.55, 0.40, -0.15], dtype=np.float32)
NEUTRAL_CHEEK_LEFT = np.array([-0.65, -0.05, -0.2], dtype=np.float32)
NEUTRAL_CHEEK_RIGHT = np.array([0.65, -0.05, -0.2], dtype=np.float32)
NEUTRAL_TEMPLE_LEFT = np.array([-0.58, -0.55, -0.25], dtype=np.float32)
NEUTRAL_TEMPLE_RIGHT = np.array([0.58, -0.55, -0.25], dtype=np.float32)
NEUTRAL_FOREHEAD_TOP = np.array([0.0, -0.85, -0.1], dtype=np.float32)

# Eyebrows
NEUTRAL_L_BROW = [
    np.array([-0.45, -0.42, -0.05]),
    np.array([-0.35, -0.48, -0.02]),
    np.array([-0.22, -0.47, 0.00]),
    np.array([-0.10, -0.42, 0.02])
]
NEUTRAL_R_BROW = [
    np.array([0.10, -0.42, 0.02]),
    np.array([0.22, -0.47, 0.00]),
    np.array([0.35, -0.48, -0.02]),
    np.array([0.45, -0.42, -0.05])
]

# Eyes
NEUTRAL_L_EYE = [
    np.array([-0.42, -0.28, -0.05]),
    np.array([-0.32, -0.34, -0.02]),
    np.array([-0.18, -0.34, 0.00]),
    np.array([-0.12, -0.28, 0.02]),
    np.array([-0.18, -0.22, 0.00]),
    np.array([-0.32, -0.22, -0.02])
]
NEUTRAL_R_EYE = [
    np.array([0.12, -0.28, 0.02]),
    np.array([0.18, -0.34, 0.00]),
    np.array([0.32, -0.34, -0.02]),
    np.array([0.42, -0.28, -0.05]),
    np.array([0.32, -0.22, -0.02]),
    np.array([0.18, -0.22, 0.00])
]

# Nose
NEUTRAL_NOSE_BRIDGE = [
    np.array([0.0, -0.35, 0.08]),
    np.array([0.0, -0.15, 0.16]),
    np.array([0.0, 0.05, 0.22])   # Nose tip
]
NEUTRAL_NOSTRILS = [
    np.array([-0.12, 0.12, 0.14]),
    np.array([0.0, 0.08, 0.20]),
    np.array([0.12, 0.12, 0.14])
]

# Mouth / Lips (Base neutral closed shape)
NEUTRAL_LIPS_OUTER = [
    np.array([-0.28, 0.35, 0.05]),  # Corner Left
    np.array([-0.18, 0.28, 0.10]),  # Upper Lip Left
    np.array([0.00, 0.26, 0.12]),   # Cupid Bow Center
    np.array([0.18, 0.28, 0.10]),   # Upper Lip Right
    np.array([0.28, 0.35, 0.05]),   # Corner Right
    np.array([0.18, 0.42, 0.08]),   # Lower Lip Right
    np.array([0.00, 0.44, 0.10]),   # Lower Lip Bottom Center
    np.array([-0.18, 0.42, 0.08])   # Lower Lip Left
]

NEUTRAL_LIPS_INNER = [
    np.array([-0.22, 0.35, 0.06]),
    np.array([0.00, 0.31, 0.10]),
    np.array([0.22, 0.35, 0.06]),
    np.array([0.00, 0.38, 0.08])
]


def deform_face_skeleton(
    blendshapes: dict,
    dental: np.ndarray,
    head_pose: np.ndarray
) -> dict:
    """Deforms the 3D canonical face skeleton according to model's predicted animation keys."""
    jaw_open = float(blendshapes.get("jawOpen", 0.0))
    mouth_smile_l = float(blendshapes.get("mouthSmileLeft", 0.0))
    mouth_smile_r = float(blendshapes.get("mouthSmileRight", 0.0))
    mouth_pucker = float(blendshapes.get("mouthPucker", 0.0))
    mouth_funnel = float(blendshapes.get("mouthFunnel", 0.0))
    brow_up = float(blendshapes.get("browInnerUp", 0.0))
    brow_down_l = float(blendshapes.get("browDownLeft", 0.0))
    brow_down_r = float(blendshapes.get("browDownRight", 0.0))
    eye_blink_l = float(blendshapes.get("eyeBlinkLeft", 0.0))
    eye_blink_r = float(blendshapes.get("eyeBlinkRight", 0.0))

    # Amplify speech jaw dynamics slightly for clear visual response
    jaw_drop = jaw_open * 0.35

    # 1. Deform Jaw & Chin
    chin = NEUTRAL_CHIN.copy()
    chin[1] += jaw_drop * 1.2
    chin[2] -= jaw_drop * 0.2

    jaw_l = NEUTRAL_JAW_LEFT.copy()
    jaw_l[1] += jaw_drop * 0.5
    jaw_r = NEUTRAL_JAW_RIGHT.copy()
    jaw_r[1] += jaw_drop * 0.5

    # 2. Deform Lips (Speech kinematics)
    lips_outer = [p.copy() for p in NEUTRAL_LIPS_OUTER]
    lips_inner = [p.copy() for p in NEUTRAL_LIPS_INNER]

    # Smile stretches corners outward and up
    smile_x = (mouth_smile_l + mouth_smile_r) * 0.08
    smile_y = (mouth_smile_l + mouth_smile_r) * 0.06
    lips_outer[0][0] -= smile_x
    lips_outer[0][1] -= smile_y
    lips_outer[4][0] += smile_x
    lips_outer[4][1] -= smile_y

    lips_inner[0][0] -= smile_x * 0.8
    lips_inner[2][0] += smile_x * 0.8

    # Pucker/Funnel narrows mouth width and rounds lips
    pucker_factor = (mouth_pucker * 0.12) + (mouth_funnel * 0.08)
    lips_outer[0][0] += pucker_factor
    lips_outer[4][0] -= pucker_factor
    lips_inner[0][0] += pucker_factor * 0.8
    lips_inner[2][0] -= pucker_factor * 0.8

    # Lower lip drops with jaw
    lips_outer[5][1] += jaw_drop
    lips_outer[6][1] += jaw_drop * 1.1
    lips_outer[7][1] += jaw_drop
    lips_inner[3][1] += jaw_drop * 0.95

    # Upper lip slightly moves
    upper_drop = jaw_drop * 0.15
    lips_outer[1][1] += upper_drop
    lips_outer[2][1] += upper_drop
    lips_outer[3][1] += upper_drop
    lips_inner[1][1] += upper_drop

    # 3. Eyebrows
    l_brow = [p.copy() for p in NEUTRAL_L_BROW]
    r_brow = [p.copy() for p in NEUTRAL_R_BROW]
    for p in l_brow:
        p[1] -= (brow_up * 0.08) - (brow_down_l * 0.06)
    for p in r_brow:
        p[1] -= (brow_up * 0.08) - (brow_down_r * 0.06)

    # 4. Eyes (Blinking / squinting)
    l_eye = [p.copy() for p in NEUTRAL_L_EYE]
    r_eye = [p.copy() for p in NEUTRAL_R_EYE]
    blink_scale_l = max(0.08, 1.0 - eye_blink_l)
    blink_scale_r = max(0.08, 1.0 - eye_blink_r)

    # Scale eye height relative to eye center
    eye_l_center_y = -0.28
    eye_r_center_y = -0.28
    for p in l_eye:
        p[1] = eye_l_center_y + (p[1] - eye_l_center_y) * blink_scale_l
    for p in r_eye:
        p[1] = eye_r_center_y + (p[1] - eye_r_center_y) * blink_scale_r

    # Head Pose Rotation (Pitch, Yaw, Roll)
    pitch = math.radians(head_pose[0] * 15.0)
    yaw = math.radians(head_pose[1] * 20.0)
    roll = math.radians(head_pose[2] * 10.0)

    # 3D Euler Rotation Matrix
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(pitch), -math.sin(pitch)],
        [0, math.sin(pitch), math.cos(pitch)]
    ])
    Ry = np.array([
        [math.cos(yaw), 0, math.sin(yaw)],
        [0, 1, 0],
        [-math.sin(yaw), 0, math.cos(yaw)]
    ])
    Rz = np.array([
        [math.cos(roll), -math.sin(roll), 0],
        [math.sin(roll), math.cos(roll), 0],
        [0, 0, 1]
    ])
    R = Rz @ Ry @ Rx

    def rotate_pt(pt):
        return R @ pt

    # Apply 3D rotation to all skeleton components
    contour = [
        rotate_pt(NEUTRAL_FOREHEAD_TOP),
        rotate_pt(NEUTRAL_TEMPLE_LEFT),
        rotate_pt(NEUTRAL_CHEEK_LEFT),
        rotate_pt(jaw_l),
        rotate_pt(chin),
        rotate_pt(jaw_r),
        rotate_pt(NEUTRAL_CHEEK_RIGHT),
        rotate_pt(NEUTRAL_TEMPLE_RIGHT)
    ]

    return {
        "contour": contour,
        "l_brow": [rotate_pt(p) for p in l_brow],
        "r_brow": [rotate_pt(p) for p in r_brow],
        "l_eye": [rotate_pt(p) for p in l_eye],
        "r_eye": [rotate_pt(p) for p in r_eye],
        "nose_bridge": [rotate_pt(p) for p in NEUTRAL_NOSE_BRIDGE],
        "nostrils": [rotate_pt(p) for p in NEUTRAL_NOSTRILS],
        "lips_outer": [rotate_pt(p) for p in lips_outer],
        "lips_inner": [rotate_pt(p) for p in lips_inner],
        "jaw_open_val": jaw_open,
        "teeth_gap": max(0.0, jaw_drop)
    }


def project_to_canvas(pt3d: np.ndarray, width: int, height: int) -> tuple:
    """Projects 3D normalized coordinates onto 2D canvas with perspective."""
    focal = 3.0
    z = pt3d[2] + focal
    x_proj = (pt3d[0] / z) * 1.8
    y_proj = (pt3d[1] / z) * 1.8

    cx = width // 2
    cy = int(height * 0.46)
    scale = min(width, height) * 0.44

    x_px = int(cx + x_proj * scale)
    y_px = int(cy + y_proj * scale)
    return (x_px, y_px)


def draw_skeleton_frame(
    skeleton: dict,
    audio_energy: float,
    frame_idx: int,
    total_frames: int,
    fps: int = 30,
    width: int = 800,
    height: int = 800
) -> np.ndarray:
    """Draws a high-tech glowing wireframe face skeleton with audio reactivity and HUD."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # Cyberpunk dark background with subtle grid
    canvas[:] = (12, 16, 24)
    grid_spacing = 40
    for gx in range(0, width, grid_spacing):
        cv2.line(canvas, (gx, 0), (gx, height), (18, 24, 34), 1)
    for gy in range(0, height, grid_spacing):
        cv2.line(canvas, (0, gy), (width, gy), (18, 24, 34), 1)

    # Helper to draw polyline
    def draw_wire(pts, color, thickness=2, is_closed=False):
        px_pts = [project_to_canvas(p, width, height) for p in pts]
        for i in range(len(px_pts) - 1):
            cv2.line(canvas, px_pts[i], px_pts[i + 1], color, thickness, cv2.LINE_AA)
        if is_closed and len(px_pts) > 2:
            cv2.line(canvas, px_pts[-1], px_pts[0], color, thickness, cv2.LINE_AA)
        # Draw joint nodes
        for pt in px_pts:
            cv2.circle(canvas, pt, thickness + 1, (min(255, color[0] + 40), min(255, color[1] + 40), min(255, color[2] + 40)), -1, cv2.LINE_AA)

    # 1. Face Contour (Cyan Wireframe)
    draw_wire(skeleton["contour"], (180, 140, 50), thickness=2, is_closed=True)

    # 2. Eyebrows (Neon Blue)
    draw_wire(skeleton["l_brow"], (255, 180, 60), thickness=2, is_closed=False)
    draw_wire(skeleton["r_brow"], (255, 180, 60), thickness=2, is_closed=False)

    # 3. Eyes (Bright Cyan)
    draw_wire(skeleton["l_eye"], (240, 200, 70), thickness=2, is_closed=True)
    draw_wire(skeleton["r_eye"], (240, 200, 70), thickness=2, is_closed=True)

    # Pupils / Iris centers
    l_pupil = project_to_canvas((skeleton["l_eye"][0] + skeleton["l_eye"][3]) * 0.5, width, height)
    r_pupil = project_to_canvas((skeleton["r_eye"][0] + skeleton["r_eye"][3]) * 0.5, width, height)
    cv2.circle(canvas, l_pupil, 4, (255, 255, 100), -1, cv2.LINE_AA)
    cv2.circle(canvas, r_pupil, 4, (255, 255, 100), -1, cv2.LINE_AA)

    # 4. Nose (Subtle Ice Blue)
    draw_wire(skeleton["nose_bridge"], (200, 170, 90), thickness=2, is_closed=False)
    draw_wire(skeleton["nostrils"], (200, 170, 90), thickness=2, is_closed=False)

    # 5. Teeth / Dental Exposure (White bars visible when mouth opens)
    if skeleton["teeth_gap"] > 0.02:
        inner_top = project_to_canvas(skeleton["lips_inner"][1], width, height)
        inner_bot = project_to_canvas(skeleton["lips_inner"][3], width, height)
        teeth_w = int(width * 0.08)
        # Upper teeth line
        cv2.line(canvas, (inner_top[0] - teeth_w, inner_top[1] + 2), (inner_top[0] + teeth_w, inner_top[1] + 2), (240, 240, 240), 3, cv2.LINE_AA)
        # Lower teeth line
        cv2.line(canvas, (inner_bot[0] - teeth_w, inner_bot[1] - 2), (inner_bot[0] + teeth_w, inner_bot[1] - 2), (220, 220, 220), 2, cv2.LINE_AA)

    # 6. Lips / Talking Mouth (Active Speech Color: Electric Orange / Crimson)
    speech_glow = int(min(255, 120 + audio_energy * 300))
    lip_color_outer = (40, 140, speech_glow)  # BGR
    lip_color_inner = (30, 90, 240)

    draw_wire(skeleton["lips_outer"], lip_color_outer, thickness=3, is_closed=True)
    draw_wire(skeleton["lips_inner"], lip_color_inner, thickness=2, is_closed=True)

    # ---------------------------------------------------------------------------
    # HUD Telemetry & Real-Time Audio Level Meter
    # ---------------------------------------------------------------------------
    # Header
    cv2.putText(canvas, "FACEKEY 3D SKELETON - AUDIO SPEECH REACTION ENGINE", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2, cv2.LINE_AA)
    time_sec = frame_idx / fps
    cv2.putText(canvas, f"TIME: {time_sec:05.2f}s | FRAME: {frame_idx:04d}/{total_frames:04d} @ {fps}FPS", (30, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 160, 200), 1, cv2.LINE_AA)

    # Audio Energy VU Meter Bar (Bottom HUD)
    hud_y = height - 70
    cv2.rectangle(canvas, (30, hud_y), (width - 30, hud_y + 35), (20, 28, 42), -1)
    cv2.rectangle(canvas, (30, hud_y), (width - 30, hud_y + 35), (50, 70, 100), 1)

    meter_w = width - 60
    bar_fill = int(meter_w * min(1.0, audio_energy * 3.5))
    cv2.rectangle(canvas, (30, hud_y), (30 + bar_fill, hud_y + 35), (30, 180, 240), -1)
    cv2.putText(canvas, f"SPEECH ACOUSTIC ENERGY: {audio_energy:.4f}", (45, hud_y + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Active Blendshape Telemetry on Left
    jaw_val = skeleton["jaw_open_val"]
    cv2.putText(canvas, f"jawOpen: {jaw_val:.3f}", (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 180, 255), 1, cv2.LINE_AA)
    jaw_bar_len = int(120 * min(1.0, jaw_val * 4.0))
    cv2.rectangle(canvas, (30, 125), (150, 133), (30, 40, 55), -1)
    cv2.rectangle(canvas, (30, 125), (30 + jaw_bar_len, 133), (30, 180, 255), -1)

    # Live status badge
    is_active = audio_energy > 0.015
    status_text = "SPEAKING (REACTIVE)" if is_active else "IDLE / PAUSE"
    status_color = (0, 220, 100) if is_active else (100, 110, 120)
    cv2.circle(canvas, (width - 180, 40), 6, status_color, -1, cv2.LINE_AA)
    cv2.putText(canvas, status_text, (width - 165, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1, cv2.LINE_AA)

    return canvas


def generate_face_skeleton_video(
    model: nn.Module,
    dataset_npz: str,
    output_mp4: str,
    output_gif: str,
    device: torch.device,
    duration_frames: int = 180,  # 6.0 seconds @ 30 FPS
    fps: int = 30
):
    """Feeds validation speech into the model and renders animated face skeleton MP4 and GIF."""
    print("=" * 82)
    print(" 🎬 RENDERING FACE SKELETON SPEECH-REACTION VIDEO & GIF")
    print(f" Source Dataset : {dataset_npz}")
    print(f" Output Video   : {output_mp4}")
    print(f" Output GIF     : {output_gif}")
    print(f" Total Frames   : {duration_frames} ({duration_frames / fps:.1f}s @ {fps} FPS)")
    print("=" * 82)

    os.makedirs(os.path.dirname(os.path.abspath(output_mp4)), exist_ok=True)
    data = np.load(dataset_npz, allow_pickle=True)

    # Extract active speech sequence from validation split
    split_mask = data.get("split_mask", None)
    if split_mask is not None:
        val_mask = ~split_mask
    else:
        n_tot = len(data["blendshapes"])
        val_mask = np.zeros(n_tot, dtype=bool)
        val_mask[int(0.85 * n_tot):] = True

    val_indices = np.where(val_mask)[0]
    total_val = len(val_indices)

    # Find a lively speech section with active energy
    ae_all = data["audio_energy"][val_indices]
    # Pick a segment with high variance / lively speech
    best_start = 0
    max_energy_sum = 0
    for offset in range(0, total_val - duration_frames, 30):
        e_sum = np.sum(ae_all[offset:offset + duration_frames])
        if e_sum > max_energy_sum:
            max_energy_sum = e_sum
            best_start = offset

    eval_indices = val_indices[best_start:best_start + duration_frames]
    n_frames = len(eval_indices)

    ae = data["audio_energy"][eval_indices]
    ap = data["audio_speech_prob"][eval_indices]

    # Construct 64-dim lag acoustic features
    audio_feat = np.zeros((n_frames, 64), dtype=np.float32)
    audio_feat[:, 0] = ae
    audio_feat[:, 1] = ap
    for lag in range(1, 16):
        if 2 * lag + 1 < 64:
            audio_feat[lag:, 2 * lag] = ae[:-lag]
            audio_feat[lag:, 2 * lag + 1] = ap[:-lag]

    # Run Model Inference
    inp_tensor = torch.from_numpy(audio_feat).unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            pred_bs, pred_dental, pred_pose = model(inp_tensor)

    pred_bs_np = pred_bs.squeeze(0).cpu().numpy()      # (N, 52)
    pred_dental_np = pred_dental.squeeze(0).cpu().numpy()  # (N, 4)
    pred_pose_np = pred_pose.squeeze(0).cpu().numpy()      # (N, 3)

    STANDARD_BS_NAMES = [
        "eyeBlinkLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft", "eyeSquintLeft", "eyeWideLeft",
        "eyeBlinkRight", "eyeLookDownRight", "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight", "eyeSquintRight", "eyeWideRight",
        "jawForward", "jawLeft", "jawRight", "jawOpen",
        "mouthClose", "mouthFunnel", "mouthPucker", "mouthLeft", "mouthRight",
        "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight",
        "mouthDimpleLeft", "mouthDimpleRight", "mouthStretchLeft", "mouthStretchRight",
        "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
        "mouthPressLeft", "mouthPressRight", "mouthLowerDownLeft", "mouthLowerDownRight",
        "mouthUpperUpLeft", "mouthUpperUpRight",
        "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
        "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
        "noseSneerLeft", "noseSneerRight", "tongueOut"
    ]

    width, height = 720, 720
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(output_mp4, fourcc, fps, (width, height))

    gif_frames = []

    print(f"[*] Rendering {n_frames} frames of animated 3D face skeleton...")
    for f in range(n_frames):
        bs_dict = {STANDARD_BS_NAMES[j]: pred_bs_np[f, j] for j in range(min(52, len(STANDARD_BS_NAMES)))}
        skeleton = deform_face_skeleton(
            blendshapes=bs_dict,
            dental=pred_dental_np[f],
            head_pose=pred_pose_np[f]
        )

        frame_img = draw_skeleton_frame(
            skeleton=skeleton,
            audio_energy=float(ae[f]),
            frame_idx=f,
            total_frames=n_frames,
            fps=fps,
            width=width,
            height=height
        )

        out_video.write(frame_img)

        # Downsample slightly for fast GIF preview
        if f % 2 == 0:  # 15 FPS for GIF
            from PIL import Image
            rgb_frame = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame).resize((480, 480), Image.Resampling.LANCZOS)
            gif_frames.append(pil_img)

    out_video.release()
    print(f"[✓] MP4 Video Rendered: {output_mp4}")

    if gif_frames:
        print("[*] Generating looping GIF animation...")
        gif_frames[0].save(
            output_gif,
            save_all=True,
            append_images=gif_frames[1:],
            duration=int(1000 / 15),
            loop=0,
            optimize=True
        )
        print(f"[✓] Animated GIF Saved: {output_gif}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render Face Skeleton Speech-Reaction Video")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--dataset", type=str, default="sessions/normalized_training_dataset.npz", help="Dataset NPZ")
    parser.add_argument("--output-mp4", type=str, default="test_results_epoch10/face_skeleton_reaction.mp4", help="Output MP4 path")
    parser.add_argument("--output-gif", type=str, default="test_results_epoch10/face_skeleton_reaction.gif", help="Output GIF path")
    parser.add_argument("--duration-frames", type=int, default=180, help="Frames to render (180 = 6.0s @ 30fps)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device}")

    # Resolve checkpoint
    ckpt_path = args.checkpoint
    if not ckpt_path:
        candidates = [
            "test_results_epoch10/checkpoint_latest.pt",
            "checkpoints/checkpoint_latest.pt",
            "checkpoints/checkpoint_best.pt"
        ]
        import glob
        snap_ckpts = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/**/checkpoint_latest.pt"), recursive=True)
        if snap_ckpts:
            candidates.insert(0, snap_ckpts[0])

        for c in candidates:
            if os.path.exists(c):
                ckpt_path = c
                break

    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError("Could not find model checkpoint.")

    print(f"[*] Loading model from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = SpeechToFaceAnimator(audio_in_dim=64, hidden_dim=256, num_lstm_layers=2, num_blendshapes=52, num_dental=4).to(device)
    st = ckpt.get("model_state_dict", ckpt)
    cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in st.items()}
    model.load_state_dict(cleaned)
    model.eval()

    generate_face_skeleton_video(
        model=model,
        dataset_npz=args.dataset,
        output_mp4=args.output_mp4,
        output_gif=args.output_gif,
        device=device,
        duration_frames=args.duration_frames
    )
