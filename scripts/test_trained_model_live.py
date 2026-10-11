"""Live Inference & Telemetry Benchmark for FaceKey Gen2 Trained Model.
Loads checkpoint_best.pt, runs inference on speech audio, calculates real-world metrics & plots animation curves.
"""

import os
import sys
import time
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.face_animator_model import SpeechToFaceAnimator
from src.core.audio_features import LogMelFilterbankExtractor


STANDARD_ARKIT_NAMES = [
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


def run_model_benchmark():
    ckpt_path = os.path.abspath("checkpoints/checkpoint_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    print("=" * 80)
    print(" 🚀 FACEKEY STUDIO - GEN2 MODEL LIVE INFERENCE BENCHMARK")
    print("=" * 80)

    # 1. Inspect Checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print("\n[+] CHECKPOINT METADATA:")
    print(f"  • Trained Epochs     : {ckpt.get('epoch', 'N/A')}")
    print(f"  • Saved Timestamp    : {ckpt.get('saved_at', 'N/A')}")
    print(f"  • Best Validation Loss: {ckpt.get('best_val_loss', 'N/A'):.5f}")
    print(f"  • Final Training Loss: {ckpt.get('train_loss', 'N/A'):.5f}")
    print(f"  • File Size on Disk  : {os.path.getsize(ckpt_path) / (1024 * 1024):.2f} MB")

    # 2. Build Model & Load Weights
    state_dict = ckpt["model_state_dict"]
    cleaned_state = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}

    model = SpeechToFaceAnimator(
        audio_in_dim=64,
        hidden_dim=256,
        num_lstm_layers=2,
        num_blendshapes=52,
        num_dental=4,
        num_emotions=5
    )
    model.load_state_dict(cleaned_state)
    model.eval()

    # 3. Model Architecture Statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    print("\n[+] ARCHITECTURAL SPECIFICATIONS:")
    print(f"  • Total Parameters   : {total_params:,} ({total_params / 1e6:.2f} Million)")
    print(f"  • Trainable Parameters: {trainable_params:,}")
    print(f"  • Raw Weights Size   : {model_size_mb:.2f} MB (FP32) / {model_size_mb / 2:.2f} MB (FP16)")
    print(f"  • Input Audio Dims   : 64 Log-Mel Filterbank Frequency Channels")
    print(f"  • Output Predictions : 52 ARKit Blendshapes + 4 Dental Points + 3D Head Rotation")

    # Layer Groups Breakdown
    groups = {
        "audio_proj": 0, "emotion_embed": 0, "tcn_blocks": 0,
        "bilstm": 0, "blendshape_head": 0, "dental_head": 0, "head_pose_head": 0
    }
    for name, param in model.named_parameters():
        prefix = name.split(".")[0]
        if prefix in groups:
            groups[prefix] += param.numel()

    print("\n[+] LAYER PARAMETER DISTRIBUTION:")
    for group, count in groups.items():
        pct = (count / total_params) * 100
        print(f"  • {group:20s}: {count:8,d} params  ({pct:5.1f}%)")

    # 4. Audio Ingestion & Feature Extraction
    audio_file = os.path.abspath("test_results_epoch10/steve_jobs_clip.wav")
    print(f"\n[+] BENCHMARK AUDIO SOURCE:")
    print(f"  • Audio File: {audio_file}")

    sr, audio_data = wavfile.read(audio_file)
    if audio_data.dtype == np.int16:
        audio_samples = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_samples = audio_data.astype(np.float32) / 2147483648.0
    else:
        audio_samples = audio_data.astype(np.float32)

    if audio_samples.ndim > 1:
        audio_samples = audio_samples.mean(axis=1)

    duration_sec = len(audio_samples) / sr
    target_fps = 30.0
    total_frames = int(round(duration_sec * target_fps))
    print(f"  • Sample Rate : {sr} Hz")
    print(f"  • Duration    : {duration_sec:.2f} seconds ({total_frames} video animation frames @ {target_fps} FPS)")

    # Extract 64-band Log-Mel features
    extractor = LogMelFilterbankExtractor(sample_rate=sr, n_mels=64)
    t_feat_start = time.perf_counter()
    mel_features = extractor.extract_sequence(audio_samples, sample_rate=sr, fps=target_fps, num_frames=total_frames)
    t_feat_ms = (time.perf_counter() - t_feat_start) * 1000.0
    print(f"  • LogMel Extraction Time: {t_feat_ms:.2f} ms ({duration_sec / (t_feat_ms / 1000.0):.1f}x real-time)")

    # 5. Neural Model Inference Benchmark
    audio_tensor = torch.from_numpy(mel_features).unsqueeze(0).float()  # (1, T, 64)
    emotion_tensor = torch.zeros((1,), dtype=torch.long)                 # 0 = Neutral / Natural

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(audio_tensor[:, :30, :], emotion_id=emotion_tensor)

    # High-precision benchmark
    t_infer_start = time.perf_counter()
    with torch.no_grad():
        pred_bs, pred_dental, pred_pose = model(audio_tensor, emotion_id=emotion_tensor)
    t_infer_sec = time.perf_counter() - t_infer_start
    t_infer_ms = t_infer_sec * 1000.0

    bs_out = pred_bs.squeeze(0).cpu().numpy()       # (T, 52)
    dental_out = pred_dental.squeeze(0).cpu().numpy() # (T, 4)
    pose_out = pred_pose.squeeze(0).cpu().numpy()   # (T, 3)

    latency_per_frame_ms = t_infer_ms / total_frames
    infer_fps = total_frames / t_infer_sec

    print("\n[+] INFERENCE PERFORMANCE & SPEED (CPU):")
    print(f"  • Full Clip Inference Time : {t_infer_ms:.2f} ms (For {duration_sec:.2f}s of speech)")
    print(f"  • Latency Per Frame        : {latency_per_frame_ms:.2f} ms/frame")
    print(f"  • Inference Throughput     : {infer_fps:.1f} FPS (Real-time requires 30 FPS -> {infer_fps / 30.0:.1f}x REALTIME SPEED!)")

    # 6. Animation Output Analysis
    print("\n[+] PREDICTED FACIAL KINEMATICS ANALYSIS:")
    print(f"  • Blendshape Output Shape  : {bs_out.shape} -> (Frames, 52 ARKit Blendshapes)")
    print(f"  • Dental Exposure Shape    : {dental_out.shape} -> (Frames, 4 Dental Parameters)")
    print(f"  • Head Orientation Shape   : {pose_out.shape} -> (Frames, Pitch/Yaw/Roll)")

    # Top active blendshapes
    bs_variances = np.var(bs_out, axis=0)
    top_active_indices = np.argsort(bs_variances)[::-1][:8]
    print("\n  Top Dynamically Articulated Blendshapes During Speech:")
    for rank, idx in enumerate(top_active_indices, 1):
        name = STANDARD_ARKIT_NAMES[idx]
        mean_v = np.mean(bs_out[:, idx])
        max_v = np.max(bs_out[:, idx])
        print(f"    {rank}. {name:20s}: Mean={mean_v:.3f}, Peak={max_v:.3f}, Var={bs_variances[idx]:.5f}")

    # 7. Generate Multi-Track Facial Telemetry Plot
    time_axis = np.linspace(0, duration_sec, total_frames)
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True, dpi=130)

    # Track 1: Audio Energy
    audio_energy = np.mean(mel_features, axis=1)
    axs[0].plot(time_axis, audio_energy, color="#3b82f6", lw=1.5, label="Acoustic Energy (Mel RMS)")
    axs[0].set_ylabel("Audio Signal", fontsize=9, fontweight="bold")
    axs[0].set_title("FaceKey Studio Gen2: 100-Epoch Trained Model Real-Time Telemetry Benchmark", fontsize=12, fontweight="bold")
    axs[0].grid(True, linestyle="--", alpha=0.4)
    axs[0].legend(loc="upper right", fontsize=8)

    # Track 2: Jaw & Mouth Visemes
    jaw_open_idx = STANDARD_ARKIT_NAMES.index("jawOpen")
    mouth_pucker_idx = STANDARD_ARKIT_NAMES.index("mouthPucker")
    mouth_funnel_idx = STANDARD_ARKIT_NAMES.index("mouthFunnel")
    mouth_smile_idx = STANDARD_ARKIT_NAMES.index("mouthSmileLeft")

    axs[1].plot(time_axis, bs_out[:, jaw_open_idx], color="#ef4444", lw=2.0, label="jawOpen (Vowel Phonation)")
    axs[1].plot(time_axis, bs_out[:, mouth_funnel_idx], color="#f97316", lw=1.5, label="mouthFunnel (/o/, /u/)")
    axs[1].plot(time_axis, bs_out[:, mouth_pucker_idx], color="#8b5cf6", lw=1.5, label="mouthPucker (/w/, /oo/)")
    axs[1].plot(time_axis, bs_out[:, mouth_smile_idx], color="#10b981", lw=1.5, label="mouthSmile (/i/, /e/)")
    axs[1].set_ylabel("Mouth Blendshapes", fontsize=9, fontweight="bold")
    axs[1].set_ylim(-0.02, 1.0)
    axs[1].grid(True, linestyle="--", alpha=0.4)
    axs[1].legend(loc="upper right", fontsize=8)

    # Track 3: Dental Exposure Dynamics
    axs[2].plot(time_axis, dental_out[:, 0], color="#06b6d4", lw=1.5, label="Upper Teeth Visibility")
    axs[2].plot(time_axis, dental_out[:, 1], color="#3b82f6", lw=1.5, label="Lower Teeth Visibility")
    axs[2].plot(time_axis, dental_out[:, 2], color="#ec4899", lw=1.5, label="Inter-Dental Aperture")
    axs[2].set_ylabel("Dental Dynamics", fontsize=9, fontweight="bold")
    axs[2].grid(True, linestyle="--", alpha=0.4)
    axs[2].legend(loc="upper right", fontsize=8)

    # Track 4: 3D Head Orientation (Degrees)
    axs[3].plot(time_axis, pose_out[:, 0], color="#f59e0b", lw=1.5, label="Head Pitch (Nodding)")
    axs[3].plot(time_axis, pose_out[:, 1], color="#10b981", lw=1.5, label="Head Yaw (Turning)")
    axs[3].plot(time_axis, pose_out[:, 2], color="#6366f1", lw=1.5, label="Head Roll (Tilting)")
    axs[3].set_ylabel("Head Pose (°)", fontsize=9, fontweight="bold")
    axs[3].set_xlabel("Time (seconds)", fontsize=10, fontweight="bold")
    axs[3].grid(True, linestyle="--", alpha=0.4)
    axs[3].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    output_plot = os.path.abspath("test_inference_telemetry.png")
    fig.savefig(output_plot)
    plt.close(fig)
    print(f"\n[✓] Telemetry benchmark plot saved: {output_plot}")

    # Copy to artifacts directory
    artifact_plot = "C:/Users/Vijay/.gemini/antigravity/brain/8cda1bd6-80d5-49fe-bb76-e27120a22c87/test_inference_telemetry.png"
    import shutil
    try:
        shutil.copyfile(output_plot, artifact_plot)
        print(f"[✓] Synced telemetry plot to conversation artifact.")
    except Exception as e:
        print(f"[!] Artifact copy note: {e}")

    # 8. Export ARKit JSON format for 3D digital human player
    export_json = os.path.abspath("test_inference_blendshapes.json")
    json_payload = {
        "audio_file": os.path.basename(audio_file),
        "fps": target_fps,
        "total_frames": total_frames,
        "duration_seconds": duration_sec,
        "blendshape_names": STANDARD_ARKIT_NAMES,
        "frames": bs_out.tolist(),
        "head_pose": pose_out.tolist(),
        "dental": dental_out.tolist()
    }
    with open(export_json, "w", encoding="utf-8") as f:
        json.dump(json_payload, f)
    print(f"[✓] Exported 52-blendshape animation JSON: {export_json} ({os.path.getsize(export_json) / (1024 * 1024):.2f} MB)")
    print("=" * 80)


if __name__ == "__main__":
    run_model_benchmark()
