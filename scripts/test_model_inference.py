"""Speech-to-Facial Animation: Test & Inference Evaluation Engine.
Evaluates trained checkpoints on held-out validation subjects, custom audio files, or microphone recordings.
Outputs:
  1. Side-by-side validation comparison plots (Ground Truth vs Model Prediction).
  2. Numerical accuracy metrics (MAE, Pearson Correlation, Lip-Sync Dynamic Error).
  3. Exportable ARKit 52-blendshape JSON / CSV for 3D digital avatars (Blender, Unreal Engine, Maya).
"""

import os
import sys
import json
import time
import argparse

# Enable UTF-8 encoding on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from typing import Tuple, Dict, List, Optional, Any
import numpy as np
import torch
import torch.nn as nn

# Ensure root workspace in path
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


def load_model_from_checkpoint(ckpt_path: str, device: torch.device) -> Tuple[nn.Module, dict]:
    """Loads SpeechToFaceAnimator with weights from checkpoint."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    print(f"[*] Loading model checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    model = SpeechToFaceAnimator(
        audio_in_dim=64,
        hidden_dim=256,
        num_lstm_layers=2,
        num_blendshapes=52,
        num_dental=4
    ).to(device)

    state_dict = ckpt.get("model_state_dict", ckpt)
    # Strip 'module.' prefix if trained under DataParallel
    cleaned_state = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(cleaned_state)
    model.eval()

    epoch = ckpt.get("epoch", "N/A")
    val_loss = ckpt.get("val_loss", "N/A")
    best_val_loss = ckpt.get("best_val_loss", "N/A")
    saved_at = ckpt.get("saved_at", "Unknown")

    print(f"[✓] Model loaded successfully! (Trained Epoch: {epoch}, Val Loss: {val_loss}, Best: {best_val_loss}, Saved: {saved_at})")
    return model, ckpt


def evaluate_on_validation_data(
    model: nn.Module,
    dataset_npz: str,
    output_dir: str,
    device: torch.device,
    sample_frames: int = 400
):
    """Evaluates model predictions against ground truth validation speech sequences."""
    print("=" * 82)
    print(" 🧪 RUNNING BENCHMARK EVALUATION ON HELD-OUT TEST SUBJECTS")
    print(f" Dataset NPZ : {dataset_npz}")
    print(f" Output Dir  : {output_dir}")
    print("=" * 82)

    os.makedirs(output_dir, exist_ok=True)
    data = np.load(dataset_npz, allow_pickle=True)

    # Resolve validation mask
    split_mask = data.get("split_mask", None)
    if split_mask is not None:
        val_mask = ~split_mask
    else:
        n_tot = len(data["blendshapes"])
        val_mask = np.zeros(n_tot, dtype=bool)
        val_mask[int(0.85 * n_tot):] = True

    val_indices = np.where(val_mask)[0]
    total_val_frames = len(val_indices)
    print(f"[+] Total Available Validation Test Frames: {total_val_frames:,}")

    if total_val_frames < 64:
        print("[!] Not enough validation frames to evaluate.")
        return

    # Select contiguous test segment
    eval_len = min(sample_frames, total_val_frames - 64)
    start_idx = val_indices[0]
    end_idx = start_idx + eval_len

    gt_blendshapes = data["blendshapes"][start_idx:end_idx]      # (N, 52)
    ae = data["audio_energy"][start_idx:end_idx] if "audio_energy" in data else np.zeros(eval_len, dtype=np.float32)
    ap = data["audio_speech_prob"][start_idx:end_idx] if "audio_speech_prob" in data else np.ones(eval_len, dtype=np.float32)

    # Load 64-band Log-Mel spectral features if available, else construct legacy lag features
    if "audio_features" in data:
        audio_feat = data["audio_features"][start_idx:end_idx].astype(np.float32)
    else:
        audio_feat = np.zeros((eval_len, 64), dtype=np.float32)
        audio_feat[:, 0] = ae
        audio_feat[:, 1] = ap
        for lag in range(1, 16):
            if 2 * lag + 1 < 64:
                audio_feat[lag:, 2 * lag] = ae[:-lag]
                audio_feat[lag:, 2 * lag + 1] = ap[:-lag]

    # Model inference
    inp_tensor = torch.from_numpy(audio_feat).unsqueeze(0).to(device)  # (1, N, 64)
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            pred_bs, pred_dental, pred_pose = model(inp_tensor)

    pred_bs_np = pred_bs.squeeze(0).cpu().numpy()  # (N, 52)

    # Resolve blendshape names
    bs_names = list(data.get("blendshape_names", []))
    if not bs_names or len(bs_names) != 52:
        bs_names = STANDARD_ARKIT_NAMES

    # Compute Error Metrics
    mae_all = np.mean(np.abs(pred_bs_np - gt_blendshapes))
    jaw_idx = bs_names.index("jawOpen") if "jawOpen" in bs_names else 17
    mae_jaw = np.mean(np.abs(pred_bs_np[:, jaw_idx] - gt_blendshapes[:, jaw_idx]))

    # Correlation on jawOpen (primary speech driver)
    std_p = np.std(pred_bs_np[:, jaw_idx])
    std_g = np.std(gt_blendshapes[:, jaw_idx])
    if std_p > 1e-6 and std_g > 1e-6:
        corr_jaw = float(np.corrcoef(pred_bs_np[:, jaw_idx], gt_blendshapes[:, jaw_idx])[0, 1])
    else:
        corr_jaw = 0.0

    print("\n" + "=" * 50)
    print(" 📊 MODEL ACCURACY & LIP-SYNC BENCHMARK RESULTS")
    print("=" * 50)
    print(f"  • Overall 52-Blendshape MAE : {mae_all:.4f} (Scale: 0.0 to 1.0)")
    print(f"  • Jaw Open (Speech) MAE     : {mae_jaw:.4f}")
    print(f"  • Jaw Open Sync Correlation : {corr_jaw * 100:.1f}%")
    print(f"  • Dynamic Range (Min - Max) : {pred_bs_np.min():.3f} - {pred_bs_np.max():.3f}")
    print("=" * 50 + "\n")

    # Generate Comparison Plot
    plot_path = os.path.join(output_dir, "test_predictions_vs_groundtruth.png")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True, dpi=120)

        time_axis = np.arange(eval_len) / 30.0  # Seconds @ 30 FPS

        # Plot 1: jawOpen
        axs[0].plot(time_axis, gt_blendshapes[:, jaw_idx], label="Ground Truth (Real Actor)", color="#2563eb", lw=2)
        axs[0].plot(time_axis, pred_bs_np[:, jaw_idx], label="Model Prediction", color="#f97316", lw=2, linestyle="--")
        axs[0].set_title(f"Speech Driver: jawOpen (Sync Correlation: {corr_jaw * 100:.1f}%)", fontweight="bold")
        axs[0].set_ylabel("Activation [0 - 1]")
        axs[0].grid(True, linestyle="--", alpha=0.4)
        axs[0].legend(loc="upper right")

        # Plot 2: mouthSmileLeft / Right
        smile_idx = bs_names.index("mouthSmileLeft") if "mouthSmileLeft" in bs_names else 23
        axs[1].plot(time_axis, gt_blendshapes[:, smile_idx], label="Ground Truth", color="#2563eb", lw=2)
        axs[1].plot(time_axis, pred_bs_np[:, smile_idx], label="Model Prediction", color="#10b981", lw=2, linestyle="--")
        axs[1].set_title("Expression Nuance: mouthSmileLeft", fontweight="bold")
        axs[1].set_ylabel("Activation [0 - 1]")
        axs[1].grid(True, linestyle="--", alpha=0.4)
        axs[1].legend(loc="upper right")

        # Plot 3: mouthPucker ("O" / "U" sounds)
        pucker_idx = bs_names.index("mouthPucker") if "mouthPucker" in bs_names else 20
        axs[2].plot(time_axis, gt_blendshapes[:, pucker_idx], label="Ground Truth", color="#2563eb", lw=2)
        axs[2].plot(time_axis, pred_bs_np[:, pucker_idx], label="Model Prediction", color="#8b5cf6", lw=2, linestyle="--")
        axs[2].set_title("Phoneme Articulation: mouthPucker ('O', 'U', 'W' sounds)", fontweight="bold")
        axs[2].set_ylabel("Activation [0 - 1]")
        axs[2].grid(True, linestyle="--", alpha=0.4)
        axs[2].legend(loc="upper right")

        # Plot 4: Input Audio Energy
        axs[3].plot(time_axis, ae, label="Speech Acoustic Energy (Input)", color="#64748b", lw=1.5)
        axs[3].fill_between(time_axis, 0, ae, color="#94a3b8", alpha=0.3)
        axs[3].set_title("Input Acoustic Audio Energy Curve", fontweight="bold")
        axs[3].set_xlabel("Time (seconds)")
        axs[3].set_ylabel("Energy RMS")
        axs[3].grid(True, linestyle="--", alpha=0.4)
        axs[3].legend(loc="upper right")

        fig.suptitle("FaceKey Studio — Speech-to-Animation Model Test Evaluation", fontsize=15, fontweight="bold", y=0.98)
        fig.tight_layout()
        fig.subplots_adjust(top=0.92)
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"[✓] Saved side-by-side comparison plot: {plot_path}")
    except Exception as e:
        print(f"[!] Warning: Could not generate comparison plot ({e})")

    # Export test animation frames to JSON & CSV
    json_path = os.path.join(output_dir, "test_animation_arkit.json")
    csv_path = os.path.join(output_dir, "test_animation_blendshapes.csv")

    export_payload = {
        "metadata": {
            "fps": 30,
            "total_frames": eval_len,
            "duration_sec": eval_len / 30.0,
            "blendshape_count": 52,
            "mae": float(mae_all),
            "jaw_correlation": float(corr_jaw)
        },
        "blendshape_names": bs_names,
        "frames": []
    }

    for f_idx in range(eval_len):
        frame_dict = {
            "frame": f_idx,
            "time_sec": round(f_idx / 30.0, 4),
            "blendshapes": {bs_names[i]: float(pred_bs_np[f_idx, i]) for i in range(52)}
        }
        export_payload["frames"].append(frame_dict)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)
    print(f"[✓] Exported ARKit 3D animation JSON: {json_path}")

    # CSV Export (For Blender / Maya)
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_sec"] + bs_names)
        for f_idx in range(eval_len):
            writer.writerow([round(f_idx / 30.0, 4)] + [round(float(v), 5) for v in pred_bs_np[f_idx]])
    print(f"[✓] Exported Blender/Maya CSV: {csv_path}")

    return {
        "mae_all": mae_all,
        "mae_jaw": mae_jaw,
        "corr_jaw": corr_jaw,
        "plot_path": plot_path,
        "json_path": json_path,
        "csv_path": csv_path
    }


def infer_on_audio_file(
    model: nn.Module,
    audio_path: str,
    output_dir: str,
    device: torch.device
):
    """Extracts acoustics from custom audio file and generates animation timeline."""
    print("=" * 82)
    print(f" 🎙️ INFERRING ANIMATION FROM AUDIO FILE: '{audio_path}'")
    print("=" * 82)

    os.makedirs(output_dir, exist_ok=True)

    # Read audio using scipy
    try:
        from scipy.io import wavfile
        sr, samples = wavfile.read(audio_path)
    except Exception as e:
        print(f"[!] Error loading audio file: {e}")
        return

    # Convert to mono float32
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if samples.dtype != np.float32:
        samples = samples.astype(np.float32) / (np.max(np.abs(samples)) + 1e-9)

    # Frame-rate alignment: 30 FPS animation frames
    fps = 30
    hop_length = int(sr / fps)
    n_frames = len(samples) // hop_length

    print(f"[*] Audio length: {len(samples) / sr:.2f}s | Sample Rate: {sr} Hz | Animation Frames @ {fps}fps: {n_frames}")

    # Extract 64-band Log-Mel Filterbank spectral features matching the training pipeline
    extractor = LogMelFilterbankExtractor(sample_rate=sr, n_mels=64)
    audio_feat = extractor.extract_sequence(samples, sample_rate=sr, fps=fps, num_frames=n_frames)

    # Compute short-term RMS energy curve for telemetry and visualization
    ae = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        chunk = samples[i * hop_length:(i + 1) * hop_length]
        ae[i] = float(np.sqrt(np.mean(chunk**2))) if len(chunk) > 0 else 0.0
    if ae.max() > 0:
        ae = ae / ae.max()

    inp_tensor = torch.from_numpy(audio_feat).unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            pred_bs, pred_dental, pred_pose = model(inp_tensor)

    pred_bs_np = pred_bs.squeeze(0).cpu().numpy()

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    out_json = os.path.join(output_dir, f"{base_name}_animation_arkit.json")

    export_payload = {
        "audio_file": os.path.basename(audio_path),
        "fps": fps,
        "total_frames": n_frames,
        "duration_sec": round(n_frames / fps, 2),
        "blendshape_names": STANDARD_ARKIT_NAMES,
        "frames": [
            {
                "frame": f,
                "time_sec": round(f / fps, 4),
                "blendshapes": {STANDARD_ARKIT_NAMES[j]: float(pred_bs_np[f, j]) for j in range(52)}
            }
            for f in range(n_frames)
        ]
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)
    print(f"[✓] Generated speech animation successfully: {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test and Inference Evaluation for FaceKey Speech-to-Animation")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt file")
    parser.add_argument("--dataset", type=str, default=None, help="Path to normalized_training_dataset.npz")
    parser.add_argument("--audio-file", type=str, default=None, help="Optional custom audio file (.wav) to animate")
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/test_results", help="Output directory for test results")
    parser.add_argument("--hf-repo", type=str, default="VijayTheOne/facekey-dataset-chunks", help="Hugging Face repo for checkpoint download")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face token")
    parser.add_argument("--sample-frames", type=int, default=450, help="Number of frames to evaluate (~15 seconds @ 30fps)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Evaluation Device: {device}")

    ckpt_path = args.checkpoint
    # Auto-resolve checkpoint if not provided
    if not ckpt_path:
        local_candidates = [
            "/kaggle/working/checkpoints/checkpoint_best.pt",
            "/kaggle/working/checkpoints/checkpoint_latest.pt",
            "\\kaggle\\working\\checkpoints\\checkpoints\\checkpoint_best.pt",
            "\\kaggle\\working\\checkpoints\\checkpoints\\checkpoint_latest.pt",
            "checkpoints/checkpoint_best.pt",
            "checkpoints/checkpoint_latest.pt"
        ]
        for c in local_candidates:
            if os.path.exists(c):
                ckpt_path = c
                break

    # If still not found, download from Hugging Face Hub
    if not ckpt_path or not os.path.exists(ckpt_path):
        token = args.hf_token or os.environ.get("HF_TOKEN")
        target_ckpt_dir = "/kaggle/working/checkpoints" if os.path.exists("/kaggle") else os.path.abspath("checkpoints")
        print(f"[*] Pulling 'checkpoint_best.pt' from Hugging Face: {args.hf_repo}...")
        try:
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download(
                repo_id=args.hf_repo,
                filename="checkpoints/checkpoint_best.pt",
                repo_type="dataset",
                token=token,
                local_dir=target_ckpt_dir
            )
        except Exception as e:
            print(f"[!] Could not download best checkpoint, trying latest ({e})...")
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download(
                repo_id=args.hf_repo,
                filename="checkpoints/checkpoint_latest.pt",
                repo_type="dataset",
                token=token,
                local_dir=target_ckpt_dir
            )

    model, ckpt_meta = load_model_from_checkpoint(ckpt_path, device)

    # 1. Validation Dataset Evaluation
    ds_path = args.dataset
    if not ds_path:
        ds_candidates = [
            "/kaggle/working/training_data/normalized_training_dataset.npz",
            "training_data/normalized_training_dataset.npz"
        ]
        for d in ds_candidates:
            if os.path.exists(d):
                ds_path = d
                break

    if ds_path and os.path.exists(ds_path):
        eval_results = evaluate_on_validation_data(
            model=model,
            dataset_npz=ds_path,
            output_dir=args.output_dir,
            device=device,
            sample_frames=args.sample_frames
        )

        # Upload comparison plot to Hugging Face Hub if token available
        token = args.hf_token or os.environ.get("HF_TOKEN")
        if token and eval_results and eval_results.get("plot_path"):
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=token)
                api.upload_file(
                    path_or_fileobj=eval_results["plot_path"],
                    path_in_repo="checkpoints/test_predictions_vs_groundtruth.png",
                    repo_id=args.hf_repo,
                    repo_type="dataset",
                    token=token
                )
                print(f"[+] Synced validation evaluation plot to Hugging Face Hub: checkpoints/test_predictions_vs_groundtruth.png")
            except Exception as up_err:
                print(f"[!] Notice: Could not upload test plot to Hub ({up_err})")

    # 2. Custom Audio Evaluation (if provided)
    if args.audio_file and os.path.exists(args.audio_file):
        infer_on_audio_file(model, args.audio_file, args.output_dir, device)

    print("\n" + "=" * 82)
    print(" 🏁 TEST & INFERENCE EVALUATION COMPLETE!")
    print(f" Results Saved To: {args.output_dir}")
    print("=" * 82 + "\n")
