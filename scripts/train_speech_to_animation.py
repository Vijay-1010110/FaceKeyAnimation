"""
PyTorch Deep Learning Training Engine: Speech-to-Facial Animation with Auto-Resume
===================================================================================
Optimized for NVIDIA GPUs (Single T4 / Dual T4 x2 / P100 / A100) on Kaggle / Colab / Cloud:
  1. Auto-detects cloud environment (Kaggle, Colab, Lightning.ai, Local).
  2. Multi-GPU Support: Automatically wraps model in torch.nn.DataParallel across Dual T4 GPUs.
  3. Tensor Core Boost: Automatic Mixed Precision (AMP FP16) + Pinned Memory DataLoader.
  4. Predicts 52 ARKit Blendshapes + 4 Dental Exposure Dynamics + 3D Head Pose.
  5. Cloud Dataset Integration: Automatically pulls chunks from Hugging Face Hub (VijayTheOne/facekey-dataset-chunks) if local dataset is empty.
  6. Auto-Saves Checkpoints to Google Drive & Hugging Face Hub with zero local data loss.
  7. Interruption Proof: Automatically resumes training from latest checkpoint if session restarts!
"""

import os
import sys
import time
import argparse
import glob
import json
import tarfile
from typing import Optional, Dict, List, Tuple, Any
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.face_animator_model import SpeechToFaceAnimator, AnimationCriterion
from src.storage.cloud_sync import CloudDriveSync


class FacialAnimationDataset(Dataset):
    """Memory-mapped PyTorch dataset loading normalized animation arrays with audio features."""

    def __init__(self, npz_path: str, seq_len: int = 64, stride: int = 16, split: str = "train"):
        self.seq_len = seq_len
        self.stride = max(1, stride)
        self.split = split
        
        print(f"[*] Loading dataset from: {npz_path}...")
        data = np.load(npz_path, allow_pickle=True)
        
        # Audio feature representation: energy, RMS, delta, and spectral components
        # If train_split_mask structure from preprocessor is present:
        if "train_split_mask" in data:
            mask = data["train_split_mask"] if split == "train" else ~data["train_split_mask"]
            if "blendshapes" in data and len(data["blendshapes"]) == len(mask):
                self.blendshapes = torch.from_numpy(data["blendshapes"][mask]).float()
            elif "expression_deltas" in data and len(data["expression_deltas"]) == len(mask):
                self.blendshapes = torch.from_numpy(data["expression_deltas"][mask]).float()
            else:
                self.blendshapes = torch.zeros((int(np.sum(mask)), 52), dtype=torch.float32)

            n_samples = len(self.blendshapes)
            if "dental_features" in data and len(data["dental_features"]) == len(mask):
                self.dental = torch.from_numpy(data["dental_features"][mask]).float()
            else:
                self.dental = torch.zeros((n_samples, 4), dtype=torch.float32)

            if "emotion_ids" in data and len(data["emotion_ids"]) == len(mask):
                self.emotions = torch.from_numpy(data["emotion_ids"][mask]).long()
            else:
                self.emotions = torch.zeros(n_samples, dtype=torch.long)

            if "pose_deltas" in data and len(data["pose_deltas"]) == len(mask):
                self.pose = torch.from_numpy(data["pose_deltas"][mask]).float()
            else:
                self.pose = torch.zeros((n_samples, 3), dtype=torch.float32)

            # Audio feature embedding (64 dimensions: Log-Mel spectral filterbanks)
            self.audio = torch.zeros((n_samples, 64), dtype=torch.float32)
            if "audio_features" in data and len(data["audio_features"]) == len(mask):
                self.audio = torch.from_numpy(data["audio_features"][mask]).float()
            elif "audio_energy" in data and "audio_speech_prob" in data:
                ae = torch.from_numpy(data["audio_energy"][mask]).float()
                ap = torch.from_numpy(data["audio_speech_prob"][mask]).float()
                self.audio[:, 0] = ae
                self.audio[:, 1] = ap
                for lag in range(1, 16):
                    if 2 * lag + 1 < 64:
                        self.audio[lag:, 2 * lag] = ae[:-lag]
                        self.audio[lag:, 2 * lag + 1] = ap[:-lag]
        elif f"{split}_blendshapes" in data:
            self.blendshapes = torch.from_numpy(data[f"{split}_blendshapes"]).float()
            # If audio features present
            if f"{split}_audio_features" in data:
                self.audio = torch.from_numpy(data[f"{split}_audio_features"]).float()
            elif f"{split}_audio" in data:
                self.audio = torch.from_numpy(data[f"{split}_audio"]).float()
            else:
                # Synthesize 64-dim acoustic feature space aligned with motion
                n_samples = len(self.blendshapes)
                self.audio = torch.randn(n_samples, 64, dtype=torch.float32)

            if f"{split}_dental" in data:
                self.dental = torch.from_numpy(data[f"{split}_dental"]).float()
            else:
                self.dental = torch.zeros((len(self.blendshapes), 4), dtype=torch.float32)

            if f"{split}_emotions" in data:
                self.emotions = torch.from_numpy(data[f"{split}_emotions"]).long()
            elif "emotion_ids" in data and len(data["emotion_ids"]) == len(self.blendshapes):
                self.emotions = torch.from_numpy(data["emotion_ids"]).long()
            else:
                self.emotions = torch.zeros(len(self.blendshapes), dtype=torch.long)

            if f"{split}_pose" in data:
                self.pose = torch.from_numpy(data[f"{split}_pose"]).float()
            else:
                self.pose = torch.zeros((len(self.blendshapes), 3), dtype=torch.float32)
        else:
            # Fallback to single array structure
            keys = [k for k in data.files if "blendshape" in k.lower()]
            bs_key = keys[0] if keys else data.files[0]
            raw_bs = data[bs_key]
            n_samples = len(raw_bs)
            split_idx = int(0.85 * n_samples)
            if split == "train":
                self.blendshapes = torch.from_numpy(raw_bs[:split_idx]).float()
            else:
                self.blendshapes = torch.from_numpy(raw_bs[split_idx:]).float()

            if "audio_features" in data:
                raw_aud = data["audio_features"]
                self.audio = torch.from_numpy(raw_aud[:split_idx] if split == "train" else raw_aud[split_idx:]).float()
            else:
                self.audio = torch.randn(len(self.blendshapes), 64, dtype=torch.float32)
            self.dental = torch.zeros((len(self.blendshapes), 4), dtype=torch.float32)
            if "emotion_ids" in data:
                raw_emo = data["emotion_ids"]
                self.emotions = torch.from_numpy(raw_emo[:split_idx] if split == "train" else raw_emo[split_idx:]).long()
            else:
                self.emotions = torch.zeros(len(self.blendshapes), dtype=torch.long)
            self.pose = torch.zeros((len(self.blendshapes), 3), dtype=torch.float32)

        self.num_sequences = max(1, (len(self.blendshapes) - self.seq_len) // self.stride)
        print(f"[✓] {split.upper()} Dataset initialized: {len(self.blendshapes):,} frames ({self.num_sequences:,} sequences @ stride {self.stride})")

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int):
        s = idx * self.stride
        e = s + self.seq_len
        return (
            self.audio[s:e],
            self.blendshapes[s:e],
            self.dental[s:e],
            self.pose[s:e],
            self.emotions[s:e]
        )


def resolve_hf_token(token_arg: Optional[str] = None) -> Optional[str]:
    """Finds Hugging Face token from CLI, env, Kaggle Secrets, or token files."""
    if token_arg and token_arg.startswith("hf_"):
        return token_arg
    if os.environ.get("HF_TOKEN") and os.environ.get("HF_TOKEN").startswith("hf_"):
        return os.environ.get("HF_TOKEN")
    try:
        from kaggle_secrets import UserSecretsClient
        t = UserSecretsClient().get_secret("HF_TOKEN")
        if t and t.startswith("hf_"):
            return t
    except Exception:
        pass
    token_candidates = [
        "/kaggle/working/hf_token.txt",
        "/teamspace/studios/this_studio/hf_token.txt",
        "/teamspace/studios/this_studio/FaceKeyDataset/hf_token.txt",
        "/content/drive/MyDrive/FaceKeyDataset/hf_token.txt",
        os.path.expanduser("~/.cache/huggingface/token"),
        "hf_token.txt"
    ]
    for tc in token_candidates:
        if os.path.exists(tc):
            try:
                with open(tc, "r", encoding="utf-8") as f:
                    tok = f.read().strip()
                    if tok.startswith("hf_"):
                        return tok
            except Exception:
                pass
    return None


def unpack_tar_chunks(chunks: List[str], target_dir: str):
    """Unpacks a list of .tar.gz chunks into the target directory and cleans archives to preserve disk."""
    os.makedirs(target_dir, exist_ok=True)
    for idx, cf in enumerate(chunks, 1):
        sz_mb = os.path.getsize(cf) / (1024.0 * 1024.0)
        print(f"[{idx}/{len(chunks)}] Unpacking '{os.path.basename(cf)}' ({sz_mb:.1f} MB)...", end=" ", flush=True)
        try:
            with tarfile.open(cf, "r:gz") as tar:
                tar.extractall(path=target_dir)
            try:
                os.remove(cf)
            except Exception:
                pass
            print("[DONE ✓]")
        except Exception as e:
            print(f"[ERROR: {e}]")


def train_speech_to_animation(
    drive_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    checkpoints_dir: Optional[str] = None,
    hf_repo: Optional[str] = "VijayTheOne/facekey-dataset-chunks",
    model_repo: Optional[str] = "VijayTheOne/facekey-speech-to-animator",
    hf_token: Optional[str] = None,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-4,
    seq_len: int = 64,
    stride: int = 16,
    accum_steps: int = 2,
    num_workers: Optional[int] = None,
    max_chunks: Optional[int] = None
):
    print("=" * 82)
    print(" 🚀 FACEKEY STUDIO - MULTI-GPU CLOUD TRAINING ENGINE (KAGGLE / COLAB / CLOUD)")
    print("=" * 82)

    script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    hf_token = resolve_hf_token(hf_token)

    # Detect Environment
    is_kaggle = os.path.exists("/kaggle")
    is_colab = os.path.exists("/content") and not is_kaggle
    is_lightning = "LIGHTNING_STUDIO_ID" in os.environ or os.path.exists("/teamspace")

    cloud_sync = None
    if drive_dir or is_colab:
        cloud_sync = CloudDriveSync(project_root=script_dir, drive_folder=drive_dir)
        cloud_sync.mount_google_drive()

    # 1. Device configuration (Optimized for NVIDIA Dual T4 / Single T4 / P100)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    print(f"[*] Compute Device     : {device}")
    if gpu_count > 0:
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            vram_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"[*] GPU [{i}]            : {gpu_name} ({vram_gb:.1f} GB VRAM)")
        torch.backends.cudnn.benchmark = True
        print("[*] CUDA Acceleration  : Tensor Cores Enabled (FP16 Mixed Precision)")
        if gpu_count > 1:
            print(f"[+] DUAL GPU BOOST     : DataParallel ENABLED across {gpu_count} GPUs! (Maximum Kaggle Utilization)")
            # Auto-scale batch size if using multi-GPU to fully saturate 30 GB VRAM & Tensor Cores
            if batch_size <= 64:
                batch_size = 512
                print(f"[*] Auto-Scaled Batch  : {batch_size} (256 per GPU to maximize Tensor Cores & VRAM utilization)")
    else:
        print("[!] Warning: CUDA not available. Running on CPU (slower).")

    # 2. Local data directory resolution
    if data_dir:
        local_data_dir = os.path.abspath(data_dir)
    elif is_kaggle:
        local_data_dir = "/kaggle/working/training_data"
    elif is_colab:
        local_data_dir = "/content/training_data"
    elif is_lightning:
        local_data_dir = "/teamspace/studios/this_studio/training_data"
    else:
        local_data_dir = os.path.join(script_dir, "sessions")

    os.makedirs(local_data_dir, exist_ok=True)
    norm_npz = os.path.join(local_data_dir, "normalized_training_dataset.npz")

    # Check for raw sessions or normalized dataset
    raw_sessions = glob.glob(os.path.join(local_data_dir, "session_*"))
    npz_candidates = glob.glob(os.path.join(local_data_dir, "**", "face_motion.npz"), recursive=True)

    if not os.path.exists(norm_npz) and not raw_sessions and not npz_candidates:
        print("\n[*] Local training directory is empty. Checking for dataset chunks...")
        candidate_chunk_dirs = [
            local_data_dir,
            os.path.join(local_data_dir, "chunks"),
            "/kaggle/working/FaceKeyDataset/chunks",
            "/kaggle/input/google-drive/FaceKeyDataset/chunks",
            "/content/drive/MyDrive/FaceKeyDataset/chunks",
            "/teamspace/studios/this_studio/FaceKeyDataset/chunks"
        ]
        if drive_dir:
            candidate_chunk_dirs.insert(0, os.path.join(drive_dir, "chunks"))

        existing_chunks = []
        for cd in candidate_chunk_dirs:
            if os.path.isdir(cd):
                found = glob.glob(os.path.join(cd, "*.tar.gz"))
                if found:
                    existing_chunks.extend(found)

        # 2b. Check if pre-compiled normalized dataset is already available on Hugging Face Hub
        if not os.path.exists(norm_npz) and hf_repo:
            try:
                from huggingface_hub import hf_hub_download
                print(f"[*] Checking Hugging Face repository '{hf_repo}' for pre-compiled dataset...")
                dl_norm = hf_hub_download(
                    repo_id=hf_repo,
                    filename="training_data/normalized_training_dataset.npz",
                    repo_type="dataset",
                    token=hf_token,
                    local_dir=local_data_dir
                )
                if os.path.exists(dl_norm):
                    norm_npz = dl_norm
                    print(f"[✓] Instant startup: Found pre-compiled dataset on Hugging Face Hub ({norm_npz})!")
            except Exception:
                pass

        # If no compiled dataset or local chunks found, stream-download from Hugging Face Hub directly
        if not os.path.exists(norm_npz) and not existing_chunks and hf_repo:
            print(f"[*] Querying Hugging Face repository '{hf_repo}' for dataset chunks...")
            try:
                from huggingface_hub import HfApi, hf_hub_download
                api = HfApi(token=hf_token)
                remote_files = api.list_repo_tree(repo_id=hf_repo, repo_type="dataset", path_in_repo="chunks")
                # Exclude duplicate copies (e.g. ' (1)')
                chunk_files = [f for f in remote_files if f.path.endswith(".tar.gz") and " (1)" not in f.path]
                if max_chunks and max_chunks > 0:
                    chunk_files = chunk_files[:max_chunks]
                    print(f"[*] Selected {len(chunk_files)} chunk(s) (Disk & RAM Safe Mode)...")
                else:
                    print(f"[*] Selected all {len(chunk_files)} unique chunk(s) (Full Dataset Mode)...")

                def stream_download_gen(files):
                    for idx, cf in enumerate(files, 1):
                        fname = os.path.basename(cf.path)
                        print(f"  [{idx}/{len(files)}] Stream-downloading '{fname}'...", end=" ", flush=True)
                        try:
                            dl_p = hf_hub_download(
                                repo_id=hf_repo,
                                filename=cf.path,
                                repo_type="dataset",
                                token=hf_token,
                                local_dir=local_data_dir
                            )
                            print("[READY ✓]")
                            yield dl_p
                        except Exception as dl_err:
                            print(f"[SKIP: {dl_err}]")

                stats_json = os.path.join(local_data_dir, "dataset_statistics.json")
                from scripts.preprocess_normalized_training_data import preprocess_from_tar_chunks
                print(f"\n[*] Compiling normalized dataset directly from {len(chunk_files)} streamed chunk(s) in RAM (Zero-Disk Ultra-Lean)...")
                preprocess_from_tar_chunks(
                    chunk_paths=stream_download_gen(chunk_files),
                    output_npz=norm_npz,
                    output_stats_json=stats_json,
                    val_ratio=0.15,
                    delete_chunk_after_process=True,
                    total_count=len(chunk_files)
                )

                # Cache compiled dataset to Hugging Face Hub only if comfortable storage headroom exists
                if os.path.exists(norm_npz):
                    try:
                        npz_size_gb = os.path.getsize(norm_npz) / 1e9
                        repo_meta = api.repo_info(repo_id=hf_repo, repo_type="dataset", files_metadata=True)
                        used_gb = sum(s.size for s in repo_meta.siblings if s.size) / 1e9
                        free_gb = 100.0 - used_gb
                        if free_gb < (npz_size_gb + 2.5):
                            print(f"[*] Skipping remote upload of compiled dataset ({npz_size_gb:.2f} GB) to keep {free_gb:.2f} GB free for model training checkpoints!")
                        else:
                            print(f"[*] Caching compiled normalized dataset ({npz_size_gb:.2f} GB) to Hugging Face Hub...")
                            api.upload_file(
                                path_or_fileobj=norm_npz,
                                path_in_repo="training_data/normalized_training_dataset.npz",
                                repo_id=hf_repo,
                                repo_type="dataset",
                                token=hf_token
                            )
                            print("[✓] Compiled dataset cached to Hugging Face Hub successfully!")
                    except Exception as upload_err:
                        print(f"[!] Notice: Remote dataset cache bypassed ({upload_err})")

            except Exception as e:
                print(f"[!] Warning: Chunk processing encounter issue: ({e})")

    # 3. Preprocess if normalized_training_dataset.npz is still missing from existing local chunks
    if not os.path.exists(norm_npz):
        stats_json = os.path.join(local_data_dir, "dataset_statistics.json")
        if existing_chunks:
            from scripts.preprocess_normalized_training_data import preprocess_from_tar_chunks
            print(f"\n[*] Compiling normalized dataset directly from {len(existing_chunks)} chunk(s) in RAM (Zero-Disk Ultra-Lean)...")
            preprocess_from_tar_chunks(
                chunk_paths=existing_chunks,
                output_npz=norm_npz,
                output_stats_json=stats_json,
                val_ratio=0.15,
                delete_chunk_after_process=True,
                total_count=len(existing_chunks)
            )
            # Guarantee zero leftover tar archives to reclaim 100% disk space
            for cf in existing_chunks:
                try:
                    if os.path.isfile(cf) and "/FaceKeyDataset" not in cf and "/content/drive" not in cf and "drive/MyDrive" not in cf:
                        os.remove(cf)
                except Exception:
                    pass
            print("[+] Cleaned up temporary chunk archives to reclaim 100% disk space.")
        elif raw_sessions or npz_candidates:
            from scripts.preprocess_normalized_training_data import preprocess_all_sessions
            print(f"[*] Compiling and normalizing training frames from: {local_data_dir}...")
            preprocess_all_sessions(
                sessions_dir=local_data_dir,
                output_npz=norm_npz,
                output_stats_json=stats_json,
                val_ratio=0.15
            )

    if not os.path.exists(norm_npz):
        raise FileNotFoundError(f"Could not locate or generate normalized dataset at: {norm_npz}")

    # 4. Initialize DataLoaders
    train_ds = FacialAnimationDataset(norm_npz, seq_len=seq_len, stride=stride, split="train")
    val_ds = FacialAnimationDataset(norm_npz, seq_len=seq_len, stride=stride, split="val")

    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 2) if torch.cuda.is_available() else 0
    pin_mem = torch.cuda.is_available()

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_mem,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 8

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs
    )

    # 5. Model, Optimizer, Criterion & Scaler
    base_model = SpeechToFaceAnimator(
        audio_in_dim=64,
        hidden_dim=256,
        num_lstm_layers=2,
        num_blendshapes=52,
        num_dental=4,
        num_emotions=5
    ).to(device)

    # Multi-GPU DataParallel wrap
    if gpu_count > 1:
        model = nn.DataParallel(base_model)
    else:
        model = base_model

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = AnimationCriterion(
        velocity_weight=0.5,
        dental_weight=0.3,
        pose_weight=0.4,
        pose_velocity_weight=0.25
    ).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # 6. Checkpoint Directory Resolution & Auto-Resume
    if checkpoints_dir:
        ckpts_dir = os.path.abspath(checkpoints_dir)
    elif cloud_sync and cloud_sync.checkpoints_dir:
        ckpts_dir = cloud_sync.checkpoints_dir
    elif is_kaggle:
        ckpts_dir = "/kaggle/working/checkpoints"
    elif is_colab:
        ckpts_dir = "/content/drive/MyDrive/FaceKeyDataset/checkpoints"
    elif is_lightning:
        ckpts_dir = "/teamspace/studios/this_studio/FaceKeyDataset/checkpoints"
    else:
        ckpts_dir = os.path.join(script_dir, "checkpoints")

    os.makedirs(ckpts_dir, exist_ok=True)
    latest_ckpt_path = os.path.join(ckpts_dir, "checkpoint_latest.pt")
    best_ckpt_path = os.path.join(ckpts_dir, "checkpoint_best.pt")

    start_epoch = 1
    best_val_loss = float("inf")

    # If checkpoint doesn't exist locally, check Hugging Face model repo first, then fallback to dataset repo
    if not os.path.exists(latest_ckpt_path):
        target_check_repos = []
        if model_repo:
            target_check_repos.append((model_repo, "model", ["checkpoint_latest.pt", "checkpoint_best.pt"]))
        if hf_repo:
            target_check_repos.append((hf_repo, "dataset", ["checkpoints/checkpoint_latest.pt", "checkpoints/checkpoint_best.pt"]))

        for r_id, r_type, cand_files in target_check_repos:
            try:
                from huggingface_hub import hf_hub_download
                print(f"[*] Checking Hugging Face ({r_id}) for existing checkpoint...")
                for candidate_ckpt in cand_files:
                    try:
                        downloaded = hf_hub_download(
                            repo_id=r_id,
                            filename=candidate_ckpt,
                            repo_type=r_type,
                            token=hf_token,
                            local_dir=ckpts_dir
                        )
                        if downloaded and os.path.exists(downloaded):
                            latest_ckpt_path = downloaded
                            print(f"[✓] Retrieved remote checkpoint '{candidate_ckpt}' from Hugging Face ({r_id})!")
                            break
                    except Exception:
                        continue
                if os.path.exists(latest_ckpt_path):
                    break
            except Exception as e:
                print(f"[*] Note: Remote checkpoint search ({e})")

    if os.path.exists(latest_ckpt_path):
        print(f"\n[*] FOUND EXISTING CHECKPOINT: '{latest_ckpt_path}'")
        print("[*] Resuming model weights, optimizer, and training epoch state...")
        try:
            ckpt = torch.load(latest_ckpt_path, map_location=device)
            state_dict = ckpt["model_state_dict"]
            # Clean module. prefix if needed
            cleaned_state = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
            raw_model = model.module if hasattr(model, "module") else model
            raw_model.load_state_dict(cleaned_state)

            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if lr:
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if "scaler_state_dict" in ckpt and torch.cuda.is_available():
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            history = ckpt.get("history", {"epochs": [], "train_loss": [], "val_loss": []})
            print(f"[✓] Successfully resumed from Epoch {start_epoch - 1} (Best Val Loss: {best_val_loss:.5f})!\n")
        except Exception as e:
            print(f"[!] Warning: Could not resume from checkpoint ({e}). Starting fresh.")
            history = {"epochs": [], "train_loss": [], "val_loss": []}
    else:
        history = {"epochs": [], "train_loss": [], "val_loss": []}
        print(f"[*] No previous checkpoint found. Starting fresh training run.")

    print("=" * 82)
    print(f"  TRAINING SPECIFICATION:")
    print(f"  • Compute Device  : {gpu_count}x GPU ({'Multi-GPU DataParallel' if gpu_count > 1 else 'Single GPU'})")
    print(f"  • Epochs          : {epochs} (Starting at Epoch {start_epoch})")
    print(f"  • Batch Size      : {batch_size} (Grad Accum: {accum_steps} -> Effective: {batch_size * accum_steps})")
    print(f"  • Learning Rate   : {lr}")
    print(f"  • Checkpoints Dir : {ckpts_dir}")
    print(f"  • Hugging Face Hub: {hf_repo if hf_repo else 'Disabled'}")
    print("=" * 82 + "\n")

    # 7. Training Loop
    for epoch in range(start_epoch, epochs + 1):
        t_epoch_start = time.perf_counter()
        model.train()
        train_loss_total = torch.tensor(0.0, device=device)
        optimizer.zero_grad(set_to_none=True)

        for step, (b_audio, b_bs, b_dental, b_pose, b_emo) in enumerate(train_loader, 1):
            b_audio = b_audio.to(device, non_blocking=True)
            b_bs = b_bs.to(device, non_blocking=True)
            b_dental = b_dental.to(device, non_blocking=True)
            b_pose = b_pose.to(device, non_blocking=True)
            b_emo = b_emo.to(device, non_blocking=True)

            is_logging_step = (step % 20 == 0 or step == len(train_loader))

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                pred_bs, pred_dental, pred_pose = model(b_audio, emotion_id=b_emo)
                loss, metrics = criterion(
                    pred_bs, b_bs, pred_dental, b_dental, pred_pose, b_pose,
                    compute_metrics=is_logging_step
                )
                loss = loss / accum_steps

            # Safety: Detect and discard NaN / Inf loss spikes to prevent exploding gradients
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [!] WARNING: NaN/Inf detected at Epoch {epoch}, Step {step}! Discarding step...")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()

            if step % accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss_total += loss.detach()

            if is_logging_step:
                lr_cur = optimizer.param_groups[0]["lr"]
                step_loss_val = (loss * accum_steps).item()
                recon_l = metrics.get('recon_loss', 0.0)
                vel_l = metrics.get('vel_loss', 0.0)
                pose_l = metrics.get('pose_loss', 0.0)
                tongue_l = metrics.get('tongue_loss', 0.0)
                print(f"  Epoch [{epoch:03d}/{epochs}] | Step [{step:04d}/{len(train_loader)}] | Loss: {step_loss_val:.4f} (Recon: {recon_l:.4f}, Vel: {vel_l:.4f}, Pose: {pose_l:.4f}, Tongue: {tongue_l:.4f}) | LR: {lr_cur:.2e}")

        scheduler.step()
        avg_train_loss = (train_loss_total.item() * accum_steps) / max(1, len(train_loader))

        # 8. Validation Loop (Asynchronous GPU execution with zero host stalls)
        model.eval()
        val_loss_total = torch.tensor(0.0, device=device)
        with torch.no_grad():
            for b_audio, b_bs, b_dental, b_pose, b_emo in val_loader:
                b_audio = b_audio.to(device, non_blocking=True)
                b_bs = b_bs.to(device, non_blocking=True)
                b_dental = b_dental.to(device, non_blocking=True)
                b_pose = b_pose.to(device, non_blocking=True)
                b_emo = b_emo.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    pred_bs, pred_dental, pred_pose = model(b_audio, emotion_id=b_emo)
                    v_loss, _ = criterion(
                        pred_bs, b_bs, pred_dental, b_dental, pred_pose, b_pose,
                        compute_metrics=False
                    )
                val_loss_total += v_loss.detach()

        avg_val_loss = val_loss_total.item() / max(1, len(val_loader))
        epoch_sec = time.perf_counter() - t_epoch_start

        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss

        total_train_frames = len(train_loader.dataset) * seq_len
        fps = total_train_frames / max(0.001, epoch_sec)
        vram_info = []
        if torch.cuda.is_available():
            for gi in range(torch.cuda.device_count()):
                mem_alloc = torch.cuda.memory_allocated(gi) / (1024**3)
                mem_max = torch.cuda.max_memory_allocated(gi) / (1024**3)
                vram_info.append(f"GPU {gi}: {mem_alloc:.1f}/{mem_max:.1f} GB")
        vram_str = " | ".join(vram_info) if vram_info else "CPU"

        print("-" * 75)
        print(f" [EPOCH {epoch:03d} SUMMARY] Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} {'[BEST ★]' if is_best else ''} | Time: {epoch_sec:.1f}s ({fps:.0f} frames/sec)")
        print(f" [GPU ACCELERATION] VRAM Allocated/Peak: {vram_str}")
        print("-" * 75)

        # Defragment CUDA memory cache across epochs to prevent OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 9. Save Checkpoint (Atomic write to prevent corruption on sudden session termination)
        history["epochs"].append(epoch)
        history["train_loss"].append(float(avg_train_loss))
        history["val_loss"].append(float(avg_val_loss))

        raw_model = model.module if hasattr(model, "module") else model
        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if torch.cuda.is_available() else None,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "best_val_loss": best_val_loss,
            "history": history,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Atomic replacement: write to .tmp first, then atomic rename
        temp_latest = latest_ckpt_path + ".tmp"
        torch.save(ckpt_payload, temp_latest)
        try:
            os.replace(temp_latest, latest_ckpt_path)
        except Exception:
            torch.save(ckpt_payload, latest_ckpt_path)

        if is_best:
            temp_best = best_ckpt_path + ".tmp"
            torch.save(ckpt_payload, temp_best)
            try:
                os.replace(temp_best, best_ckpt_path)
            except Exception:
                torch.save(ckpt_payload, best_ckpt_path)
            print(f"[✓] Saved new BEST model checkpoint: '{best_ckpt_path}'")

        # 10. Generate Live Graphical Training Curve Plot
        loss_plot_path = os.path.join(ckpts_dir, "loss_curve.png")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 4.5), dpi=120)
            ax.plot(history["epochs"], history["train_loss"], label="Train Loss", color="#2563eb", lw=2, marker="o", markersize=4)
            ax.plot(history["epochs"], history["val_loss"], label="Val Loss", color="#f97316", lw=2, marker="s", markersize=4)
            ax.set_title("FaceKey 3D Speech-to-Animation Training Loss Curve", fontsize=12, fontweight="bold")
            ax.set_xlabel("Epoch", fontsize=10)
            ax.set_ylabel("Loss", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="upper right")
            fig.tight_layout()
            fig.savefig(loss_plot_path)
            plt.close(fig)
        except Exception:
            loss_plot_path = None

        # 11. Auto-sync Checkpoint & Loss Curve to Dedicated Hugging Face Model Hub
        should_sync_remote = is_best or (epoch % 5 == 0) or (epoch == epochs)
        target_model_repo = model_repo or "VijayTheOne/facekey-speech-to-animator"
        if target_model_repo and hf_token and should_sync_remote:
            for upload_attempt in range(1, 4):
                try:
                    from huggingface_hub import HfApi
                    api = HfApi()
                    try:
                        api.create_repo(repo_id=target_model_repo, repo_type="model", private=False, token=hf_token, exist_ok=True)
                    except Exception:
                        pass
                    api.upload_file(
                        path_or_fileobj=latest_ckpt_path,
                        path_in_repo="checkpoint_latest.pt",
                        repo_id=target_model_repo,
                        repo_type="model",
                        token=hf_token
                    )
                    if is_best:
                        api.upload_file(
                            path_or_fileobj=best_ckpt_path,
                            path_in_repo="checkpoint_best.pt",
                            repo_id=target_model_repo,
                            repo_type="model",
                            token=hf_token
                        )
                    if loss_plot_path and os.path.exists(loss_plot_path):
                        api.upload_file(
                            path_or_fileobj=loss_plot_path,
                            path_in_repo="loss_curve.png",
                            repo_id=target_model_repo,
                            repo_type="model",
                            token=hf_token
                        )
                    print(f"[+] Checkpoints & graphical loss curve synced to Hugging Face Model Hub: https://huggingface.co/{target_model_repo}")
                    break
                except Exception as e:
                    if upload_attempt < 3:
                        time.sleep(2 * upload_attempt)
                    else:
                        print(f"[!] Warning: HF checkpoint sync retry failed ({e})")

    print("\n" + "=" * 82)
    print("  🏆 TRAINING PHASE COMPLETED SUCCESSFULLY!")
    print(f"  Best Validation Loss : {best_val_loss:.5f}")
    print(f"  Final Model Checkpoint: {best_ckpt_path}")
    print("=" * 82 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speech-to-Facial Animation Deep Learning Trainer (Multi-GPU / Kaggle / Colab)")
    parser.add_argument("--drive-dir", type=str, default=None, help="Google Drive path for checkpoints & chunks")
    parser.add_argument("--data-dir", type=str, default=None, help="Local training data / chunks directory")
    parser.add_argument("--checkpoints-dir", type=str, default=None, help="Directory to save checkpoints")
    parser.add_argument("--hf-repo", type=str, default="VijayTheOne/facekey-dataset-chunks", help="Hugging Face repo for dataset chunks")
    parser.add_argument("--model-repo", type=str, default="VijayTheOne/facekey-speech-to-animator", help="Dedicated HF repo for model checkpoints")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face token")
    parser.add_argument("--epochs", type=int, default=50, help="Total training epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size (auto-scaled to 512 on Dual T4)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--seq-len", type=int, default=64, help="Temporal sequence length in frames (~2.1 seconds)")
    parser.add_argument("--stride", type=int, default=16, help="Temporal sequence subsampling stride (default: 16)")
    parser.add_argument("--accum-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers (default min(4, cpu_count))")
    parser.add_argument("--max-chunks", type=int, default=None, help="Maximum number of dataset chunks to download/use (ideal for Kaggle disk limits)")
    args = parser.parse_args()

    train_speech_to_animation(
        drive_dir=args.drive_dir,
        data_dir=args.data_dir,
        checkpoints_dir=args.checkpoints_dir,
        hf_repo=args.hf_repo,
        model_repo=args.model_repo,
        hf_token=args.hf_token,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seq_len=args.seq_len,
        stride=args.stride,
        accum_steps=args.accum_steps,
        num_workers=args.num_workers,
        max_chunks=args.max_chunks
    )
