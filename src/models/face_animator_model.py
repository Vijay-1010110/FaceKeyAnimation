"""PyTorch Deep Learning Architecture: Speech-to-Facial Animation Neural Model.
Optimized for NVIDIA T4 Tensor Cores with Automatic Mixed Precision (AMP FP16).
Predicts 52 ARKit Blendshapes + 4 Dental Exposure Dynamics + 3D Head Orientation from Audio.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class ConvTemporalBlock(nn.Module):
    """1D Dilated Temporal Convolution Block with LayerNorm and Residual Connection."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm2 = nn.GroupNorm(8, channels)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (batch, channels, time)
        res = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.norm2(self.conv2(out))
        return self.act(out + res)


class SpeechToFaceAnimator(nn.Module):
    """Deep Neural Speech-to-Facial Motion and Dental Dynamics Generator.
    Maps temporal speech acoustics to fluid 52-ARKit blendshapes and dental exposure.
    """

    def __init__(
        self,
        audio_in_dim: int = 64,      # Audio input dimension (Mel / MFCC / spectral features)
        hidden_dim: int = 256,       # Hidden representation dimension
        num_lstm_layers: int = 2,    # Bi-directional LSTM depth
        num_blendshapes: int = 52,   # Standard ARKit blendshapes
        num_dental: int = 4,         # [upper_teeth, lower_teeth, inter_dental_gap, state]
        dropout: float = 0.15
    ):
        super().__init__()
        self.audio_in_dim = audio_in_dim
        self.hidden_dim = hidden_dim

        # 1. Acoustic Audio Projection
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 2. Multi-Scale Temporal Convolutions (Dilations: 1, 2, 4)
        self.tcn_blocks = nn.ModuleList([
            ConvTemporalBlock(hidden_dim, kernel_size=3, dilation=1, dropout=dropout),
            ConvTemporalBlock(hidden_dim, kernel_size=3, dilation=2, dropout=dropout),
            ConvTemporalBlock(hidden_dim, kernel_size=3, dilation=4, dropout=dropout)
        ])

        # 3. Bi-directional LSTM for long-range coarticulation and sentence context
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0
        )

        # 4. Multi-Head Output Decoders
        # Head A: 52 ARKit Blendshapes (Constrained to [0.0, 1.0] via Sigmoid)
        self.blendshape_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_blendshapes),
            nn.Sigmoid()
        )

        # Head B: Dental Exposure Dynamics (Continuous [0, 1] + state)
        self.dental_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, num_dental)
        )

        # Head C: 3D Head Orientation (Euler pitch, yaw, roll)
        self.head_pose_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 3)
        )

    def forward(
        self,
        audio_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        Args:
            audio_features: (batch_size, seq_len, audio_in_dim)
        Returns:
            pred_blendshapes: (batch_size, seq_len, 52)
            pred_dental     : (batch_size, seq_len, 4)
            pred_pose       : (batch_size, seq_len, 3)
        """
        # (B, T, D) -> Proj
        x = self.audio_proj(audio_features)

        # (B, T, D) -> (B, D, T) for Convolutions
        x_t = x.transpose(1, 2)
        for block in self.tcn_blocks:
            x_t = block(x_t)
        x = x_t.transpose(1, 2)

        # BiLSTM Context
        lstm_out, _ = self.bilstm(x)

        # Output Heads
        pred_blendshapes = self.blendshape_head(lstm_out)
        pred_dental = self.dental_head(lstm_out)
        pred_pose = self.head_pose_head(lstm_out)

        return pred_blendshapes, pred_dental, pred_pose


class AnimationCriterion(nn.Module):
    """Phonetically-Grounded Viseme & Articulation Composite Loss Function.
    
    Tackles the "open/close volume collapse" problem by:
      1. Viseme Channel Weighting: 4x weight on 28 mouth/speech visemes vs static face channels.
      2. Active-Phoneme Loss: Penalizes flat predictions heavily when target visemes are active (>0.1).
      3. Directional Cosine Loss: Forces matching of the phonetic mouth shape vector (pucker vs smile vs open).
      4. Velocity & Acceleration Loss: Forces snappy, distinct phonetic transitions and stops jitter.
      5. Dental Exposure Dynamics Loss: Aligns teeth reveal with open vowels and labiodentals.
    """

    def __init__(
        self,
        viseme_weight: float = 3.5,
        active_boost: float = 3.0,
        cosine_weight: float = 0.8,
        velocity_weight: float = 0.6,
        accel_weight: float = 0.3,
        dental_weight: float = 0.4
    ):
        super().__init__()
        self.huber = nn.SmoothL1Loss(beta=0.02, reduction="none")
        self.l1 = nn.L1Loss()
        self.viseme_weight = viseme_weight
        self.active_boost = active_boost
        self.cosine_weight = cosine_weight
        self.velocity_weight = velocity_weight
        self.accel_weight = accel_weight
        self.dental_weight = dental_weight

        # 52 ARKit Channel Weights: High priority for speech articulations
        weights = torch.ones(52, dtype=torch.float32) * 0.5  # default base weight for eyes/ears
        
        # Brows & Cheeks (Expressive & Inflections)
        for idx in [0, 7, 41, 42, 43, 44, 45, 46, 47, 48]:
            if idx < 52:
                weights[idx] = 1.5

        # Speech Articulation Channels (Jaw, Lips, Mouth, Tongue) -> 4.0x Priority
        # 14: jawForward, 15: jawLeft, 16: jawRight, 17: jawOpen
        # 18-40: mouthClose, mouthFunnel, mouthPucker, mouthSmile, mouthStretch, mouthRoll, etc.
        # 51: tongueOut
        speech_indices = list(range(14, 41)) + [51]
        for idx in speech_indices:
            if idx < 52:
                weights[idx] = 4.0

        self.register_buffer("channel_weights", weights.view(1, 1, 52))

    def forward(
        self,
        pred_bs: torch.Tensor,
        target_bs: torch.Tensor,
        pred_dental: torch.Tensor,
        target_dental: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes multi-component phonetic loss."""
        # 1. Base Element-wise Huber Loss
        raw_huber = self.huber(pred_bs, target_bs)

        # 2. Viseme Channel Weighting + Active-Phoneme Magnitude Scaling
        # When target has strong expression (e.g. pucker=0.8), scale loss up so it cannot be ignored
        active_scale = 1.0 + (self.active_boost * target_bs)
        weighted_loss = raw_huber * self.channel_weights * active_scale
        recon_loss = weighted_loss.mean()

        # 3. Directional Cosine Viseme Loss on Mouth Shapes (Indices 14..40)
        mouth_pred = pred_bs[:, :, 14:41]
        mouth_target = target_bs[:, :, 14:41]
        cos_sim = F.cosine_similarity(mouth_pred + 1e-6, mouth_target + 1e-6, dim=-1)
        cos_loss = (1.0 - cos_sim).mean()

        # 4. First-Order Velocity Loss (Snappy phoneme onsets/offsets)
        if pred_bs.shape[1] > 1:
            pred_vel = pred_bs[:, 1:, :] - pred_bs[:, :-1, :]
            target_vel = target_bs[:, 1:, :] - target_bs[:, :-1, :]
            vel_loss = self.l1(pred_vel, target_vel)
        else:
            vel_loss = torch.tensor(0.0, device=pred_bs.device)

        # 5. Second-Order Acceleration Loss (Zero jitter & crisp syllable stops)
        if pred_bs.shape[1] > 2:
            pred_acc = pred_vel[:, 1:, :] - pred_vel[:, :-1, :]
            target_acc = target_vel[:, 1:, :] - target_vel[:, :-1, :]
            acc_loss = self.l1(pred_acc, target_acc)
        else:
            acc_loss = torch.tensor(0.0, device=pred_bs.device)

        # 6. Dental Exposure Dynamics Loss
        dental_loss = F.smooth_l1_loss(pred_dental, target_dental, beta=0.02)

        # Total Composite Loss
        total_loss = (
            recon_loss
            + (self.cosine_weight * cos_loss)
            + (self.velocity_weight * vel_loss)
            + (self.accel_weight * acc_loss)
            + (self.dental_weight * dental_loss)
        )

        metrics = {
            "recon_loss": float(recon_loss.item()),
            "cos_loss": float(cos_loss.item()),
            "vel_loss": float(vel_loss.item()),
            "acc_loss": float(acc_loss.item()),
            "dental_loss": float(dental_loss.item()),
            "total_loss": float(total_loss.item())
        }
        return total_loss, metrics
