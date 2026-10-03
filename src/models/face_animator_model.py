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
    """Composite loss function with Huber regression, velocity smoothness, and dental loss."""

    def __init__(self, velocity_weight: float = 0.5, dental_weight: float = 0.3):
        super().__init__()
        self.huber = nn.SmoothL1Loss(beta=0.02)
        self.mse = nn.MSELoss()
        self.velocity_weight = velocity_weight
        self.dental_weight = dental_weight

    def forward(
        self,
        pred_bs: torch.Tensor,
        target_bs: torch.Tensor,
        pred_dental: torch.Tensor,
        target_dental: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 1. Reconstruction Loss on Blendshapes
        recon_loss = self.huber(pred_bs, target_bs)

        # 2. Velocity / Smoothness Loss (Inter-frame delta consistency)
        if pred_bs.shape[1] > 1:
            pred_vel = pred_bs[:, 1:, :] - pred_bs[:, :-1, :]
            target_vel = target_bs[:, 1:, :] - target_bs[:, :-1, :]
            vel_loss = self.mse(pred_vel, target_vel)
        else:
            vel_loss = torch.tensor(0.0, device=pred_bs.device)

        # 3. Dental Exposure Dynamics Loss
        dental_loss = self.huber(pred_dental, target_dental)

        # Total Loss
        total_loss = recon_loss + (self.velocity_weight * vel_loss) + (self.dental_weight * dental_loss)

        metrics = {
            "recon_loss": float(recon_loss.item()),
            "vel_loss": float(vel_loss.item()),
            "dental_loss": float(dental_loss.item()),
            "total_loss": float(total_loss.item())
        }
        return total_loss, metrics
