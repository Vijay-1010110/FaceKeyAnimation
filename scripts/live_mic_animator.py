"""FaceKey Studio: Real-Time Live Microphone Face Skeleton Animator.
Captures live voice from the PC microphone, runs GPU neural inference (Epoch 10 SpeechToFaceAnimator),
and renders an interactive 3D face wireframe skeleton articulating synchronously to your voice in real time.

Controls:
  - Speak into your microphone to animate the skeleton
  - Mouse Drag: Rotate face skeleton in 3D
  - '+' / '=' : Boost mouth articulation
  - '-' / '_' : Reduce mouth articulation
  - 'r'       : Reset 3D camera rotation
  - 'q' / ESC : Quit application
"""

import os
import sys
import math
import glob
import queue
import argparse
import numpy as np
import cv2
import torch
import sounddevice as sd

# Enable UTF-8 encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure root workspace in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.face_animator_model import SpeechToFaceAnimator
from src.core.speaker_attribution import VoiceActivityDetector
from scripts.render_face_skeleton_reaction import (
    deform_face_skeleton,
    draw_skeleton_frame
)

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


class LiveFaceAnimatorApp:
    def __init__(
        self,
        checkpoint_path: str,
        sample_rate: int = 16000,
        fps: int = 30,
        device_str: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.sample_rate = sample_rate
        self.fps = fps
        self.samples_per_frame = int(round(sample_rate / fps))
        self.device = torch.device(device_str)

        print("=" * 80)
        print(" 🎙️ FACEKEY STUDIO: REAL-TIME LIVE MICROPHONE FACE SKELETON ANIMATOR")
        print(f" [*] Compute Device : {self.device}")
        print(f" [*] Audio Rate     : {self.sample_rate} Hz ({self.samples_per_frame} samples/frame @ {self.fps} FPS)")
        print(f" [*] Checkpoint     : {checkpoint_path}")
        print("=" * 80)

        # Load Neural Model
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model = SpeechToFaceAnimator(
            audio_in_dim=64,
            hidden_dim=256,
            num_lstm_layers=2,
            num_blendshapes=52,
            num_dental=4
        ).to(self.device)

        st = ckpt.get("model_state_dict", ckpt)
        cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in st.items()}
        self.model.load_state_dict(cleaned)
        self.model.eval()
        print("[OK] PyTorch Neural Animator loaded successfully!")

        # VAD & Acoustic Feature Tracker
        self.vad = VoiceActivityDetector(sample_rate=sample_rate)
        self.audio_queue = queue.Queue(maxsize=100)

        # 16-frame lag rolling buffer for 64-dim acoustic lag tensor
        self.lag_history_ae = [0.0] * 16
        self.lag_history_ap = [0.0] * 16

        # Dynamic Calibration State
        self.smooth_drive = 0.0
        self.articulation_boost = 1.2
        self.peak_energy = 0.05
        self.frame_idx = 0

        # Mouse 3D Orbit State
        self.rot_x = 0.0
        self.rot_y = 0.0
        self.is_dragging = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0

    def audio_callback(self, indata, frames, time_info, status):
        """Sounddevice streaming callback."""
        if status:
            pass
        # indata is shape (frames, channels)
        mono = indata[:, 0].copy()
        try:
            self.audio_queue.put_nowait(mono)
        except queue.Full:
            pass

    def on_mouse(self, event, x, y, flags, param):
        """OpenCV mouse callback for interactive 3D rotation."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_dragging = True
            self.last_mouse_x = x
            self.last_mouse_y = y
        elif event == cv2.EVENT_LBUTTONUP:
            self.is_dragging = False
        elif event == cv2.EVENT_MOUSEMOVE and self.is_dragging:
            dx = x - self.last_mouse_x
            dy = y - self.last_mouse_y
            self.rot_y += dx * 0.008
            self.rot_x += dy * 0.008
            self.rot_x = max(-0.6, min(0.6, self.rot_x))
            self.last_mouse_x = x
            self.last_mouse_y = y

    def run(self, input_device: int = None):
        """Main real-time loop."""
        window_name = "FaceKey Live Microphone 3D Skeleton (Press Q to quit)"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, self.on_mouse)

        print("[*] Starting microphone audio stream...")
        stream = sd.InputStream(
            device=input_device,
            samplerate=self.sample_rate,
            channels=1,
            blocksize=self.samples_per_frame,
            dtype="float32",
            callback=self.audio_callback
        )

        width, height = 720, 720
        last_skeleton = None
        current_energy = 0.0

        with stream:
            print("[OK] Microphone active! Speak now. Press 'q' in the window to quit.")
            while True:
                # 1. Process latest audio chunk
                try:
                    chunk = self.audio_queue.get(timeout=0.04)
                    # Drain any backlog to ensure zero latency
                    while not self.audio_queue.empty():
                        chunk = self.audio_queue.get_nowait()
                except queue.Empty:
                    chunk = np.zeros(self.samples_per_frame, dtype=np.float32)

                # 2. Compute Voice Activity & RMS Energy
                t = self.frame_idx / self.fps
                vad_res = self.vad.process_chunk(chunk, t)
                ae = float(vad_res.energy_rms)
                ap = float(vad_res.vad_confidence)
                current_energy = ae

                # Update adaptive noise / peak energy
                if ae > self.peak_energy:
                    self.peak_energy = 0.92 * self.peak_energy + 0.08 * ae
                else:
                    self.peak_energy = max(0.02, 0.998 * self.peak_energy)

                # Update Lag Buffer
                self.lag_history_ae.insert(0, ae)
                self.lag_history_ae.pop()
                self.lag_history_ap.insert(0, ap)
                self.lag_history_ap.pop()

                # 3. Construct 64-dim lag acoustic features
                feat = np.zeros((1, 1, 64), dtype=np.float32)
                for lag in range(16):
                    if 2 * lag + 1 < 64:
                        feat[0, 0, 2 * lag] = self.lag_history_ae[lag]
                        feat[0, 0, 2 * lag + 1] = self.lag_history_ap[lag]

                # 4. Neural Network Inference
                inp_tensor = torch.from_numpy(feat).to(self.device)
                with torch.no_grad():
                    with torch.amp.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                        pred_bs, pred_dental, pred_pose = self.model(inp_tensor)

                bs_np = pred_bs.squeeze().cpu().numpy()
                dental_np = pred_dental.squeeze().cpu().numpy()
                pose_np = pred_pose.squeeze().cpu().numpy()

                bs_dict = {STANDARD_BS_NAMES[j]: bs_np[j] for j in range(min(52, len(STANDARD_BS_NAMES)))}

                # 5. Calibrate Speech Drive Envelope
                raw_jaw = float(bs_dict.get("jawOpen", 0.0))
                ae_norm = min(1.0, ae / max(1e-4, self.peak_energy * 0.85))
                raw_drive = 0.65 * ae_norm + 0.35 * min(1.0, raw_jaw * 15.0)

                # Attack / Release Envelope
                alpha = 0.70 if raw_drive > self.smooth_drive else 0.25
                self.smooth_drive = alpha * raw_drive + (1.0 - alpha) * self.smooth_drive

                # 6. Deform 3D Skeleton
                skeleton = deform_face_skeleton(
                    blendshapes=bs_dict,
                    dental=dental_np,
                    head_pose=pose_np,
                    speech_drive=self.smooth_drive * self.articulation_boost,
                    audio_energy=ae
                )
                last_skeleton = skeleton

                # 7. Render Skeleton Canvas
                canvas = draw_skeleton_frame(
                    skeleton=skeleton,
                    audio_energy=ae,
                    frame_idx=self.frame_idx,
                    total_frames=9999,
                    fps=self.fps,
                    width=width,
                    height=height
                )

                # Extra HUD Info
                cv2.putText(
                    canvas,
                    f"ARTICULATION BOOST: {self.articulation_boost:.1f}x  [+/-]",
                    (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (240, 200, 80),
                    1,
                    cv2.LINE_AA
                )
                cv2.putText(
                    canvas,
                    "DRAG MOUSE: 3D Rotate | R: Reset | Q: Quit",
                    (30, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (140, 150, 160),
                    1,
                    cv2.LINE_AA
                )

                cv2.imshow(window_name, canvas)
                self.frame_idx += 1

                # Handle Keyboard Input
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), ord('Q'), 27]:  # 'q' or ESC
                    print("[*] Quitting live microphone session...")
                    break
                elif key in [ord('+'), ord('=')]:
                    self.articulation_boost = min(3.0, self.articulation_boost + 0.2)
                    print(f"[*] Articulation Boost: {self.articulation_boost:.1f}x")
                elif key in [ord('-'), ord('_')]:
                    self.articulation_boost = max(0.4, self.articulation_boost - 0.2)
                    print(f"[*] Articulation Boost: {self.articulation_boost:.1f}x")
                elif key in [ord('r'), ord('R')]:
                    self.rot_x = 0.0
                    self.rot_y = 0.0
                    print("[*] 3D Camera reset to neutral orientation.")

        cv2.destroyAllWindows()
        print("[OK] Live microphone session ended cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Live Microphone Face Skeleton Animator")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint_latest.pt")
    parser.add_argument("--device-idx", type=int, default=None, help="Audio input device index")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    args = parser.parse_args()

    # Resolve Checkpoint
    ckpt_path = args.checkpoint
    if not ckpt_path:
        candidates = [
            "test_results_epoch10/checkpoint_latest.pt",
            "checkpoints/checkpoint_latest.pt",
            "checkpoints/checkpoint_best.pt"
        ]
        snap_ckpts = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/**/checkpoint_latest.pt"), recursive=True)
        if snap_ckpts:
            candidates.insert(0, snap_ckpts[0])

        for c in candidates:
            if os.path.exists(c):
                ckpt_path = c
                break

    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError("Could not find trained model checkpoint.")

    app = LiveFaceAnimatorApp(
        checkpoint_path=ckpt_path,
        fps=args.fps
    )
    app.run(input_device=args.device_idx)
