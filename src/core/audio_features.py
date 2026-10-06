"""Audio Feature Extraction: 64-band Log-Mel Filterbank and Formant Representation.
Provides phonetically-grounded acoustic representations matching the SpeechToFaceAnimator
neural network's 64-dimensional audio input space.
Pure NumPy implementation: ultra-fast, zero extra dependencies, fully portable across
local machines, cloud backends (Colab/Kaggle), and real-time streaming pipelines.
"""

from typing import Optional, Union, Sequence
import numpy as np


def hz_to_mel(hz: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert Frequency in Hz to the Mel acoustic scale."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert Mel scale back to Frequency in Hz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


class LogMelFilterbankExtractor:
    """Computes standard 64-band Log-Mel filterbank energies from raw audio samples.
    
    Phonetic Frequency Distribution across 64 Bins (at 16 kHz):
      - Bins 0 - 15  (50 - 700 Hz)   : Fundamental pitch F0, nasal resonance, low F1 vowels (/u/, /o/, /m/, /b/)
      - Bins 16 - 35 (700 - 2200 Hz)  : Open vowel F1 & back vowel F2 (/a/, /ae/, /aw/)
      - Bins 36 - 50 (2200 - 4000 Hz) : Front vowel F2/F3 (/i/, /e/, lip spreading & smile)
      - Bins 51 - 63 (4000 - 8000 Hz) : Dental/alveolar fricatives & sibilants (/s/, /z/, /sh/, /t/)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 64,
        n_fft: int = 512,
        f_min: float = 60.0,
        f_max: Optional[float] = None
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else float(sample_rate / 2.0)
        self.filterbank = self._build_mel_filterbank()
        self.window = np.hanning(self.n_fft).astype(np.float32)

    def _build_mel_filterbank(self) -> np.ndarray:
        """Construct triangular overlapping Mel filterbank matrix of shape (n_mels, n_fft // 2 + 1)."""
        mel_min = hz_to_mel(self.f_min)
        mel_max = hz_to_mel(self.f_max)
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)

        n_freqs = self.n_fft // 2 + 1
        filterbank = np.zeros((self.n_mels, n_freqs), dtype=np.float32)

        for m in range(1, self.n_mels + 1):
            left = bin_points[m - 1]
            center = bin_points[m]
            right = bin_points[m + 1]

            for k in range(left, center):
                if center > left and k < n_freqs:
                    filterbank[m - 1, k] = (k - left) / (center - left)
            for k in range(center, right):
                if right > center and k < n_freqs:
                    filterbank[m - 1, k] = (right - k) / (right - center)

        return filterbank

    def extract_frame(self, audio_samples: Optional[np.ndarray]) -> np.ndarray:
        """Extracts 64-band log-Mel energy vector from a single audio slice (e.g. 1 video frame).
        
        Args:
            audio_samples: 1D array of audio samples (float32, [-1.0, 1.0]).
        Returns:
            np.ndarray of shape (64,), values scaled to [0.0, 1.0].
        """
        if audio_samples is None or len(audio_samples) == 0:
            return np.zeros(self.n_mels, dtype=np.float32)

        samples = np.asarray(audio_samples, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        # Pad or center slice to n_fft length
        if len(samples) < self.n_fft:
            pad_left = (self.n_fft - len(samples)) // 2
            pad_right = self.n_fft - len(samples) - pad_left
            samples = np.pad(samples, (pad_left, pad_right), mode="constant")
        elif len(samples) > self.n_fft:
            start = (len(samples) - self.n_fft) // 2
            samples = samples[start:start + self.n_fft]

        # Apply Hanning window
        windowed = samples * self.window

        # Real FFT Power Spectrum
        spec = np.abs(np.fft.rfft(windowed, n=self.n_fft))
        power = (spec ** 2) / float(self.n_fft)

        # Project through triangular Mel filterbank
        mel_energies = np.dot(self.filterbank, power)

        # Log-dynamic compression: log1p ensures quiet noise -> 0.0 and speech -> smooth curve
        log_mel = np.log1p(1e4 * mel_energies)

        # Scaled to stable [0.0, 1.0] range
        scaled = np.clip(log_mel / 10.0, 0.0, 1.0).astype(np.float32)
        return scaled

    def extract_sequence(
        self,
        audio_samples: Optional[np.ndarray],
        sample_rate: Optional[int] = None,
        fps: float = 30.0,
        num_frames: Optional[int] = None,
        timestamps: Optional[Sequence[float]] = None
    ) -> np.ndarray:
        """Extracts continuous sequence of 64-band log-Mel feature vectors aligned with video frames.
        
        Args:
            audio_samples: Complete 1D or 2D audio waveform array.
            sample_rate: Sample rate in Hz (defaults to self.sample_rate).
            fps: Video / animation frame rate.
            num_frames: Exact count of output frames required.
            timestamps: Explicit timestamps in seconds for each frame.
        Returns:
            np.ndarray of shape (N, 64) with float32 values in [0.0, 1.0].
        """
        sr = sample_rate or self.sample_rate

        if audio_samples is None or len(audio_samples) == 0:
            count = num_frames or (len(timestamps) if timestamps is not None else 1)
            return np.zeros((count, self.n_mels), dtype=np.float32)

        samples = np.asarray(audio_samples, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        # Peak normalization
        peak = float(np.max(np.abs(samples)))
        if peak > 1e-4:
            samples = samples / peak

        if timestamps is not None:
            n_frames = len(timestamps)
            time_points = np.asarray(timestamps, dtype=np.float32)
        elif num_frames is not None:
            n_frames = num_frames
            time_points = np.arange(n_frames, dtype=np.float32) / fps
        else:
            n_frames = int(round(len(samples) / (sr / fps)))
            time_points = np.arange(n_frames, dtype=np.float32) / fps

        half_w = self.n_fft // 2
        feats = np.zeros((n_frames, self.n_mels), dtype=np.float32)

        for i in range(n_frames):
            center = int(round(time_points[i] * sr))
            s_start = center - half_w
            s_end = center + half_w

            if s_start < 0:
                chunk = samples[:max(0, s_end)]
                pad_l = abs(s_start)
                chunk = np.pad(chunk, (pad_l, 0), mode="constant")
            elif s_end > len(samples):
                chunk = samples[min(len(samples), s_start):]
                pad_r = s_end - len(samples)
                chunk = np.pad(chunk, (0, pad_r), mode="constant")
            else:
                chunk = samples[s_start:s_end]

            if len(chunk) != self.n_fft:
                chunk = np.pad(chunk, (0, max(0, self.n_fft - len(chunk))), mode="constant")[:self.n_fft]

            windowed = chunk * self.window
            spec = np.abs(np.fft.rfft(windowed, n=self.n_fft))
            power = (spec ** 2) / float(self.n_fft)
            mel_energies = np.dot(self.filterbank, power)
            log_mel = np.log1p(1e4 * mel_energies)
            feats[i] = np.clip(log_mel / 10.0, 0.0, 1.0)

        return feats
