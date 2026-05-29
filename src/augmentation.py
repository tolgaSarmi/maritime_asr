"""
src/augmentation.py
══════════════════════════════════════════════════════════════════════════════
Audio augmentation pipeline for maritime VHF ASR robustness.

Augmentations:
  • Time stretching
  • Pitch shifting
  • Additive Gaussian / background noise
  • VHF radio channel simulation (band-pass, squelch noise, clipping)
  • SpecAugment (frequency + time masking)
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import random
from typing import Any

import librosa
import numpy as np
import torch
import torchaudio.transforms as T

log = logging.getLogger(__name__)


# ─── Base Augment ────────────────────────────────────────────────────────────

class AudioAugmentation:
    """Abstract base for a single audio augmentation."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def apply(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        if random.random() < self.p:
            try:
                return self.apply(waveform, sr)
            except Exception as exc:
                log.debug("%s failed: %s", type(self).__name__, exc)
        return waveform


# ─── Time Domain Augmentations ───────────────────────────────────────────────

class TimeStretch(AudioAugmentation):
    """
    Stretch or compress audio in time without changing pitch.
    Simulates faster/slower speech rate.
    """

    def __init__(self, min_rate: float = 0.9, max_rate: float = 1.1, p: float = 0.5):
        super().__init__(p)
        self.min_rate = min_rate
        self.max_rate = max_rate

    def apply(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        rate = random.uniform(self.min_rate, self.max_rate)
        return librosa.effects.time_stretch(waveform, rate=rate).astype(np.float32)


class PitchShift(AudioAugmentation):
    """
    Shift pitch by a random number of semitones.
    Models speaker variability without changing duration.
    """

    def __init__(self, min_semitones: float = -2.0, max_semitones: float = 2.0, p: float = 0.5):
        super().__init__(p)
        self.min_semitones = min_semitones
        self.max_semitones = max_semitones

    def apply(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        n_steps = random.uniform(self.min_semitones, self.max_semitones)
        return librosa.effects.pitch_shift(waveform, sr=sr, n_steps=n_steps).astype(np.float32)


class AddGaussianNoise(AudioAugmentation):
    """
    Add white Gaussian noise at a random SNR.
    Models general background noise and sensor noise.
    """

    def __init__(self, min_snr_db: float = 10.0, max_snr_db: float = 40.0, p: float = 0.5):
        super().__init__(p)
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db

    def apply(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        snr_db = random.uniform(self.min_snr_db, self.max_snr_db)
        rms_signal = np.sqrt(np.mean(waveform ** 2))
        rms_noise = rms_signal / (10 ** (snr_db / 20.0))
        noise = np.random.normal(0, rms_noise, len(waveform)).astype(np.float32)
        noisy = waveform + noise
        return noisy.clip(-1.0, 1.0).astype(np.float32)


class VolumeJitter(AudioAugmentation):
    """Random gain between ±6 dB."""

    def __init__(self, min_gain_db: float = -6.0, max_gain_db: float = 6.0, p: float = 0.5):
        super().__init__(p)
        self.min_gain = 10 ** (min_gain_db / 20.0)
        self.max_gain = 10 ** (max_gain_db / 20.0)

    def apply(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        gain = random.uniform(self.min_gain, self.max_gain)
        return (waveform * gain).clip(-1.0, 1.0).astype(np.float32)


class SimulateVHFChannel(AudioAugmentation):
    """
    Simulate VHF maritime radio channel characteristics:
      • Band-pass filter (300 Hz – 3 kHz, telephone quality)
      • Soft clipping (radio saturation)
      • Additive squelch/carrier noise
      • Optional frequency offset / Doppler
    """

    def __init__(
        self,
        bandwidth: tuple[float, float] = (300.0, 3000.0),
        clip_db: float = -3.0,
        noise_snr_db: float = 25.0,
        p: float = 0.4,
    ):
        super().__init__(p)
        self.bandwidth = bandwidth
        self.clip_threshold = 10 ** (clip_db / 20.0)
        self.noise_snr_db = noise_snr_db

    def apply(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        from scipy.signal import butter, sosfilt  # noqa: PLC0415

        # 1. Band-pass filter
        low = self.bandwidth[0] / (sr / 2)
        high = self.bandwidth[1] / (sr / 2)
        low = max(low, 0.001)
        high = min(high, 0.999)
        sos = butter(5, [low, high], btype="band", output="sos")
        filtered = sosfilt(sos, waveform).astype(np.float32)

        # 2. Soft clipping
        filtered = np.tanh(filtered / self.clip_threshold) * self.clip_threshold

        # 3. Squelch noise
        rms = np.sqrt(np.mean(filtered ** 2)) + 1e-9
        noise_rms = rms / (10 ** (self.noise_snr_db / 20.0))
        noise = np.random.normal(0, noise_rms, len(filtered)).astype(np.float32)
        output = filtered + noise

        # 4. Peak-normalise
        peak = np.abs(output).max()
        if peak > 1e-6:
            output = output / peak * 0.95

        return output.astype(np.float32)


# ─── SpecAugment ─────────────────────────────────────────────────────────────

class SpecAugment:
    """
    SpecAugment (Park et al., 2019) for transformer ASR.
    Applied in the feature domain (on log-mel spectrograms).

    This wrapper applies masking via torchaudio's FrequencyMasking and
    TimeMasking transforms to a (freq, time) tensor.
    """

    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
    ):
        self.freq_maskers = [
            T.FrequencyMasking(freq_mask_param=freq_mask_param)
            for _ in range(num_freq_masks)
        ]
        self.time_maskers = [
            T.TimeMasking(time_mask_param=time_mask_param)
            for _ in range(num_time_masks)
        ]

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        spec : (freq, time) or (batch, freq, time) tensor
        """
        for masker in self.freq_maskers:
            spec = masker(spec)
        for masker in self.time_maskers:
            spec = masker(spec)
        return spec


# ─── Augmentation Pipeline ───────────────────────────────────────────────────

class AugmentationPipeline:
    """
    Composes multiple audio augmentations.
    Each augmentation is applied independently with its own probability.
    """

    def __init__(self, augmentations: list[AudioAugmentation]):
        self.augmentations = augmentations

    def __call__(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        for aug in self.augmentations:
            waveform = aug(waveform, sr)
        return waveform

    @classmethod
    def from_config(cls, cfg: Any) -> "AugmentationPipeline":
        """Build pipeline from OmegaConf augmentation config block."""
        augs: list[AudioAugmentation] = []
        aug_cfg = cfg.augmentation

        if aug_cfg.time_stretch.enabled:
            augs.append(
                TimeStretch(
                    min_rate=aug_cfg.time_stretch.min_rate,
                    max_rate=aug_cfg.time_stretch.max_rate,
                )
            )
        if aug_cfg.pitch_shift.enabled:
            augs.append(
                PitchShift(
                    min_semitones=aug_cfg.pitch_shift.min_semitones,
                    max_semitones=aug_cfg.pitch_shift.max_semitones,
                )
            )
        if aug_cfg.add_noise.enabled:
            augs.append(
                AddGaussianNoise(
                    min_snr_db=aug_cfg.add_noise.min_snr_db,
                    max_snr_db=aug_cfg.add_noise.max_snr_db,
                )
            )
        # Always add VHF simulation for maritime robustness
        augs.append(SimulateVHFChannel(p=0.3))
        augs.append(VolumeJitter(p=0.5))

        log.info("AugmentationPipeline built with %d transforms", len(augs))
        return cls(augs)

    def disabled(self) -> "AugmentationPipeline":
        """Return a no-op pipeline (for val/test)."""
        return AugmentationPipeline([])
