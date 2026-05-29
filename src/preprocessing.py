"""
src/preprocessing.py
══════════════════════════════════════════════════════════════════════════════
Audio preprocessing pipeline for maritime VHF ASR.

Handles:
  • Loading audio from various formats (wav, mp3, flac, ogg, m4a)
  • Resampling to 16 kHz
  • Mono conversion
  • Amplitude normalisation
  • Silence trimming
  • Duration filtering
  • Feature extraction for Whisper and Wav2Vec2
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T

log = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────
TARGET_SR = 16_000          # Whisper and Wav2Vec2 expect 16 kHz
TARGET_CHANNELS = 1         # Mono
NORM_TARGET_LEVEL = -20.0   # dBFS target for loudness normalisation


# ─── Core Audio I/O ──────────────────────────────────────────────────────────

def load_audio(
    path: str | Path,
    target_sr: int = TARGET_SR,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Load an audio file to a NumPy array.

    Returns:
        waveform : float32 NumPy array, shape (samples,) or (channels, samples)
        sr       : sample rate after resampling
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        # torchaudio is fast and handles most formats
        waveform, sr = torchaudio.load(str(path))          # (C, T) float32
    except Exception:
        # Fallback: librosa handles more exotic formats
        waveform_np, sr = librosa.load(str(path), sr=None, mono=False)
        if waveform_np.ndim == 1:
            waveform_np = waveform_np[np.newaxis, :]
        waveform = torch.from_numpy(waveform_np.astype(np.float32))

    # Resample
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        sr = target_sr

    # Mono conversion
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform.squeeze().numpy(), sr


def save_audio(
    waveform: np.ndarray,
    path: str | Path,
    sr: int = TARGET_SR,
) -> None:
    """Save a NumPy waveform to a WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform.astype(np.float32), sr)


# ─── Normalization ───────────────────────────────────────────────────────────

def rms_normalize(
    waveform: np.ndarray,
    target_db: float = NORM_TARGET_LEVEL,
) -> np.ndarray:
    """
    Root-Mean-Square loudness normalisation.
    Scales the signal so its RMS energy matches *target_db* dBFS.
    """
    rms = np.sqrt(np.mean(waveform ** 2))
    if rms < 1e-9:
        return waveform  # silence guard
    target_rms = 10 ** (target_db / 20.0)
    gain = target_rms / rms
    return (waveform * gain).clip(-1.0, 1.0).astype(np.float32)


def peak_normalize(waveform: np.ndarray) -> np.ndarray:
    """Scale waveform so the absolute peak equals 1.0."""
    peak = np.abs(waveform).max()
    if peak < 1e-9:
        return waveform
    return (waveform / peak).astype(np.float32)


# ─── Silence Trimming ────────────────────────────────────────────────────────

def trim_silence(
    waveform: np.ndarray,
    sr: int = TARGET_SR,
    top_db: float = 30.0,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Remove leading/trailing silence using librosa's trim utility.

    Parameters
    ----------
    top_db : Energy below this threshold (relative to peak) is treated as silence.
    """
    trimmed, _ = librosa.effects.trim(
        waveform,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    return trimmed.astype(np.float32)


# ─── Duration Filtering ──────────────────────────────────────────────────────

def check_duration(
    waveform: np.ndarray,
    sr: int = TARGET_SR,
    min_sec: float = 0.5,
    max_sec: float = 30.0,
) -> bool:
    """Return True if clip length is within [min_sec, max_sec]."""
    duration = len(waveform) / sr
    return min_sec <= duration <= max_sec


# ─── Full Preprocessing Pipeline ─────────────────────────────────────────────

def preprocess_audio(
    path: str | Path,
    target_sr: int = TARGET_SR,
    normalize: bool = True,
    trim: bool = True,
    min_duration: float = 0.5,
    max_duration: float = 30.0,
) -> np.ndarray | None:
    """
    End-to-end preprocessing for a single audio file.

    Steps:
        1. Load + resample to 16 kHz mono
        2. Optional silence trimming
        3. Duration filter (returns None if out of range)
        4. Optional RMS normalisation

    Returns float32 NumPy array or None if the clip should be skipped.
    """
    try:
        waveform, _ = load_audio(path, target_sr=target_sr, mono=True)
    except Exception as exc:
        log.warning("Could not load %s: %s", path, exc)
        return None

    if trim:
        waveform = trim_silence(waveform, sr=target_sr)

    if not check_duration(waveform, sr=target_sr, min_sec=min_duration, max_sec=max_duration):
        log.debug("Skipping %s: duration out of range", path)
        return None

    if normalize:
        waveform = rms_normalize(waveform)

    return waveform


def batch_preprocess(
    paths: list[str | Path],
    output_dir: str | Path | None = None,
    **preprocess_kwargs: Any,
) -> dict[str, np.ndarray | None]:
    """
    Preprocess multiple audio files. Optionally save processed versions.

    Returns a dict mapping original path → processed waveform (or None).
    """
    from tqdm import tqdm  # noqa: PLC0415

    results: dict[str, np.ndarray | None] = {}
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    for path in tqdm(paths, desc="Preprocessing audio", unit="file"):
        waveform = preprocess_audio(path, **preprocess_kwargs)
        results[str(path)] = waveform

        if waveform is not None and output_dir is not None:
            out_path = output_dir / Path(path).with_suffix(".wav").name
            save_audio(waveform, out_path, sr=preprocess_kwargs.get("target_sr", TARGET_SR))

    valid = sum(1 for v in results.values() if v is not None)
    log.info("Preprocessing complete: %d / %d files kept", valid, len(paths))
    return results


# ─── Feature Extraction ──────────────────────────────────────────────────────

def extract_log_mel(
    waveform: np.ndarray,
    sr: int = TARGET_SR,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
    fmin: float = 0.0,
    fmax: float = 8_000.0,
) -> np.ndarray:
    """
    Compute log-mel filterbank features.
    Used as a diagnostic / analysis tool.
    (Whisper / Wav2Vec2 do their own feature extraction internally.)

    Returns: (n_mels, T) float32 array
    """
    mel = librosa.feature.melspectrogram(
        y=waveform.astype(np.float64),
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


def get_audio_stats(
    waveform: np.ndarray,
    sr: int = TARGET_SR,
) -> dict[str, float]:
    """Return a dict of basic audio statistics for QA/EDA."""
    rms = float(np.sqrt(np.mean(waveform ** 2)))
    peak = float(np.abs(waveform).max())
    duration = len(waveform) / sr
    rms_db = 20 * np.log10(rms + 1e-9)
    peak_db = 20 * np.log10(peak + 1e-9)

    return {
        "duration_s": round(duration, 3),
        "rms": round(rms, 6),
        "rms_db": round(float(rms_db), 2),
        "peak": round(peak, 6),
        "peak_db": round(float(peak_db), 2),
        "samples": len(waveform),
        "sample_rate": sr,
    }


# ─── Text Normalisation ──────────────────────────────────────────────────────

def normalize_text(text: str, domain: str = "maritime") -> str:
    """
    Normalise transcription text for WER computation.

    • Lowercase
    • Remove punctuation (except hyphens in compound words)
    • Expand common maritime abbreviations
    • Collapse whitespace
    """
    import re  # noqa: PLC0415

    MARITIME_EXPANSIONS = {
        r"\bchan\b": "channel",
        r"\bch\b": "channel",
        r"\bhdr\b": "heading",
        r"\bkts\b": "knots",
        r"\bnm\b": "nautical miles",
        r"\bmmsi\b": "m m s i",
        r"\bvhf\b": "v h f",
        r"\bdsc\b": "d s c",
        r"\bmayday\b": "mayday",
        r"\bpan pan\b": "pan pan",
    }

    text = text.lower().strip()

    if domain == "maritime":
        for pattern, replacement in MARITIME_EXPANSIONS.items():
            text = re.sub(pattern, replacement, text)

    # Remove punctuation except apostrophes and hyphens
    text = re.sub(r"[^\w\s\'\-]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text
