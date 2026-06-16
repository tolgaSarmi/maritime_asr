"""
src/preprocessing.py
══════════════════════════════════════════════════════════════════════════════
Audio preprocessing pipeline for maritime VHF ASR.

Handles:
  • Loading audio from local files (wav, mp3, flac, ogg, m4a)
  • Loading audio from remote URLs (Azure Blob Storage, public or SAS-signed)
    → Cached to /tmp/asr_audio_cache/ so each file is only fetched once
  • Resampling to 16 kHz
  • Mono conversion
  • Amplitude normalisation
  • Silence trimming
  • Duration filtering
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import hashlib
import io
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
TARGET_SR          = 16_000
TARGET_CHANNELS    = 1
NORM_TARGET_LEVEL  = -20.0   # dBFS
AUDIO_CACHE_DIR    = Path("/tmp/asr_audio_cache")


# ─── URL Detection ───────────────────────────────────────────────────────────
def is_url(s: str) -> bool:
    return str(s).startswith(("http://", "https://"))


# ─── URL → bytes (with disk cache) ──────────────────────────────────────────
def _fetch_url_bytes(url: str) -> bytes:
    """
    Download audio from URL with local disk cache.

    Cache key = MD5 of the URL path (ignoring SAS query params) so the same
    file is not downloaded twice across training epochs.
    """
    from urllib.parse import urlparse  # noqa: PLC0415

    AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Strip SAS query string for the cache key so the same blob
    # always maps to the same cache file even if the SAS token rotates.
    parsed   = urlparse(url)
    cache_key = hashlib.md5(
        (parsed.netloc + parsed.path).encode()
    ).hexdigest()
    suffix   = Path(parsed.path).suffix or ".wav"
    cache_path = AUDIO_CACHE_DIR / f"{cache_key}{suffix}"

    if cache_path.exists():
        return cache_path.read_bytes()

    import requests as _requests  # plain — no auth headers  # noqa: PLC0415
    r = _requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.content
    cache_path.write_bytes(data)
    return data


# ─── Core Audio I/O ──────────────────────────────────────────────────────────
def load_audio(
    path_or_url: str | Path,
    target_sr: int = TARGET_SR,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Load audio from a local path OR a remote URL.

    Returns:
        waveform : float32 NumPy array, shape (samples,)
        sr       : sample rate after resampling
    """
    src = str(path_or_url)

    if is_url(src):
        # Fetch (or retrieve from cache) and decode in-memory
        raw = _fetch_url_bytes(src)
        buf = io.BytesIO(raw)
        try:
            waveform, sr = torchaudio.load(buf)
        except Exception:
            buf.seek(0)
            waveform_np, sr = librosa.load(buf, sr=None, mono=False)
            if waveform_np.ndim == 1:
                waveform_np = waveform_np[np.newaxis, :]
            waveform = torch.from_numpy(waveform_np.astype(np.float32))
    else:
        path = Path(src)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        try:
            waveform, sr = torchaudio.load(str(path))
        except Exception:
            waveform_np, sr = librosa.load(str(path), sr=None, mono=False)
            if waveform_np.ndim == 1:
                waveform_np = waveform_np[np.newaxis, :]
            waveform = torch.from_numpy(waveform_np.astype(np.float32))

    # Resample
    if sr != target_sr:
        waveform = T.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
        sr = target_sr

    # Mono
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


# ─── Normalisation ───────────────────────────────────────────────────────────
def rms_normalize(
    waveform: np.ndarray,
    target_db: float = NORM_TARGET_LEVEL,
) -> np.ndarray:
    rms = np.sqrt(np.mean(waveform ** 2))
    if rms < 1e-9:
        return waveform
    target_rms = 10 ** (target_db / 20.0)
    gain = target_rms / rms
    return (waveform * gain).clip(-1.0, 1.0).astype(np.float32)


def peak_normalize(waveform: np.ndarray) -> np.ndarray:
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
    trimmed, _ = librosa.effects.trim(
        waveform, top_db=top_db,
        frame_length=frame_length, hop_length=hop_length,
    )
    return trimmed.astype(np.float32)


# ─── Duration Filtering ──────────────────────────────────────────────────────
def check_duration(
    waveform: np.ndarray,
    sr: int = TARGET_SR,
    min_sec: float = 0.5,
    max_sec: float = 30.0,
) -> bool:
    duration = len(waveform) / sr
    return min_sec <= duration <= max_sec


# ─── Full Preprocessing Pipeline ─────────────────────────────────────────────
def preprocess_audio(
    path_or_url: str | Path,
    target_sr: int = TARGET_SR,
    normalize: bool = True,
    trim: bool = True,
    min_duration: float = 0.5,
    max_duration: float = 30.0,
) -> np.ndarray | None:
    """
    End-to-end preprocessing for a single audio file or URL.

    Accepts:
        • Local file path  – data/simulated/audio/file.wav
        • Azure Blob URL   – https://ircgaudiostorage.blob.core.windows.net/...

    Steps:
        1. Load + resample to 16 kHz mono (URL fetched and cached on first access)
        2. Optional silence trimming
        3. Duration filter (returns None if out of range)
        4. Optional RMS normalisation

    Returns float32 NumPy array or None if the clip should be skipped.
    """
    try:
        waveform, _ = load_audio(path_or_url, target_sr=target_sr, mono=True)
    except Exception as exc:
        log.warning("Could not load %s: %s", path_or_url, exc)
        return None

    if trim:
        waveform = trim_silence(waveform, sr=target_sr)

    if not check_duration(waveform, sr=target_sr, min_sec=min_duration, max_sec=max_duration):
        log.debug("Skipping %s: duration out of range", path_or_url)
        return None

    if normalize:
        waveform = rms_normalize(waveform)

    return waveform



# ─── Text Normalisation ──────────────────────────────────────────────────────
def normalize_text(text: str, domain: str = "maritime") -> str:
    """
    Normalise transcription text for WER computation.
    • Lowercase
    • Expand common maritime abbreviations
    • Remove punctuation (except hyphens)
    • Collapse whitespace
    """
    import re  # noqa: PLC0415

    MARITIME_EXPANSIONS = {
        r"\bchan\b":  "channel",
        r"\bch\b":    "channel",
        r"\bhdr\b":   "heading",
        r"\bkts\b":   "knots",
        r"\bnm\b":    "nautical miles",
        r"\bmmsi\b":  "m m s i",
        r"\bvhf\b":   "v h f",
        r"\bdsc\b":   "d s c",
        r"\bmayday\b": "mayday",
        r"\bpan pan\b": "pan pan",
    }

    text = text.lower().strip()
    if domain == "maritime":
        for pattern, replacement in MARITIME_EXPANSIONS.items():
            text = re.sub(pattern, replacement, text)

    text = re.sub(r"[^\w\s\'\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
