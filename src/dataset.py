"""
src/dataset.py
══════════════════════════════════════════════════════════════════════════════
PyTorch Dataset classes for maritime VHF ASR.

Supports:
  • ASRDataset          — generic manifest-based loader
  • WhisperASRDataset   — wraps Whisper feature extractor + tokeniser
  • Wav2Vec2ASRDataset  — wraps Wav2Vec2 processor
  • DatasetFactory      — factory to build train/val/test splits
  • DataCollator        — model-specific batch collators
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    Wav2Vec2Processor,
)

from src.preprocessing import is_url, normalize_text, preprocess_audio

log = logging.getLogger(__name__)

TARGET_SR = 16_000



# ─── Base Dataset ────────────────────────────────────────────────────────────

class ASRDataset(Dataset):
    """
    Generic manifest-driven ASR dataset.

    The manifest JSON/CSV must contain at minimum:
        • audio_file    – path relative to *data_root* (or absolute)
        • transcription – ground-truth text
        • data_type     – 'real' or 'simulated'
    """

    def __init__(
        self,
        manifest_path: str | Path,
        data_root: str | Path,
        split: str = "train",
        max_duration: float = 30.0,
        min_duration: float = 0.5,
        normalize: bool = True,
        normalize_text_fn: bool = True,
        domain: str = "maritime",
        max_samples: int | None = None,
        data_type_filter: str | None = None,   # 'real', 'simulated', or None
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.normalize = normalize
        self.normalize_text_fn = normalize_text_fn
        self.domain = domain

        self.samples: list[dict] = self._load_manifest(
            manifest_path,
            max_samples=max_samples,
            data_type_filter=data_type_filter,
        )
        log.info(
            "[%s] %s split: %d samples loaded",
            data_type_filter or "all",
            split,
            len(self.samples),
        )

    # ── Manifest Loading ──────────────────────────────────────────────────────
    def _load_manifest(
        self,
        path: str | Path,
        max_samples: int | None,
        data_type_filter: str | None,
    ) -> list[dict]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")

        if path.suffix == ".csv":
            df = pd.read_csv(path)
            records = df.to_dict(orient="records")
        else:
            with open(path, encoding="utf-8") as f:
                records = json.load(f)

        # Filter by data type
        if data_type_filter:
            records = [r for r in records if r.get("data_type") == data_type_filter]

        # Drop records with missing fields
        valid = []
        for r in records:
            if not r.get("transcription") or not r.get("audio_file"):
                continue
            valid.append(r)

        if max_samples:
            valid = valid[:max_samples]

        return valid

    # ── PyTorch interface ────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        audio_path = self._resolve_path(sample["audio_file"])

        waveform = preprocess_audio(
            audio_path,
            normalize=self.normalize,
            min_duration=self.min_duration,
            max_duration=self.max_duration,
        )
        if waveform is None:
            # Return zero-length signal — collator will skip
            waveform = np.zeros(TARGET_SR // 4, dtype=np.float32)

        text = sample["transcription"]
        if self.normalize_text_fn:
            text = normalize_text(text, domain=self.domain)

        return {
            "input_values": waveform,
            "labels": text,
            "audio_path": str(audio_path),
            "data_type": sample.get("data_type", "unknown"),
            "duration": len(waveform) / TARGET_SR,
        }

    def _resolve_path(self, audio_file: str) -> str | Path:
        # If it's a URL, return as-is (preprocessing.load_audio handles URLs directly)
        if is_url(audio_file):
            return audio_file

        # Local path resolution
        p = Path(audio_file)
        if p.is_absolute() and p.exists():
            return p
        candidate = self.data_root / audio_file
        if candidate.exists():
            return candidate
        # Last resort: try stripping leading dirs
        return self.data_root / p.name

    # ── Statistics ───────────────────────────────────────────────────────────
    def statistics(self) -> dict[str, Any]:
        durations = []
        url_count = 0
        for s in self.samples:
            p = self._resolve_path(s["audio_file"])
            if is_url(str(p)):
                url_count += 1
                continue
            try:
                import soundfile as sf  # noqa: PLC0415
                info = sf.info(str(p))
                durations.append(info.duration)
            except Exception:
                pass
        total_h = sum(durations) / 3600 if durations else 0
        texts = [s["transcription"] for s in self.samples]
        stats = {
            "n_samples": len(self.samples),
            "total_hours": round(total_h, 2),
            "mean_duration_s": round(np.mean(durations), 2) if durations else 0,
            "vocab_size": len(set(" ".join(texts).split())),
            "mean_words": round(np.mean([len(t.split()) for t in texts]), 1),
        }
        if url_count:
            stats["url_audio_count"] = url_count
            stats["note"] = "duration unavailable for cloud audio"
        return stats


# ─── Whisper Dataset ─────────────────────────────────────────────────────────

class WhisperASRDataset(ASRDataset):
    """
    Whisper-specific dataset that runs the Whisper FeatureExtractor
    and tokenises the labels.
    """

    def __init__(
        self,
        *args: Any,
        feature_extractor: WhisperFeatureExtractor,
        tokenizer: WhisperTokenizer,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def __getitem__(self, idx: int) -> dict[str, Any]:
        base = super().__getitem__(idx)
        waveform = base["input_values"]

        # Whisper feature extraction: 80-channel log-mel, 30s fixed length
        features = self.feature_extractor(
            waveform,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
        )
        input_features = features.input_features[0]  # (80, 3000)

        # Tokenize transcript
        labels = self.tokenizer(base["labels"], return_tensors="pt").input_ids[0]

        return {
            "input_features": input_features,
            "labels": labels,
            "audio_path": base["audio_path"],
            "data_type": base["data_type"],
            "transcription": base["labels"],
        }


# ─── Wav2Vec2 Dataset ────────────────────────────────────────────────────────

class Wav2Vec2ASRDataset(ASRDataset):
    """
    Wav2Vec2-specific dataset.
    Runs the Wav2Vec2Processor for both audio and label tokenisation.
    """

    def __init__(
        self,
        *args: Any,
        processor: Wav2Vec2Processor,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.processor = processor

    def __getitem__(self, idx: int) -> dict[str, Any]:
        base = super().__getitem__(idx)
        waveform = base["input_values"]

        # Processor normalises to zero mean / unit variance
        inputs = self.processor(
            waveform,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
        )
        input_values = inputs.input_values[0]

        # Tokenise labels via the processor vocabulary
        # as_target_processor() was removed in transformers 5.0; use tokenizer directly.
        # facebook/wav2vec2-base-960h vocab is uppercase-only (do_lower_case=false);
        # normalize_text() lowercases, so we must uppercase here to avoid all-<unk> labels
        # which produce target_lengths=0, loss=0, and NaN gradients via degenerate CTC.
        labels = self.processor.tokenizer(base["labels"].upper()).input_ids

        return {
            "input_values": input_values,
            "labels": torch.tensor(labels, dtype=torch.long),
            "audio_path": base["audio_path"],
            "data_type": base["data_type"],
            "transcription": base["labels"],
        }


# ─── Data Collators ──────────────────────────────────────────────────────────

@dataclass
class WhisperDataCollator:
    """
    Pad Whisper batch items.
    input_features → already 30-s padded by feature extractor (no padding needed)
    labels         → right-pad with -100 (ignored by loss)
    """

    tokenizer: WhisperTokenizer
    padding: bool = True

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        label_features = [{"input_ids": f["labels"]} for f in features]

        # input_features are already fixed 80×3000 mel spectrograms —
        # stack directly; do NOT call tokenizer.pad() on audio tensors
        batch = {
            "input_features": torch.stack([f["input_features"] for f in features])
        }
        labels_batch = self.tokenizer.pad(
            label_features, padding=True, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # Whisper SOS token should not be predicted
        if (labels[:, 0] == self.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        # Note: "transcriptions" removed - metadata fields cause ValueError in model.generate()
        # Standalone evaluators decode refs from labels instead
        return batch


@dataclass
class Wav2Vec2DataCollator:
    """
    Pad Wav2Vec2 batch items.
    input_values → left/right pad to max length in batch
    labels       → pad with -100
    """

    processor: Wav2Vec2Processor
    padding: bool = True

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        input_vals = [{"input_values": f["input_values"]} for f in features]
        label_vals = [{"input_ids": f["labels"].tolist()} for f in features]

        batch = self.processor.pad(
            input_vals, padding=True, return_tensors="pt"
        )
        # as_target_processor() was removed in transformers 5.0; use tokenizer directly
        labels_batch = self.processor.tokenizer.pad(
            label_vals, padding=True, return_tensors="pt"
        )

        # Guard: attention_mask may be absent when Wav2Vec2CTCTokenizer's
        # model_input_names omits it.  Fall back to an all-real mask so we
        # never silently replace every label with -100.
        if "attention_mask" in labels_batch:
            mask = labels_batch.attention_mask.ne(1)
        else:
            mask = torch.zeros(
                labels_batch["input_ids"].shape, dtype=torch.bool
            )

        labels = labels_batch["input_ids"].masked_fill(mask, -100)
        batch["labels"] = labels

        # Note: "transcriptions" removed - metadata fields cause ValueError in model.generate()
        # Standalone evaluators decode refs from labels instead
        return batch
