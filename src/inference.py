"""
src/inference.py
══════════════════════════════════════════════════════════════════════════════
Transcription module.
  • Transcribe a single file or a directory of audio files
  • Supports Whisper and Wav2Vec2 (pretrained or fine-tuned checkpoint)
  • Reports WER when ground-truth transcriptions are provided
  • Benchmark inference speed (RTF — Real Time Factor)
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    pipeline,
)

from src.preprocessing import preprocess_audio, normalize_text
from src.metrics import compute_wer, compute_cer, InferenceTimer

log = logging.getLogger(__name__)
TARGET_SR = 16_000


# ─── Model Loading ───────────────────────────────────────────────────────────

class WhisperTranscriber:
    """
    Transcribe audio using a Whisper model.
    Works with both pretrained (openai/whisper-*) and fine-tuned checkpoints.
    """

    def __init__(
        self,
        model_path: str,
        language: str = "english",
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Loading Whisper from %s on %s", model_path, self.device)

        self.processor = WhisperProcessor.from_pretrained(
            model_path, language=language, task="transcribe"
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
        self.model.generation_config.forced_decoder_ids = None
        self.model.generation_config.suppress_tokens = []
        self.model.eval().to(self.device)

    def transcribe(self, audio: np.ndarray, max_new_tokens: int = 448) -> str:
        """Transcribe a single waveform array (float32, 16kHz)."""
        inputs = self.processor(
            audio, sampling_rate=TARGET_SR, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            ids = self.model.generate(
                inputs.input_features, max_new_tokens=max_new_tokens
            )

        return self.processor.tokenizer.decode(ids[0], skip_special_tokens=True)

    def transcribe_file(self, audio_path: str | Path) -> tuple[str, float]:
        """
        Transcribe an audio file.
        Returns (transcription, elapsed_seconds).
        """
        waveform = preprocess_audio(audio_path)
        if waveform is None:
            return "", 0.0

        with InferenceTimer(self.device) as t:
            text = self.transcribe(waveform)

        return text, t.elapsed


class Wav2Vec2Transcriber:
    """Transcribe audio using a Wav2Vec2 model."""

    def __init__(self, model_path: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Loading Wav2Vec2 from %s on %s", model_path, self.device)

        self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path)
        self.model.eval().to(self.device)

    def transcribe(self, audio: np.ndarray) -> str:
        inputs = self.processor(
            audio, sampling_rate=TARGET_SR, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        pred_ids = torch.argmax(logits, dim=-1)
        return self.processor.decode(pred_ids[0])

    def transcribe_file(self, audio_path: str | Path) -> tuple[str, float]:
        waveform = preprocess_audio(audio_path)
        if waveform is None:
            return "", 0.0

        with InferenceTimer(self.device) as t:
            text = self.transcribe(waveform)

        return text, t.elapsed


# ─── Batch Inference ─────────────────────────────────────────────────────────

def batch_transcribe(
    audio_paths: list[str | Path],
    transcriber: WhisperTranscriber | Wav2Vec2Transcriber,
    references: list[str] | None = None,
    normalize: bool = True,
) -> dict[str, Any]:
    """
    Transcribe a list of audio files and optionally compute WER.

    Returns:
        {
          "predictions": [{"file": ..., "hypothesis": ..., "reference": ...}],
          "wer": float,
          "cer": float,
          "avg_rtf": float,
        }
    """
    predictions = []
    elapsed_times = []

    for i, path in enumerate(tqdm(audio_paths, desc="Transcribing", unit="file")):
        hyp, elapsed = transcriber.transcribe_file(path)
        duration = len(preprocess_audio(path) or [1]) / TARGET_SR
        rtf = elapsed / max(duration, 1e-6)
        elapsed_times.append(rtf)

        entry = {
            "file": str(path),
            "hypothesis": normalize_text(hyp) if normalize else hyp,
        }
        if references is not None and i < len(references):
            entry["reference"] = (
                normalize_text(references[i]) if normalize else references[i]
            )
        predictions.append(entry)

    results: dict[str, Any] = {
        "predictions": predictions,
        "avg_rtf": round(float(np.mean(elapsed_times)), 4) if elapsed_times else 0,
        "n_files": len(audio_paths),
    }

    if references is not None:
        hyps = [p["hypothesis"] for p in predictions]
        refs = [p.get("reference", "") for p in predictions]
        wer_metrics = compute_wer(refs, hyps)
        results.update(wer_metrics)
        results["cer"] = compute_cer(refs, hyps)

    return results


# ─── Speed Benchmark ─────────────────────────────────────────────────────────

def benchmark_speed(
    transcriber: WhisperTranscriber | Wav2Vec2Transcriber,
    n_samples: int = 50,
    duration_seconds: float = 5.0,
) -> dict[str, float]:
    """
    Benchmark inference speed using synthetic silence audio.
    Returns avg_latency_ms, RTF (real-time factor), throughput (files/sec).
    """
    log.info("Benchmarking inference speed (%d samples × %.1fs)...", n_samples, duration_seconds)
    audio = np.zeros(int(duration_seconds * TARGET_SR), dtype=np.float32)
    times = []

    for _ in range(n_samples):
        with InferenceTimer(transcriber.device) as t:
            if isinstance(transcriber, WhisperTranscriber):
                transcriber.transcribe(audio)
            else:
                transcriber.transcribe(audio)
        times.append(t.elapsed)

    avg_latency = float(np.mean(times))
    return {
        "avg_latency_ms": round(avg_latency * 1000, 2),
        "rtf": round(avg_latency / duration_seconds, 4),
        "throughput_files_per_sec": round(1.0 / avg_latency, 2),
        "n_samples": n_samples,
        "clip_duration_s": duration_seconds,
    }


# ─── Factory ─────────────────────────────────────────────────────────────────

def build_transcriber(
    model_type: str,
    model_path: str,
    **kwargs: Any,
) -> WhisperTranscriber | Wav2Vec2Transcriber:
    if model_type == "whisper":
        return WhisperTranscriber(model_path, **kwargs)
    elif model_type == "wav2vec2":
        return Wav2Vec2Transcriber(model_path, **kwargs)
    raise ValueError(f"Unknown model_type: {model_type!r}")
