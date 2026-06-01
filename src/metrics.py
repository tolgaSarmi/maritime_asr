"""
src/metrics.py
══════════════════════════════════════════════════════════════════════════════
Evaluation metrics for ASR:
  • Word Error Rate (WER)
  • Character Error Rate (CER)
  • BLEU Score
  • Edit-distance breakdown (S/D/I)
  • Per-sample and corpus-level evaluation
  • Inference speed benchmarking
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


# ─── Levenshtein / Edit Distance ─────────────────────────────────────────────

def _edit_distance(
    reference: list[str],
    hypothesis: list[str],
) -> tuple[int, int, int, int]:
    """
    Compute edit distance at token level.

    Returns:
        (distance, substitutions, deletions, insertions)
    """
    n, m = len(reference), len(hypothesis)
    # DP table
    dp = np.zeros((n + 1, m + 1), dtype=int)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1],  # substitution
                )

    # Back-trace to count S/D/I
    i, j = n, m
    subs = dels = ins = 0
    while i > 0 and j > 0:
        if reference[i - 1] == hypothesis[j - 1]:
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    dels += i
    ins += j

    return int(dp[n][m]), subs, dels, ins


# ─── WER / CER ───────────────────────────────────────────────────────────────

def compute_wer(
    references: list[str],
    hypotheses: list[str],
) -> dict[str, float]:
    """
    Corpus-level Word Error Rate.

    WER = (S + D + I) / N   where N = total reference words.

    Returns dict with wer, substitution_rate, deletion_rate, insertion_rate.
    """
    total_words = 0
    total_s = total_d = total_i = 0

    for ref, hyp in zip(references, hypotheses):
        ref_words = ref.split()
        hyp_words = hyp.split()
        _, s, d, i = _edit_distance(ref_words, hyp_words)
        total_words += len(ref_words)
        total_s += s
        total_d += d
        total_i += i

    if total_words == 0:
        return {"wer": 0.0, "sub_rate": 0.0, "del_rate": 0.0, "ins_rate": 0.0}

    return {
        "wer": round((total_s + total_d + total_i) / total_words, 4),
        "sub_rate": round(total_s / total_words, 4),
        "del_rate": round(total_d / total_words, 4),
        "ins_rate": round(total_i / total_words, 4),
        "total_ref_words": total_words,
        "total_substitutions": total_s,
        "total_deletions": total_d,
        "total_insertions": total_i,
    }


def compute_cer(
    references: list[str],
    hypotheses: list[str],
) -> float:
    """
    Corpus-level Character Error Rate.
    CER = edit_distance(chars) / total_ref_chars
    """
    total_chars = 0
    total_errors = 0

    for ref, hyp in zip(references, hypotheses):
        ref_chars = list(ref.replace(" ", ""))
        hyp_chars = list(hyp.replace(" ", ""))
        err, _, _, _ = _edit_distance(ref_chars, hyp_chars)
        total_chars += len(ref_chars)
        total_errors += err

    return round(total_errors / max(total_chars, 1), 4)


def compute_bleu(
    references: list[str],
    hypotheses: list[str],
    max_n: int = 4,
) -> float:
    """Corpus BLEU score (simplified, sentence-level average)."""
    try:
        from evaluate import load  # noqa: PLC0415

        bleu_metric = load("bleu")
        refs_formatted = [[r.split()] for r in references]
        hyps_formatted = [h.split() for h in hypotheses]
        result = bleu_metric.compute(predictions=hyps_formatted, references=refs_formatted)
        return round(float(result.get("bleu", 0.0)), 4)
    except Exception:
        return 0.0


# ─── Per-sample Metrics ──────────────────────────────────────────────────────

def compute_sample_wer(reference: str, hypothesis: str) -> float:
    """WER for a single (ref, hyp) pair."""
    return compute_wer([reference], [hypothesis])["wer"]


# ─── Metrics Aggregator ──────────────────────────────────────────────────────

@dataclass
class MetricsAccumulator:
    """Accumulate predictions/references across batches, compute at epoch end."""

    references: list[str] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    inference_times: list[float] = field(default_factory=list)

    def update(
        self,
        refs: list[str],
        hyps: list[str],
        loss: float | None = None,
        elapsed: float | None = None,
    ) -> None:
        self.references.extend(refs)
        self.hypotheses.extend(hyps)
        if loss is not None:
            self.losses.append(loss)
        if elapsed is not None:
            self.inference_times.append(elapsed)

    def compute(self) -> dict[str, Any]:
        if not self.references:
            return {}

        wer_metrics = compute_wer(self.references, self.hypotheses)
        cer = compute_cer(self.references, self.hypotheses)

        results: dict[str, Any] = {
            **wer_metrics,
            "cer": cer,
            "n_samples": len(self.references),
        }

        if self.losses:
            results["loss"] = round(float(np.mean(self.losses)), 4)

        if self.inference_times:
            total_t = sum(self.inference_times)
            results["rtf"] = round(total_t / max(len(self.references), 1), 4)
            results["avg_inference_ms"] = round(total_t / len(self.inference_times) * 1000, 2)

        return results

    def reset(self) -> None:
        self.references.clear()
        self.hypotheses.clear()
        self.losses.clear()
        self.inference_times.clear()


# ─── Error Analysis ──────────────────────────────────────────────────────────

def analyse_errors(
    references: list[str],
    hypotheses: list[str],
    top_n: int = 20,
) -> dict[str, Any]:
    """
    Analyse common error patterns.

    Returns:
        substitution_pairs : Counter of (ref_word → hyp_word)
        deleted_words      : Counter of words only in reference
        inserted_words     : Counter of words only in hypothesis
        per_sample_wer     : list of per-sample WER values
    """
    sub_pairs: list[tuple[str, str]] = []
    deleted_words: list[str] = []
    inserted_words: list[str] = []
    per_sample_wer: list[float] = []

    for ref, hyp in zip(references, hypotheses):
        ref_words = ref.split()
        hyp_words = hyp.split()
        per_sample_wer.append(compute_sample_wer(ref, hyp))

        # Simple diff (not perfect alignment, but useful for analysis)
        ref_set = set(ref_words)
        hyp_set = set(hyp_words)
        for w in ref_set - hyp_set:
            deleted_words.append(w)
        for w in hyp_set - ref_set:
            inserted_words.append(w)

    return {
        "per_sample_wer": per_sample_wer,
        "wer_mean": round(float(np.mean(per_sample_wer)), 4),
        "wer_std": round(float(np.std(per_sample_wer)), 4),
        "wer_median": round(float(np.median(per_sample_wer)), 4),
        "wer_p90": round(float(np.percentile(per_sample_wer, 90)), 4),
        "top_deleted_words": Counter(deleted_words).most_common(top_n),
        "top_inserted_words": Counter(inserted_words).most_common(top_n),
    }


# ─── Speed Benchmarking ──────────────────────────────────────────────────────

class InferenceTimer:
    """Context manager to measure inference latency."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.elapsed: float = 0.0

    def __enter__(self) -> "InferenceTimer":
        if "cuda" in self.device:
            import torch  # noqa: PLC0415

            torch.cuda.synchronize()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        if "cuda" in self.device:
            import torch  # noqa: PLC0415

            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self._start
