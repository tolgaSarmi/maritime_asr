"""
src/evaluate.py
══════════════════════════════════════════════════════════════════════════════
Model evaluation pipeline for the ASR dissertation.

Covers:
  • Zero-shot baseline evaluation (pretrained, no fine-tuning)
  • Fine-tuned model evaluation on real / simulated / cross-dataset
  • Batch inference with GPU/CPU support
  • Results persisted as JSON for downstream visualisation
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    pipeline,
)

from src.dataset import (
    ASRDataset,
    WhisperASRDataset,
    Wav2Vec2ASRDataset,
    WhisperDataCollator,
    Wav2Vec2DataCollator,
)
from src.metrics import (
    MetricsAccumulator,
    compute_wer,
    compute_cer,
    compute_bleu,
    analyse_errors,
    InferenceTimer,
)
from src.preprocessing import normalize_text
from src.utils import save_results

log = logging.getLogger(__name__)


# ─── Whisper Evaluator ───────────────────────────────────────────────────────

class WhisperEvaluator:
    """
    Evaluate a Whisper model (pretrained or fine-tuned) on a given dataset.

    Parameters
    ----------
    model_path : HuggingFace model name  OR  path to fine-tuned checkpoint dir
    """

    def __init__(
        self,
        model_path: str,
        cfg: Any,
        device: torch.device | None = None,
    ):
        self.model_path = model_path
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        log.info("Loading Whisper from: %s", model_path)
        self.processor = WhisperProcessor.from_pretrained(
            model_path,
            language=cfg.models.whisper.language,
            task="transcribe",
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
        self.model.generation_config.forced_decoder_ids = None
        self.model.generation_config.suppress_tokens = []
        self.model.eval()
        self.model.to(self.device)
        log.info("Whisper ready on %s", self.device)

    def evaluate(
        self,
        manifest_path: str | Path,
        data_root: str | Path,
        split: str = "test",
        batch_size: int = 8,
    ) -> dict[str, Any]:
        """Run evaluation and return metrics dict."""
        dataset = WhisperASRDataset(
            manifest_path=manifest_path,
            data_root=data_root,
            split=split,
            feature_extractor=self.processor.feature_extractor,
            tokenizer=self.processor.tokenizer,
        )
        collator = WhisperDataCollator(tokenizer=self.processor.tokenizer)
        loader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=collator, num_workers=2
        )

        accumulator = MetricsAccumulator()
        all_pairs: list[dict] = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Whisper eval [{split}]"):
                input_features = batch["input_features"].to(self.device)
                labels = batch["labels"]
                refs = batch.get("transcriptions", [])

                t0 = time.perf_counter()
                predicted_ids = self.model.generate(
                    input_features,
                    max_length=self.cfg.models.whisper.max_new_tokens,
                )
                elapsed = time.perf_counter() - t0

                # Decode
                hyps = self.processor.tokenizer.batch_decode(
                    predicted_ids, skip_special_tokens=True
                )

                if not refs:
                    label_ids = labels.clone()
                    label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
                    refs = self.processor.tokenizer.batch_decode(
                        label_ids, skip_special_tokens=True
                    )

                # Normalise for fair WER comparison
                refs_norm = [normalize_text(r) for r in refs]
                hyps_norm = [normalize_text(h) for h in hyps]

                accumulator.update(refs_norm, hyps_norm, elapsed=elapsed)
                for ref, hyp in zip(refs_norm, hyps_norm):
                    all_pairs.append({"reference": ref, "hypothesis": hyp})

        metrics = accumulator.compute()
        error_analysis = analyse_errors(
            [p["reference"] for p in all_pairs],
            [p["hypothesis"] for p in all_pairs],
        )
        metrics["error_analysis"] = error_analysis
        metrics["predictions"] = all_pairs[:50]  # keep first 50 for inspection
        metrics["model_path"] = str(self.model_path)
        metrics["split"] = split

        log.info(
            "Whisper eval complete — WER: %.4f  CER: %.4f  Samples: %d",
            metrics.get("wer", 0),
            metrics.get("cer", 0),
            metrics.get("n_samples", 0),
        )
        return metrics


# ─── Wav2Vec2 Evaluator ──────────────────────────────────────────────────────

class Wav2Vec2Evaluator:
    """Evaluate Wav2Vec2 (pretrained or fine-tuned) on a given manifest."""

    def __init__(
        self,
        model_path: str,
        cfg: Any,
        device: torch.device | None = None,
    ):
        self.model_path = model_path
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        log.info("Loading Wav2Vec2 from: %s", model_path)
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        except (OSError, EnvironmentError):
            # Wav2Vec2Trainer omitted processor.save_pretrained() so the
            # checkpoint has no tokenizer/feature-extractor files.
            # config.json._name_or_path is set to the output dir by
            # trainer.save_model(), not the original HF model name, so we
            # read the name from cfg instead.
            original_name = cfg.models.wav2vec2.name
            log.warning(
                "No processor files in checkpoint %s — loading from %s",
                model_path, original_name,
            )
            self.processor = Wav2Vec2Processor.from_pretrained(original_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path)
        self.model.eval()
        self.model.to(self.device)
        log.info("Wav2Vec2 ready on %s", self.device)

    def evaluate(
        self,
        manifest_path: str | Path,
        data_root: str | Path,
        split: str = "test",
        batch_size: int = 8,
    ) -> dict[str, Any]:
        dataset = Wav2Vec2ASRDataset(
            manifest_path=manifest_path,
            data_root=data_root,
            split=split,
            processor=self.processor,
        )
        collator = Wav2Vec2DataCollator(processor=self.processor)
        loader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=collator, num_workers=2
        )

        accumulator = MetricsAccumulator()
        all_pairs: list[dict] = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Wav2Vec2 eval [{split}]"):
                input_values = batch["input_values"].to(self.device)
                labels = batch["labels"]
                refs = batch.get("transcriptions", [])

                t0 = time.perf_counter()
                logits = self.model(input_values).logits
                elapsed = time.perf_counter() - t0

                pred_ids = torch.argmax(logits, dim=-1)
                hyps = self.processor.batch_decode(pred_ids)

                if not refs:
                    label_ids = labels.clone()
                    label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
                    refs = self.processor.tokenizer.batch_decode(
                        label_ids, group_tokens=False
                    )

                refs_norm = [normalize_text(r) for r in refs]
                hyps_norm = [normalize_text(h) for h in hyps]

                accumulator.update(refs_norm, hyps_norm, elapsed=elapsed)
                for ref, hyp in zip(refs_norm, hyps_norm):
                    all_pairs.append({"reference": ref, "hypothesis": hyp})

        metrics = accumulator.compute()
        error_analysis = analyse_errors(
            [p["reference"] for p in all_pairs],
            [p["hypothesis"] for p in all_pairs],
        )
        metrics["error_analysis"] = error_analysis
        metrics["predictions"] = all_pairs[:50]
        metrics["model_path"] = str(self.model_path)
        metrics["split"] = split

        log.info(
            "Wav2Vec2 eval complete — WER: %.4f  CER: %.4f  Samples: %d",
            metrics.get("wer", 0),
            metrics.get("cer", 0),
            metrics.get("n_samples", 0),
        )
        return metrics


# ─── Checkpoint Resolution ───────────────────────────────────────────────────

_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def _resolve_checkpoint(checkpoint_dir: Path, pretrained_fallback: str) -> str:
    """
    Return the best available model path for a fine-tuned experiment.

    Priority:
      1. checkpoint_dir itself (final saved model)
      2. Latest checkpoint-N/ subdirectory (training interrupted after a save)
      3. pretrained_fallback (training never saved anything useful)
    """
    def _has_weights(p: Path) -> bool:
        for w in _WEIGHT_FILES:
            f = p / w
            try:
                if f.exists() and f.stat().st_size > 0:
                    return True
            except OSError:
                pass
        return False

    if checkpoint_dir.exists():
        if _has_weights(checkpoint_dir):
            return str(checkpoint_dir)

        # Look for checkpoint-N subdirectories
        subdirs = sorted(
            [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda d: int(d.name.split("-")[1]),
        )
        for sub in reversed(subdirs):
            if _has_weights(sub):
                log.warning(
                    "Final model missing in %s — using sub-checkpoint %s",
                    checkpoint_dir, sub.name,
                )
                return str(sub)

        log.warning(
            "No model weights found in %s or its sub-checkpoints — falling back to pretrained.",
            checkpoint_dir,
        )
    else:
        log.warning("Checkpoint dir not found: %s — falling back to pretrained.", checkpoint_dir)

    return pretrained_fallback


# ─── Experiment Evaluator ────────────────────────────────────────────────────

class ExperimentEvaluator:
    """
    Run the full evaluation matrix across all trained experiments and datasets.

    Generates a results summary that feeds directly into the visualisation module.
    """

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.results_dir = Path(cfg.evaluation.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _eval_datasets(self) -> dict:
        """Build eval dataset paths from config so Drive paths are respected."""
        return {
            "real": {
                "manifest": str(Path(self.cfg.data.real_data_dir) / "test_manifest.json"),
                "data_root": self.cfg.data.real_data_dir,
            },
            "simulated": {
                "manifest": str(Path(self.cfg.data.simulated_data_dir) / "test_manifest.json"),
                "data_root": self.cfg.data.simulated_data_dir,
            },
        }

    def run_experiment(self, experiment: dict[str, Any]) -> dict[str, Any]:
        """
        Evaluate a single experiment entry from config.experiments.

        Returns results keyed by eval dataset name.
        """
        exp_name = experiment["name"]
        model_type = experiment["model"]
        eval_datasets = experiment.get("eval_data", ["real", "simulated"])

        # Resolve model checkpoint path
        checkpoint_dir = Path(self.cfg.training.output_dir) / exp_name
        if experiment.get("train_data") is None:
            # Baseline: use pretrained model from experiment config
            model_path = experiment["model_size"]
        else:
            model_path = _resolve_checkpoint(checkpoint_dir, experiment["model_size"])

        # Build evaluator — fall back to pretrained if the checkpoint is corrupt
        pretrained = experiment["model_size"]
        def _make_evaluator(path: str):
            if model_type == "whisper":
                return WhisperEvaluator(path, self.cfg, self.device)
            return Wav2Vec2Evaluator(path, self.cfg, self.device)

        try:
            evaluator = _make_evaluator(model_path)
        except OSError as exc:
            if model_path == pretrained:
                raise
            log.warning(
                "Failed to load checkpoint %s (%s) — falling back to pretrained %s",
                model_path, exc, pretrained,
            )
            evaluator = _make_evaluator(pretrained)

        exp_results: dict[str, Any] = {
            "experiment": exp_name,
            "model": model_type,
            "model_size": experiment.get("model_size", ""),
            "method": experiment.get("method", "baseline"),
            "train_data": experiment.get("train_data"),
            "description": experiment.get("description", ""),
        }

        for ds_name in eval_datasets:
            ds_info = self._eval_datasets().get(ds_name)
            if ds_info is None:
                continue
            manifest = ds_info["manifest"]
            if not Path(manifest).exists():
                log.warning("Eval manifest not found: %s", manifest)
                continue

            try:
                metrics = evaluator.evaluate(
                    manifest_path=manifest,
                    data_root=ds_info["data_root"],
                    batch_size=self.cfg.evaluation.batch_size,
                )
                exp_results[f"eval_{ds_name}"] = metrics
            except Exception as exc:
                log.error("Evaluation failed for %s on %s: %s", exp_name, ds_name, exc)
                exp_results[f"eval_{ds_name}"] = {"error": str(exc)}

        # Save individual result
        result_path = self.results_dir / f"{exp_name}.json"
        save_results(exp_results, result_path)
        log.info("Results saved → %s", result_path)

        return exp_results

    def run_all(self) -> dict[str, Any]:
        """Run evaluation for every experiment defined in config."""
        all_results: dict[str, Any] = {}
        for experiment in self.cfg.experiments:
            exp_name = experiment["name"]
            result_path = self.results_dir / f"{exp_name}.json"
            if result_path.exists():
                log.info("SKIPPING %s — results already exist at %s", exp_name, result_path)
                with open(result_path) as f:
                    all_results[exp_name] = json.load(f)
                continue
            log.info("─" * 50)
            log.info("Evaluating: %s", exp_name)
            log.info("─" * 50)
            results = self.run_experiment(dict(experiment))
            all_results[exp_name] = results

        # Save aggregated results
        save_results(all_results, self.results_dir / "all_results.json")
        log.info("\n✅ All evaluations complete. Results → %s", self.results_dir)
        return all_results

    def build_summary_table(self, all_results: dict) -> list[dict]:
        """
        Flatten results into a list of rows for easy plotting/analysis.

        Each row: {experiment, model, train_data, eval_dataset, wer, cer, ...}
        """
        rows = []
        for exp_name, res in all_results.items():
            for key, val in res.items():
                if key.startswith("eval_") and isinstance(val, dict) and "wer" in val:
                    ds_name = key[len("eval_"):]
                    rows.append(
                        {
                            "experiment": exp_name,
                            "model": res.get("model", ""),
                            "train_data": res.get("train_data", "baseline"),
                            "eval_dataset": ds_name,
                            "wer": val.get("wer", None),
                            "cer": val.get("cer", None),
                            "loss": val.get("loss", None),
                            "n_samples": val.get("n_samples", 0),
                            "avg_inference_ms": val.get("avg_inference_ms", None),
                        }
                    )
        return rows
