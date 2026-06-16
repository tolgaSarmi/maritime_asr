"""
main.py
══════════════════════════════════════════════════════════════════════════════
Single entry point for the Maritime VHF ASR dissertation project.

Usage examples:

  # ── Data preparation ──────────────────────────────────────
  python main.py --mode data

  # ── Run all experiments ───────────────────────────────────
  python main.py --mode all

  # ── Run a single experiment ───────────────────────────────
  python main.py --mode train --experiment ef_whisper_large_real
  python main.py --mode train --experiment lora_whisper_large_combined

  # ── Evaluate a trained checkpoint ─────────────────────────
  python main.py --mode eval --experiment ef_whisper_large_real

  # ── Evaluate ALL experiments and save results ─────────────
  python main.py --mode eval_all

  # ── Generate all dissertation figures ─────────────────────
  python main.py --mode figures

  # ── Quick WER on a manifest ───────────────────────────────
  python main.py --mode test_wer \\
      --checkpoint checkpoints/ef_whisper_large_real \\
      --manifest data/real/test_manifest.json

  # ── Transcribe a single audio file ───────────────────────
  python main.py --mode transcribe \\
      --checkpoint checkpoints/lora_whisper_large_combined \\
      --audio path/to/file.wav
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf
from rich.console import Console
from rich.table import Table

console = Console()
log = logging.getLogger(__name__)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _load_cfg(config_path: str = "configs/config.yaml"):
    from src.utils import load_config, setup_logger, set_seed
    setup_logger(log_file="results/logs/main.log")
    cfg = load_config(config_path)
    set_seed(cfg.data.random_seed)
    return cfg


def _find_experiment(cfg, name: str) -> dict | None:
    for exp in cfg.experiments:
        if exp["name"] == name:
            return dict(exp)
    return None


def _print_experiments(cfg) -> None:
    table = Table(title="Available Experiments", show_lines=True)
    table.add_column("Name", style="cyan")
    table.add_column("Model")
    table.add_column("Method")
    table.add_column("Train Data")

    for exp in cfg.experiments:
        method = exp.get("method", "baseline")
        table.add_row(
            exp["name"],
            exp.get("model_size", exp.get("model", "")),
            method,
            str(exp.get("train_data", "—")),
        )
    console.print(table)


# ─── Modes ───────────────────────────────────────────────────────────────────

def mode_data(args):
    """Run the full data preparation pipeline."""
    from data_pipeline import run_full_pipeline
    run_full_pipeline(args.config)


def _is_completed(output_dir: Path) -> bool:
    """Return True only when training finished and metrics were written."""
    return (output_dir / "train_results.json").exists()


def _has_checkpoint(output_dir: Path) -> bool:
    """Return True when at least one HuggingFace checkpoint subdirectory exists."""
    if not output_dir.is_dir():
        return False
    return any(
        d.is_dir() and d.name.startswith("checkpoint-")
        for d in output_dir.iterdir()
    )


def mode_train(args):
    """Train a single experiment by name."""
    cfg = _load_cfg(args.config)

    if args.experiment is None:
        log.error("--experiment required for train mode")
        _print_experiments(cfg)
        sys.exit(1)

    experiment = _find_experiment(cfg, args.experiment)
    if experiment is None:
        log.error("Experiment '%s' not found in config.", args.experiment)
        _print_experiments(cfg)
        sys.exit(1)

    if experiment.get("method") == "baseline" or experiment.get("train_data") is None:
        log.info("'%s' is a baseline — skipping training.", args.experiment)
        return

    exp_output_dir = Path(cfg.training.output_dir) / args.experiment

    if not args.force and _is_completed(exp_output_dir):
        print(f"SKIPPING: {args.experiment} (already completed)")
        return

    if _has_checkpoint(exp_output_dir):
        log.info("Checkpoint found — training will resume from latest checkpoint.")

    from src.train import build_trainer
    trainer = build_trainer(cfg, experiment)
    metrics = trainer.train()

    results_path = Path(cfg.evaluation.results_dir) / f"train_{args.experiment}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    log.info("Training metrics → %s", results_path)


def mode_eval(args):
    """Evaluate a single experiment on both test sets."""
    cfg = _load_cfg(args.config)

    if args.experiment is None:
        log.error("--experiment required for eval mode")
        sys.exit(1)

    experiment = _find_experiment(cfg, args.experiment)
    if experiment is None:
        log.error("Experiment '%s' not found.", args.experiment)
        sys.exit(1)

    from src.evaluate import ExperimentEvaluator
    evaluator = ExperimentEvaluator(cfg)
    results = evaluator.run_experiment(experiment)

    from src.utils import print_results_table
    print_results_table({args.experiment: _flatten_results(results)})


def mode_eval_all(args):
    """Evaluate every experiment defined in config."""
    cfg = _load_cfg(args.config)

    from src.evaluate import ExperimentEvaluator
    evaluator = ExperimentEvaluator(cfg)
    all_results = evaluator.run_all()

    from src.utils import print_results_table, save_results
    rows = evaluator.build_summary_table(all_results)
    flat = {r["experiment"] + "_" + r["eval_dataset"]: r for r in rows}
    print_results_table(flat, metrics=["wer", "cer"])

    save_results(all_results, Path(cfg.evaluation.results_dir) / "all_results.json")
    log.info("\nNext: python main.py --mode figures")


def mode_train_all(args):
    """Train ALL non-baseline experiments sequentially."""
    cfg = _load_cfg(args.config)

    from src.train import build_trainer
    completed, failed = [], []

    for exp in cfg.experiments:
        exp = dict(exp)
        if exp.get("method") == "baseline" or exp.get("train_data") is None:
            continue

        exp_output_dir = Path(cfg.training.output_dir) / exp["name"]
        if not args.force and _is_completed(exp_output_dir):
            print(f"SKIPPING: {exp['name']} (already completed)")
            continue

        log.info("\n" + "═" * 60)
        log.info("Starting: %s", exp["name"])
        log.info("═" * 60)
        try:
            trainer = build_trainer(cfg, exp)
            trainer.train()
            completed.append(exp["name"])
        except Exception as exc:
            log.error("FAILED: %s — %s", exp["name"], exc)
            failed.append(exp["name"])

    log.info("\n" + "═" * 60)
    log.info("Training complete: %d succeeded, %d failed", len(completed), len(failed))
    if failed:
        log.warning("Failed: %s", failed)


def mode_figures(args):
    """Generate all dissertation figures from saved results."""
    from src.visualization import plot_all
    cfg = _load_cfg(args.config)
    plot_all(cfg.evaluation.results_dir)
    log.info("Figures saved to results/figures/")


def mode_test_wer(args):
    """Quick WER evaluation on a manifest using a checkpoint."""
    if not args.checkpoint:
        log.error("--checkpoint required")
        sys.exit(1)
    if not args.manifest:
        log.error("--manifest required")
        sys.exit(1)

    cfg = _load_cfg(args.config)
    checkpoint = args.checkpoint
    manifest = args.manifest

    # Auto-detect model type from checkpoint
    model_type = "whisper"
    for f in Path(checkpoint).glob("*.json"):
        if "wav2vec" in f.read_text().lower():
            model_type = "wav2vec2"
            break

    if model_type == "whisper":
        from src.evaluate import WhisperEvaluator
        evaluator = WhisperEvaluator(checkpoint, cfg)
    else:
        from src.evaluate import Wav2Vec2Evaluator
        evaluator = Wav2Vec2Evaluator(checkpoint, cfg)

    data_root = str(Path(manifest).parent)
    results = evaluator.evaluate(manifest, data_root, split="test")

    console.print(f"\nWER  : [bold red]{results['wer']:.4f}[/bold red] ({results['wer']*100:.2f}%)")
    console.print(f"CER  : {results['cer']:.4f}")
    console.print(f"N    : {results['n_samples']} samples")
    console.print(f"\nSample predictions:")
    for p in results.get("predictions", [])[:5]:
        console.print(f"  REF: {p['reference']}")
        console.print(f"  HYP: {p['hypothesis']}")
        console.print()


def mode_transcribe(args):
    """Transcribe a single audio file."""
    if not args.checkpoint:
        log.error("--checkpoint required")
        sys.exit(1)
    if not args.audio:
        log.error("--audio required")
        sys.exit(1)

    cfg = _load_cfg(args.config)

    from src.inference import build_transcriber
    transcriber = build_transcriber(
        model_type=args.model_type or "whisper",
        model_path=args.checkpoint,
    )
    text, elapsed = transcriber.transcribe_file(args.audio)
    console.print(f"\n[bold green]Transcription:[/bold green] {text}")
    console.print(f"[dim]Inference time: {elapsed*1000:.0f}ms[/dim]")


def mode_list(args):
    """List all experiments defined in config."""
    cfg = _load_cfg(args.config)
    _print_experiments(cfg)


def mode_all(args):
    """Full pipeline: data → train all → eval all → figures."""
    log.info("Running full pipeline...")
    mode_data(args)
    mode_train_all(args)
    mode_eval_all(args)
    mode_figures(args)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _flatten_results(results: dict) -> dict:
    flat = {}
    for key, val in results.items():
        if isinstance(val, dict):
            for k2, v2 in val.items():
                flat[f"{key}_{k2}"] = v2
        else:
            flat[key] = val
    return flat


# ─── CLI ─────────────────────────────────────────────────────────────────────

MODES = {
    "data": mode_data,
    "train": mode_train,
    "train_all": mode_train_all,
    "eval": mode_eval,
    "eval_all": mode_eval_all,
    "figures": mode_figures,
    "test_wer": mode_test_wer,
    "transcribe": mode_transcribe,
    "list": mode_list,
    "all": mode_all,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Maritime VHF ASR Dissertation — Main Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--mode", choices=list(MODES.keys()), default="list",
                   help="Operation mode")
    p.add_argument("--config", default="configs/config.yaml",
                   help="Path to YAML config")
    p.add_argument("--experiment", default=None,
                   help="Experiment name (for train/eval modes)")
    p.add_argument("--checkpoint", default=None,
                   help="Path to model checkpoint (for test_wer/transcribe)")
    p.add_argument("--manifest", default=None,
                   help="Path to test manifest JSON (for test_wer)")
    p.add_argument("--audio", default=None,
                   help="Path to audio file (for transcribe mode)")
    p.add_argument("--model-type", dest="model_type", default=None,
                   choices=["whisper", "wav2vec2"],
                   help="Model type (for transcribe/test_wer)")
    p.add_argument("--force", action="store_true",
                   help="Force retrain even if checkpoint exists")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    MODES[args.mode](args)
