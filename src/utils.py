"""
src/utils.py
══════════════════════════════════════════════════════════════════════════════
Shared utilities: config management, logging, reproducibility, GPU helpers.
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from src.env import load_env_file

console = Console()


# ─── Logging ─────────────────────────────────────────────────────────────────

def setup_logger(
    name: str = "asr_dissertation",
    log_file: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure a rich-formatted logger with optional file handler."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(fh)

    return logger


log = setup_logger()


# ─── Configuration ───────────────────────────────────────────────────────────

def load_config(config_path: str | Path = "configs/config.yaml") -> DictConfig:
    """Load YAML config and merge with environment overrides."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    load_env_file()
    cfg = OmegaConf.load(config_path)

    # Allow env-var override of api key
    api_key_env = os.environ.get("LABEL_STUDIO_API_KEY", "")
    if api_key_env:
        OmegaConf.update(cfg, "label_studio.api_key", api_key_env)

    return cfg


def save_config(cfg: DictConfig, output_path: str | Path) -> None:
    """Persist config snapshot next to experiment outputs."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    log.info("Config saved → %s", output_path)


# ─── Reproducibility ─────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Fix random seeds across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    log.info("Random seed fixed to %d", seed)


# ─── Device Detection ────────────────────────────────────────────────────────

def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return the best available compute device."""
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        log.info(
            "GPU detected: %s  (VRAM: %.1f GB)",
            props.name,
            props.total_memory / 1e9,
        )
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("Apple Silicon MPS device detected")
    else:
        device = torch.device("cpu")
        log.warning("No GPU found — running on CPU (training will be slow)")
    return device


def gpu_info() -> dict[str, Any]:
    """Return a dict of GPU memory stats (bytes)."""
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


# ─── Checkpoint Helpers ──────────────────────────────────────────────────────

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    output_dir: str | Path,
    filename: str | None = None,
) -> Path:
    """Save model + optimizer state with metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = filename or f"checkpoint_epoch{epoch:03d}.pt"
    checkpoint_path = output_dir / filename

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        },
        checkpoint_path,
    )
    log.info("Checkpoint saved → %s", checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Load checkpoint and restore model/optimizer weights."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    map_location = device or torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=map_location)

    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    log.info(
        "Loaded checkpoint from epoch %d  (WER: %.4f)",
        ckpt.get("epoch", -1),
        ckpt.get("metrics", {}).get("wer", float("nan")),
    )
    return ckpt


def find_best_checkpoint(checkpoint_dir: str | Path, metric: str = "wer") -> Path | None:
    """Scan checkpoint directory and return the path with best metric."""
    checkpoint_dir = Path(checkpoint_dir)
    best_path = None
    best_val = float("inf")  # lower is better for WER

    for ckpt_file in sorted(checkpoint_dir.glob("*.pt")):
        try:
            ckpt = torch.load(ckpt_file, map_location="cpu")
            val = ckpt.get("metrics", {}).get(metric, float("inf"))
            if val < best_val:
                best_val = val
                best_path = ckpt_file
        except Exception:
            continue

    if best_path:
        log.info("Best checkpoint: %s  (%s=%.4f)", best_path.name, metric, best_val)
    return best_path


# ─── Timing ──────────────────────────────────────────────────────────────────

class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed = time.perf_counter() - self._start
        if self.name:
            log.info("%s: %.3f s", self.name, self.elapsed)


# ─── Results Persistence ─────────────────────────────────────────────────────

def save_results(results: dict[str, Any], output_path: str | Path) -> None:
    """Save experiment results as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Results saved → %s", output_path)


def load_results(results_path: str | Path) -> dict[str, Any]:
    """Load previously saved results JSON."""
    with open(results_path, encoding="utf-8") as f:
        return json.load(f)


def aggregate_results(results_dir: str | Path) -> dict[str, Any]:
    """Load all results/*.json and merge into one summary dict."""
    results_dir = Path(results_dir)
    aggregated: dict[str, Any] = {}
    for p in sorted(results_dir.glob("*.json")):
        try:
            data = load_results(p)
            aggregated[p.stem] = data
        except Exception as exc:
            log.warning("Could not load %s: %s", p, exc)
    return aggregated


# ─── Pretty Printing ─────────────────────────────────────────────────────────

def print_results_table(results: dict[str, dict], metrics: list[str] | None = None) -> None:
    """Render a rich table of experiment results to the terminal."""
    metrics = metrics or ["wer", "cer", "loss"]
    table = Table(title="Experiment Results", show_lines=True)
    table.add_column("Experiment", style="cyan", no_wrap=True)
    for m in metrics:
        table.add_column(m.upper(), justify="right")

    for exp_name, res in results.items():
        row = [exp_name]
        for m in metrics:
            val = res.get(m, res.get(f"eval_{m}", "N/A"))
            if isinstance(val, float):
                row.append(f"{val:.4f}")
            else:
                row.append(str(val))
        table.add_row(*row)

    console.print(table)


# ─── Miscellaneous ───────────────────────────────────────────────────────────

def ensure_dirs(*paths: str | Path) -> None:
    """Create multiple directories at once."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def format_number(n: int) -> str:
    """Format large int with M/K suffix."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def cleanup_old_checkpoints(checkpoint_dir: Path, keep: int = 3) -> None:
    """Keep only the `keep` most-recent checkpoints by modification time."""
    ckpts = sorted(checkpoint_dir.glob("checkpoint_epoch*.pt"), key=lambda p: p.stat().st_mtime)
    for old in ckpts[:-keep]:
        old.unlink()
        log.debug("Removed old checkpoint: %s", old.name)
