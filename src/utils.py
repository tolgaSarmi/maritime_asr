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
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

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


