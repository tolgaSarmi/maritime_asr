"""
src/visualization.py
══════════════════════════════════════════════════════════════════════════════
All dissertation figures:
  • WER comparison bar charts (baseline vs fine-tuned)
  • Real vs Simulated vs Combined training comparison
  • Cross-domain heatmap (train condition × eval dataset)
  • Training loss / WER curves
  • Error analysis (deletion / substitution / insertion breakdown)
  • Per-sample WER distribution
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

log = logging.getLogger(__name__)
FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Consistent colour palette
PALETTE = {
    "baseline": "#6c757d",
    "encoder_freezing": "#2196F3",
    "lora": "#FF9800",
    "full_finetuning": "#4CAF50",
    "real": "#1565C0",
    "simulated": "#E65100",
    "combined": "#6A1B9A",
    "whisper": "#0288D1",
    "wav2vec2": "#00897B",
}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, name: str) -> Path:
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    log.info("Saved figure → %s", path)
    return path


def load_results(results_dir: str | Path = "results") -> pd.DataFrame:
    """
    Load all results JSON files and flatten into a tidy DataFrame.
    Each row = one (experiment, eval_dataset) combination.
    """
    rows = []
    results_dir = Path(results_dir)
    for p in sorted(results_dir.glob("*.json")):
        if p.name == "all_results.json":
            try:
                data = json.loads(p.read_text())
                for exp_name, exp_data in data.items():
                    for key, val in exp_data.items():
                        if key.startswith("eval_") and isinstance(val, dict):
                            rows.append({
                                "experiment": exp_name,
                                "model": exp_data.get("model", ""),
                                "model_size": exp_data.get("model_size", ""),
                                "method": exp_data.get("method", "baseline"),
                                "train_data": exp_data.get("train_data", "none"),
                                "eval_dataset": key[len("eval_"):],
                                "wer": val.get("wer"),
                                "cer": val.get("cer"),
                                "n_samples": val.get("n_samples", 0),
                            })
            except Exception as e:
                log.warning("Could not parse %s: %s", p, e)
    if not rows:
        log.warning("No results found in %s", results_dir)
    return pd.DataFrame(rows)


# ─── Figure 1: Baseline Zero-Shot Comparison ─────────────────────────────────

def plot_baseline_comparison(df: pd.DataFrame) -> Path:
    """
    Reproduce and extend last year's Figure 20.
    Bar chart of zero-shot WER for all model variants on both test sets.
    """
    base = df[df["method"] == "baseline"].copy()
    if base.empty:
        log.warning("No baseline results to plot.")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for ax, ds in zip(axes, ["real", "simulated"]):
        sub = base[base["eval_dataset"] == ds].sort_values("wer")
        if sub.empty:
            ax.set_title(f"No data for {ds}")
            continue
        bars = ax.bar(
            range(len(sub)), sub["wer"] * 100,
            color=[PALETTE.get(sub.iloc[i]["model"], "#888") for i in range(len(sub))],
        )
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(
            [e.replace("baseline_", "").replace("_", "\n") for e in sub["experiment"]],
            fontsize=8,
        )
        ax.set_ylabel("WER (%)")
        ax.set_title(f"Zero-Shot Baseline — {ds.title()} Test Set")
        ax.set_ylim(0, 100)
        for bar, val in zip(bars, sub["wer"] * 100):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Zero-Shot Performance of Pretrained ASR Models on Maritime VHF", fontsize=13)
    plt.tight_layout()
    return _save(fig, "01_baseline_comparison")


# ─── Figure 2: PEFT Method Comparison (per model size) ───────────────────────

def plot_peft_comparison(df: pd.DataFrame, model_prefix: str = "whisper") -> Path:
    """
    For each Whisper model size, show all training methods side-by-side.
    Mirrors last year's Figure 22 but adds full_finetuning bars.
    """
    sub = df[
        (df["model"] == model_prefix) &
        (df["eval_dataset"] == "real") &
        (df["train_data"].isin(["real", "none"]))
    ].copy()

    if sub.empty:
        log.warning("No data for peft_comparison (%s)", model_prefix)
        return None

    methods = ["baseline", "encoder_freezing", "lora", "full_finetuning"]
    sizes = sorted(sub["model_size"].unique(),
                   key=lambda s: ("small" in s, "medium" in s, "large" in s))

    x = np.arange(len(sizes))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, method in enumerate(methods):
        vals = []
        for size in sizes:
            row = sub[(sub["model_size"] == size) & (sub["method"] == method)]
            vals.append(row["wer"].values[0] * 100 if not row.empty else np.nan)
        bars = ax.bar(
            x + i * width, vals, width,
            label=method.replace("_", " ").title(),
            color=PALETTE.get(method, "#aaa"),
        )
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.3, f"{val:.1f}",
                    ha="center", va="bottom", fontsize=8,
                )

    size_labels = [s.split("/")[-1].replace("-", "\n") for s in sizes]
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(size_labels)
    ax.set_ylabel("WER (%)")
    ax.set_title(
        f"{model_prefix.title()} Models — Baseline vs PEFT Methods "
        f"(Real Test Set, trained on Real Data)"
    )
    ax.legend()
    ax.set_ylim(0, max(v for v in [b for b in [vals] for v in b if not (isinstance(v, float) and np.isnan(v))] + [10]) * 1.15)
    plt.tight_layout()
    return _save(fig, f"02_peft_comparison_{model_prefix}")


# ─── Figure 3: Real vs Simulated vs Combined Training ────────────────────────

def plot_training_data_comparison(df: pd.DataFrame) -> Path:
    """
    Core novel contribution plot.
    For each model+method, compare training on real / simulated / combined.
    Evaluated on BOTH real and simulated test sets (2×2 grid).
    """
    methods = ["encoder_freezing", "lora"]
    eval_datasets = ["real", "simulated"]

    fig, axes = plt.subplots(
        len(methods), len(eval_datasets),
        figsize=(14, 8), sharey=False,
    )
    if len(methods) == 1:
        axes = [axes]

    for row_idx, method in enumerate(methods):
        for col_idx, eval_ds in enumerate(eval_datasets):
            ax = axes[row_idx][col_idx]

            sub = df[
                (df["method"] == method) &
                (df["eval_dataset"] == eval_ds) &
                (df["train_data"].isin(["real", "simulated", "combined"]))
            ].copy()

            if sub.empty:
                ax.set_title(f"No data: {method}/{eval_ds}")
                continue

            experiments = sub["experiment"].unique()
            x = np.arange(len(experiments))
            colors = [PALETTE.get(sub[sub["experiment"] == e]["train_data"].values[0], "#888")
                      for e in experiments]
            vals = [sub[sub["experiment"] == e]["wer"].values[0] * 100
                    for e in experiments]

            bars = ax.bar(x, vals, color=colors, width=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(
                [e.replace(method + "_", "").replace("_", "\n") for e in experiments],
                fontsize=7,
            )
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.3, f"{val:.1f}",
                    ha="center", va="bottom", fontsize=7,
                )
            ax.set_ylabel("WER (%)")
            ax.set_title(
                f"{method.replace('_', ' ').title()}\n"
                f"Eval: {eval_ds.title()} Test Set"
            )

    # Shared legend
    legend_patches = [
        mpatches.Patch(color=PALETTE["real"], label="Train: Real"),
        mpatches.Patch(color=PALETTE["simulated"], label="Train: Simulated"),
        mpatches.Patch(color=PALETTE["combined"], label="Train: Combined"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=10)
    fig.suptitle(
        "Real vs Simulated vs Combined Training Data\n"
        "Evaluated on Both Real and Simulated Test Sets",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    return _save(fig, "03_data_condition_comparison")


# ─── Figure 4: Cross-Domain Heatmap ──────────────────────────────────────────

def plot_cross_domain_heatmap(df: pd.DataFrame, method: str = "lora") -> Path:
    """
    Heatmap: rows = train data condition, cols = eval dataset.
    Shows whether simulated training generalises to real data and vice versa.
    This is one of the most academically interesting plots.
    """
    sub = df[(df["method"] == method) & (df["model"] == "whisper")].copy()
    if sub.empty:
        log.warning("No data for cross_domain heatmap (method=%s)", method)
        return None

    # Pick the largest model only for clarity
    large = [s for s in sub["model_size"].unique() if "large" in s]
    if large:
        sub = sub[sub["model_size"] == large[0]]

    train_conditions = ["real", "simulated", "combined"]
    eval_datasets = ["real", "simulated"]

    matrix = np.full((len(train_conditions), len(eval_datasets)), np.nan)
    for i, train_c in enumerate(train_conditions):
        for j, eval_ds in enumerate(eval_datasets):
            row = sub[
                (sub["train_data"] == train_c) &
                (sub["eval_dataset"] == eval_ds)
            ]
            if not row.empty:
                matrix[i, j] = row["wer"].values[0] * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=20, vmax=70, aspect="auto")
    plt.colorbar(im, ax=ax, label="WER (%)")

    ax.set_xticks(range(len(eval_datasets)))
    ax.set_yticks(range(len(train_conditions)))
    ax.set_xticklabels([f"Eval: {d.title()}" for d in eval_datasets])
    ax.set_yticklabels([f"Train: {c.title()}" for c in train_conditions])

    for i in range(len(train_conditions)):
        for j in range(len(eval_datasets)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.1f}%",
                        ha="center", va="center",
                        color="black", fontsize=12, fontweight="bold")

    ax.set_title(
        f"Cross-Domain Generalisation Heatmap\n"
        f"Whisper-large | {method.replace('_', ' ').title()}"
    )
    plt.tight_layout()
    return _save(fig, f"04_cross_domain_heatmap_{method}")


# ─── Figure 5: Model Comparison (Whisper vs Wav2Vec2) ────────────────────────

def plot_model_comparison(df: pd.DataFrame) -> Path:
    """Compare Whisper vs Wav2Vec2 across training conditions."""
    sub = df[
        (df["method"].isin(["encoder_freezing", "lora"])) &
        (df["eval_dataset"] == "real")
    ].copy()

    if sub.empty:
        log.warning("No data for model comparison")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, method in zip(axes, ["encoder_freezing", "lora"]):
        m_sub = sub[sub["method"] == method]
        models = m_sub["model"].unique()
        x = np.arange(len(models))
        width = 0.25
        for k, train_d in enumerate(["real", "simulated", "combined"]):
            vals = []
            for model in models:
                row = m_sub[(m_sub["model"] == model) & (m_sub["train_data"] == train_d)]
                vals.append(row["wer"].values[0] * 100 if not row.empty else np.nan)
            ax.bar(
                x + k * width, vals, width,
                label=f"Train: {train_d.title()}",
                color=PALETTE.get(train_d, "#888"),
            )
        ax.set_xticks(x + width)
        ax.set_xticklabels(models, fontsize=9)
        ax.set_ylabel("WER (%)")
        ax.set_title(f"{method.replace('_', ' ').title()}\n(Real Test Set)")
        ax.legend(fontsize=8)

    fig.suptitle("Whisper vs Wav2Vec2 Across Training Conditions", fontsize=13)
    plt.tight_layout()
    return _save(fig, "05_model_comparison")


# ─── Figure 6: Training Curves ────────────────────────────────────────────────

def plot_training_curves(
    train_losses: list[float],
    val_wers: list[float],
    val_losses: list[float],
    experiment_name: str,
    eval_steps: int = 100,
) -> Path:
    """
    Plot training loss + validation WER + validation loss on shared x-axis.
    Called automatically at end of each training run.
    """
    steps = list(range(0, len(val_wers) * eval_steps, eval_steps))
    train_steps = list(range(len(train_losses)))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    ax1.plot(train_steps, train_losses, color="#1565C0", linewidth=1.5)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")

    ax2.plot(steps[:len(val_wers)], [v * 100 for v in val_wers],
             color="#E65100", marker="o", markersize=3, linewidth=1.5)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("WER (%)")
    ax2.set_title("Validation WER")

    ax3.plot(steps[:len(val_losses)], val_losses,
             color="#6A1B9A", marker="s", markersize=3, linewidth=1.5)
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Loss")
    ax3.set_title("Validation Loss")

    fig.suptitle(f"Training Dynamics — {experiment_name}", fontsize=12)
    plt.tight_layout()
    return _save(fig, f"training_curves_{experiment_name}")


# ─── Figure 7: Error Analysis ────────────────────────────────────────────────

def plot_error_breakdown(error_analyses: dict[str, dict]) -> Path:
    """
    Stacked bar chart of substitution / deletion / insertion rates
    for each experiment.
    """
    if not error_analyses:
        log.warning("No error analysis data provided.")
        return None

    experiments = list(error_analyses.keys())
    subs = [error_analyses[e].get("sub_rate", 0) * 100 for e in experiments]
    dels = [error_analyses[e].get("del_rate", 0) * 100 for e in experiments]
    ins = [error_analyses[e].get("ins_rate", 0) * 100 for e in experiments]

    x = np.arange(len(experiments))
    fig, ax = plt.subplots(figsize=(max(10, len(experiments) * 1.2), 5))

    ax.bar(x, subs, label="Substitutions", color="#EF5350")
    ax.bar(x, dels, bottom=subs, label="Deletions", color="#FFA726")
    ax.bar(x, [i + s + d for i, s, d in zip(ins, subs, dels)],
           bottom=[s + d for s, d in zip(subs, dels)],
           label="Insertions", color="#42A5F5")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [e.replace("_", "\n") for e in experiments], fontsize=7
    )
    ax.set_ylabel("Error Rate (%)")
    ax.set_title("Error Breakdown by Type per Experiment")
    ax.legend()
    plt.tight_layout()
    return _save(fig, "06_error_breakdown")


# ─── Figure 8: WER Distribution ──────────────────────────────────────────────

def plot_wer_distribution(
    per_sample_wers: dict[str, list[float]],
    top_n: int = 4,
) -> Path:
    """Box-plot / violin of per-sample WER distributions."""
    if not per_sample_wers:
        return None

    labels = list(per_sample_wers.keys())[:top_n]
    data = [
        [v * 100 for v in per_sample_wers[l]] for l in labels
    ]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2), 5))
    parts = ax.violinplot(data, positions=range(len(labels)), showmedians=True)

    for pc in parts["bodies"]:
        pc.set_alpha(0.7)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(
        [l.replace("_", "\n") for l in labels], fontsize=8
    )
    ax.set_ylabel("Per-Sample WER (%)")
    ax.set_title("WER Distribution per Experiment")
    plt.tight_layout()
    return _save(fig, "07_wer_distribution")


# ─── Master Plot Function ────────────────────────────────────────────────────

def generate_all_figures(results_dir: str = "results") -> None:
    """
    Load all results and generate every dissertation figure.
    Call after all experiments have completed.
    """
    log.info("Generating all dissertation figures...")
    df = load_results(results_dir)

    if df.empty:
        log.error("No results found in %s — run evaluations first.", results_dir)
        return

    plot_baseline_comparison(df)
    plot_peft_comparison(df, "whisper")
    plot_peft_comparison(df, "wav2vec2")
    plot_training_data_comparison(df)
    plot_cross_domain_heatmap(df, "encoder_freezing")
    plot_cross_domain_heatmap(df, "lora")
    plot_model_comparison(df)

    log.info("All figures saved to %s", FIGURES_DIR)
