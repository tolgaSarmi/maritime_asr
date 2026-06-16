"""
src/visualization.py
══════════════════════════════════════════════════════════════════════════════
Dissertation figures for Maritime VHF ASR.

results/all_results.json has a flat key structure:
    {experiment_name}_{test_domain} → {"wer": float, "cer": float, "samples": int}

e.g.  ef_whisper_small_real_real        → EF-small trained on real, eval on real
      lora_whisper_small_combined_simulated → LoRA-small trained on combined, eval on sim

Public API
──────────
  load_results(results_dir)   → dict    raw JSON contents
  plot_all(results_dir)       → None    saves 6 PNG figures to results/figures/
  show_figures(results_dir)   → None    notebook: generate + display inline
  generate_all_figures(...)   → None    alias for plot_all (called by main.py)
══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.family":    "serif",
    "font.size":      11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi":     150,
    "savefig.dpi":    150,
    "savefig.bbox":   "tight",
})

log = logging.getLogger(__name__)

PALETTE = {
    "baseline":         "#6c757d",
    "encoder_freezing": "#2196F3",
    "lora":             "#FF9800",
    "real":             "#1565C0",
    "simulated":        "#E65100",
    "combined":         "#6A1B9A",
}

METHOD_LABEL = {
    "baseline":         "Baseline",
    "encoder_freezing": "Encoder Freezing",
    "lora":             "LoRA",
}


# ─── Public: load ─────────────────────────────────────────────────────────────

def _flatten(raw: dict) -> dict:
    """
    Normalise any supported all_results.json schema to the flat format:
        {experiment_name}_{test_domain} → {"wer": float, "cer": float, "samples": int}

    Handles two schemas:

    A – flat (already correct, written by rebuild scripts / test fixtures):
        "baseline_whisper_small_real": {"wer": 0.7577, ...}

    B – nested (written by ExperimentEvaluator.run_all):
        "baseline_whisper_small": {
            "eval_real":      {"wer": 0.7577, "cer": 0.45, "n_samples": 151, ...},
            "eval_simulated": {"wer": 0.6821, ...}
        }
    """
    if not raw:
        return raw

    # Detect schema by checking first value
    sample = next(iter(raw.values()))
    if not isinstance(sample, dict):
        return raw  # unexpected; return as-is and let callers handle it

    # Schema A: value already has "wer" at top level
    if "wer" in sample:
        return raw

    # Schema B: value has "eval_real" / "eval_simulated" sub-dicts
    flat: dict = {}
    for exp_name, exp_data in raw.items():
        if not isinstance(exp_data, dict):
            continue
        for domain in ("real", "simulated"):
            metrics = exp_data.get(f"eval_{domain}")
            if not isinstance(metrics, dict) or "wer" not in metrics:
                continue
            flat[f"{exp_name}_{domain}"] = {
                "wer":     metrics["wer"],
                "cer":     metrics.get("cer"),
                "samples": metrics.get("n_samples", metrics.get("samples", 0)),
            }
    return flat


def load_results(results_dir: str = "results") -> dict:
    """
    Read results/all_results.json and return a flat dict keyed by
    {experiment_name}_{test_domain} → {"wer": float, "cer": float, "samples": int}.

    Accepts both the nested format written by ExperimentEvaluator.run_all() and
    the flat format written by rebuild scripts.  Returns {} with a warning if the
    file is missing or unreadable.
    """
    path = Path(results_dir) / "all_results.json"
    if not path.exists():
        log.warning("all_results.json not found at %s — run: python main.py --mode eval_all", path)
        return {}
    try:
        with open(path) as fh:
            raw = json.load(fh)
        data = _flatten(raw)
        log.info("Loaded %d result entries from %s", len(data), path)
        return data
    except Exception as exc:
        log.warning("Could not read %s: %s", path, exc)
        return {}


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _wer(data: dict, key: str) -> float | None:
    """Return WER (proportion 0-1) for *key*, or None if missing."""
    entry = data.get(key)
    if not isinstance(entry, dict):
        return None
    v = entry.get("wer")
    return float(v) if v is not None else None


def _save(fig: plt.Figure, name: str, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    log.info("Saved → %s", path)


def _tight(fig: plt.Figure) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.tight_layout()


def _annotate_bars(ax: plt.Axes, bars, vals: list[float | None]) -> None:
    """Place WER % labels above each bar (skip None / zero-height bars)."""
    for bar, v in zip(bars, vals):
        if v is not None:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v * 100 + 0.6,
                f"{v * 100:.1f}%",
                ha="center", va="bottom", fontsize=8,
            )


def _annotate_hbars(ax: plt.Axes, bars, vals: list[float | None]) -> None:
    """Place WER % labels at right end of each horizontal bar."""
    for bar, v in zip(bars, vals):
        if v is not None:
            ax.text(
                v * 100 + 0.3,
                bar.get_y() + bar.get_height() / 2,
                f"{v * 100:.1f}%",
                va="center", fontsize=8,
            )


# ─── Figure 01: Baseline Comparison ──────────────────────────────────────────

def _fig01_baseline(data: dict, figures_dir: Path) -> bool:
    """Grouped bar: zero-shot WER on real vs simulated test sets."""
    models = [
        ("baseline_whisper_small",  "Whisper\nSmall"),
        ("baseline_whisper_medium", "Whisper\nMedium"),
        ("baseline_wav2vec2",       "Wav2Vec2"),
    ]
    real_wers = [_wer(data, f"{k}_real")      for k, _ in models]
    sim_wers  = [_wer(data, f"{k}_simulated") for k, _ in models]

    if all(v is None for v in real_wers + sim_wers):
        log.warning("Fig 01: no baseline data — skipping")
        return False

    labels = [lbl for _, lbl in models]
    x, w = np.arange(len(labels)), 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    br = ax.bar(x - w/2,
                [v * 100 if v is not None else np.nan for v in real_wers],
                w, label="Real test set",      color=PALETTE["real"],      alpha=0.85)
    bs = ax.bar(x + w/2,
                [v * 100 if v is not None else np.nan for v in sim_wers],
                w, label="Simulated test set", color=PALETTE["simulated"], alpha=0.85)
    _annotate_bars(ax, br, real_wers)
    _annotate_bars(ax, bs, sim_wers)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Word Error Rate (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Zero-Shot Baseline Performance on Maritime VHF Speech")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _tight(fig)
    _save(fig, "01_baseline_comparison", figures_dir)
    return True


# ─── Figure 02: PEFT Method Comparison ───────────────────────────────────────

def _fig02_peft(data: dict, figures_dir: Path) -> bool:
    """EF vs LoRA for Whisper Small and Medium, trained on real, eval on real."""
    sizes   = ["small", "medium"]
    methods = [
        ("baseline",         "baseline_whisper_{size}_real"),
        ("encoder_freezing", "ef_whisper_{size}_real_real"),
        ("lora",             "lora_whisper_{size}_real_real"),
    ]

    any_data = any(
        _wer(data, tmpl.format(size=s)) is not None
        for _, tmpl in methods for s in sizes
    )
    if not any_data:
        log.warning("Fig 02: no PEFT comparison data — skipping")
        return False

    x, w = np.arange(len(sizes)), 0.25
    offsets = [-1, 0, 1]

    fig, ax = plt.subplots(figsize=(9, 5))
    for (method, tmpl), offset in zip(methods, offsets):
        vals = [_wer(data, tmpl.format(size=s)) for s in sizes]
        bars = ax.bar(
            x + offset * w,
            [v * 100 if v is not None else np.nan for v in vals],
            w,
            label=METHOD_LABEL[method],
            color=PALETTE[method], alpha=0.85,
        )
        _annotate_bars(ax, bars, vals)

    ax.set_xticks(x)
    ax.set_xticklabels(["Whisper Small", "Whisper Medium"])
    ax.set_ylabel("Word Error Rate (%) — Real Test Set")
    ax.set_title("PEFT Methods vs Baseline\n(Trained on Real Data, Evaluated on Real Test Set)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)
    _tight(fig)
    _save(fig, "02_peft_comparison_whisper", figures_dir)
    return True


# ─── Figure 02b: Wav2Vec2 PEFT Comparison ────────────────────────────────────

def _fig02b_wav2vec2(data: dict, figures_dir: Path) -> bool:
    """Wav2Vec2: baseline reference + EF vs LoRA across train conditions, real eval."""
    train_conditions = ["real", "combined"]
    methods = [
        ("ef",   "Encoder Freezing", PALETTE["encoder_freezing"]),
        ("lora", "LoRA",             PALETTE["lora"]),
    ]

    baseline_wer = _wer(data, "baseline_wav2vec2_real")
    any_data = any(
        _wer(data, f"{pfx}_wav2vec2_{tc}_real") is not None
        for pfx, _, _ in methods for tc in train_conditions
    )
    if not any_data and baseline_wer is None:
        log.warning("Fig 02b: no wav2vec2 data — skipping")
        return False

    x, w = np.arange(len(train_conditions)), 0.3
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (pfx, label, colour) in enumerate(methods):
        vals = [_wer(data, f"{pfx}_wav2vec2_{tc}_real") for tc in train_conditions]
        bars = ax.bar(
            x + (i - 0.5) * w,
            [v * 100 if v is not None else np.nan for v in vals],
            w, label=label, color=colour, alpha=0.85,
        )
        _annotate_bars(ax, bars, vals)

    if baseline_wer is not None:
        ax.axhline(
            baseline_wer * 100, color=PALETTE["baseline"],
            linestyle="--", linewidth=1.5,
            label=f"Baseline ({baseline_wer * 100:.1f}%)",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Train: {tc.title()}" for tc in train_conditions])
    ax.set_ylabel("Word Error Rate (%) — Real Test Set")
    ax.set_title("Wav2Vec2 — Encoder Freezing vs LoRA\n(Real Test Set)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)
    _tight(fig)
    _save(fig, "02_peft_comparison_wav2vec2", figures_dir)
    return True


# ─── Figure 03: Data Condition Comparison ────────────────────────────────────

def _fig03_data_condition(data: dict, figures_dir: Path) -> bool:
    """EF Whisper Small: real/sim/combined training × real/sim eval."""
    train_conditions = ["real", "simulated", "combined"]
    test_sets = [
        ("real",      PALETTE["real"]),
        ("simulated", PALETTE["simulated"]),
    ]

    any_data = any(
        _wer(data, f"ef_whisper_small_{tc}_{ts}") is not None
        for tc in train_conditions for ts, _ in test_sets
    )
    if not any_data:
        log.warning("Fig 03: no data condition data — skipping")
        return False

    x, w = np.arange(len(train_conditions)), 0.35
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (ts, colour) in enumerate(test_sets):
        vals = [_wer(data, f"ef_whisper_small_{tc}_{ts}") for tc in train_conditions]
        bars = ax.bar(
            x + (i - 0.5) * w,
            [v * 100 if v is not None else np.nan for v in vals],
            w,
            label=f"Eval: {ts.title()} test set",
            color=colour, alpha=0.85,
        )
        _annotate_bars(ax, bars, vals)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Train: {tc.title()}" for tc in train_conditions])
    ax.set_ylabel("Word Error Rate (%)")
    ax.set_title("Encoder Freezing (Whisper Small)\nEffect of Training Data Condition")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)
    _tight(fig)
    _save(fig, "03_data_condition_comparison", figures_dir)
    return True


# ─── Figure 04: Cross-Domain Heatmap ─────────────────────────────────────────

def _fig04_heatmap(data: dict, figures_dir: Path) -> bool:
    """3×2 heatmap (train condition × test set) for EF Small and LoRA Small."""
    train_conditions = ["real", "simulated", "combined"]
    test_sets        = ["real", "simulated"]
    panels = [
        ("ef",   "Encoder Freezing"),
        ("lora", "LoRA"),
    ]

    any_data = any(
        _wer(data, f"{pfx}_whisper_small_{tc}_{ts}") is not None
        for pfx, _ in panels
        for tc in train_conditions for ts in test_sets
    )
    if not any_data:
        log.warning("Fig 04: no cross-domain heatmap data — skipping")
        return False

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, (pfx, label) in zip(axes, panels):
        matrix = np.full((len(train_conditions), len(test_sets)), np.nan)
        for i, tc in enumerate(train_conditions):
            for j, ts in enumerate(test_sets):
                v = _wer(data, f"{pfx}_whisper_small_{tc}_{ts}")
                if v is not None:
                    matrix[i, j] = v * 100

        if np.all(np.isnan(matrix)):
            ax.set_title(f"{label} — no data yet")
            ax.axis("off")
            continue

        vmin = max(0, np.nanmin(matrix) - 5)
        vmax = min(100, np.nanmax(matrix) + 5)
        im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=vmin, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax, label="WER (%)")

        ax.set_xticks(range(len(test_sets)))
        ax.set_yticks(range(len(train_conditions)))
        ax.set_xticklabels([f"Eval: {ts.title()}" for ts in test_sets])
        ax.set_yticklabels([f"Train: {tc.title()}" for tc in train_conditions])

        for i in range(len(train_conditions)):
            for j in range(len(test_sets)):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.1f}%",
                            ha="center", va="center",
                            color="black", fontsize=11, fontweight="bold")
                else:
                    ax.text(j, i, "—", ha="center", va="center",
                            color="#aaaaaa", fontsize=11)

        ax.set_title(f"{label} — Whisper Small")

    fig.suptitle("Cross-Domain Generalisation Heatmap (WER %)", fontsize=13)
    _tight(fig)
    _save(fig, "04_cross_domain_heatmap", figures_dir)
    return True


# ─── Figure 05: Summary — All Experiments on Real Test Set ───────────────────

def _fig05_summary(data: dict, figures_dir: Path) -> bool:
    """Horizontal bar chart: every experiment ranked by WER on real test set."""
    entries: list[tuple[str, float]] = []
    for key, val in data.items():
        if not key.endswith("_real"):
            continue
        if not isinstance(val, dict) or "wer" not in val:
            continue
        wer = val["wer"]
        if wer is not None:
            exp_name = key[:-5]          # strip trailing _real
            entries.append((exp_name, float(wer)))

    if not entries:
        log.warning("Fig 05: no summary data — skipping")
        return False

    entries.sort(key=lambda e: e[1])     # best WER at top

    labels = [e[0].replace("_", " ") for e in entries]
    vals   = [e[1] for e in entries]

    def _colour(name: str) -> str:
        if name.startswith("baseline"):
            return PALETTE["baseline"]
        if name.startswith("ef"):
            return PALETTE["encoder_freezing"]
        if name.startswith("lora"):
            return PALETTE["lora"]
        return "#888888"

    colours = [_colour(e[0]) for e in entries]

    height = max(5, len(entries) * 0.45)
    fig, ax = plt.subplots(figsize=(11, height))
    y    = np.arange(len(entries))
    bars = ax.barh(y, [v * 100 for v in vals], color=colours, alpha=0.85, height=0.65)
    _annotate_hbars(ax, bars, vals)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Word Error Rate (%) — Real Test Set")
    ax.set_title("All Experiments Ranked by WER on Real Test Set")
    ax.set_xlim(0, max(v * 100 for v in vals) * 1.18)
    ax.grid(axis="x", alpha=0.3)

    patches = [
        mpatches.Patch(color=PALETTE["baseline"],         label="Baseline (zero-shot)"),
        mpatches.Patch(color=PALETTE["encoder_freezing"], label="Encoder Freezing"),
        mpatches.Patch(color=PALETTE["lora"],             label="LoRA"),
    ]
    ax.legend(handles=patches, fontsize=9, loc="lower right")
    _tight(fig)
    _save(fig, "05_summary", figures_dir)
    return True


# ─── Figure 06: Model Size Effect ────────────────────────────────────────────

def _fig06_model_size(data: dict, figures_dir: Path) -> bool:
    """Small vs Medium Whisper for EF and LoRA across train conditions (real eval)."""
    panels = [
        ("ef",   "Encoder Freezing"),
        ("lora", "LoRA"),
    ]
    sizes            = ["small", "medium"]
    train_conditions = ["real", "simulated", "combined"]
    size_colours     = {"small": "#42A5F5", "medium": "#1565C0"}

    any_data = any(
        _wer(data, f"{pfx}_whisper_{sz}_{tc}_real") is not None
        for pfx, _ in panels for sz in sizes for tc in train_conditions
    )
    if not any_data:
        log.warning("Fig 06: no model size data — skipping")
        return False

    x, w = np.arange(len(train_conditions)), 0.35
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    for ax, (pfx, label) in zip(axes, panels):
        for i, sz in enumerate(sizes):
            vals = [_wer(data, f"{pfx}_whisper_{sz}_{tc}_real") for tc in train_conditions]
            bars = ax.bar(
                x + (i - 0.5) * w,
                [v * 100 if v is not None else np.nan for v in vals],
                w,
                label=f"Whisper {sz.title()}",
                color=size_colours[sz], alpha=0.85,
            )
            _annotate_bars(ax, bars, vals)

        ax.set_xticks(x)
        ax.set_xticklabels([f"Train: {tc.title()}" for tc in train_conditions])
        ax.set_ylabel("WER (%) — Real Test Set")
        ax.set_title(f"{label}: Small vs Medium")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 100)

    fig.suptitle("Model Size Effect — Whisper Small vs Medium (Real Test Set)", fontsize=13)
    _tight(fig)
    _save(fig, "06_model_size", figures_dir)
    return True


# ─── Public: generate all figures ─────────────────────────────────────────────

def plot_all(results_dir: str = "results") -> None:
    """
    Generate all 6 dissertation figures from results/all_results.json.
    Saves PNGs to results/figures/.  Each figure is skipped gracefully if its
    required keys are absent from the JSON.
    """
    data = load_results(results_dir)
    if not data:
        log.error("No results loaded — run:  python main.py --mode eval_all")
        return

    figures_dir = Path(results_dir) / "figures"
    plotters = [
        _fig01_baseline,
        _fig02_peft,
        _fig02b_wav2vec2,
        _fig03_data_condition,
        _fig04_heatmap,
        _fig05_summary,
        _fig06_model_size,
    ]
    generated = sum(1 for fn in plotters if fn(data, figures_dir))
    log.info("Generated %d / %d figures → %s", generated, len(plotters), figures_dir)


# Alias kept for backward compatibility — main.py imports generate_all_figures
generate_all_figures = plot_all


def show_figures(results_dir: str = "results") -> None:
    """
    Notebook helper: generate figures then display them inline with IPython.
    Sets %matplotlib inline when running inside Jupyter / Colab.
    """
    try:
        ip = get_ipython()          # type: ignore[name-defined]
        if ip is not None:
            ip.run_line_magic("matplotlib", "inline")
    except NameError:
        pass

    plot_all(results_dir)

    figures_dir = Path(results_dir) / "figures"
    try:
        from IPython.display import Image, display  # type: ignore[import]
        for p in sorted(figures_dir.glob("0*.png")):
            print(f"\n── {p.name} ──")
            display(Image(str(p)))
    except ImportError:
        log.warning("IPython not available — figures saved to %s", figures_dir)
