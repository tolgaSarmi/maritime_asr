"""
data_pipeline.py
══════════════════════════════════════════════════════════════════════════════
End-to-end data preparation pipeline.

Directly builds on last year's approach with key improvements:
  • pydub.silence parameters confirmed optimal by last year's grid search
    (silence_thresh=-35 dBFS, min_silence_len=1500ms, F1=0.9813, recall=1.0)
  • Whisper-small sentence-level segmentation (identical to last year)
  • Label Studio export via label_studio_export.py
  • Proper 80/10/10 train/val/test split — FIXED vs last year's tiny val set
  • Combined manifest builder for real + simulated training

Usage:
    # Step 1 — export annotated data from Label Studio
    python label_studio_export.py --api-key YOUR_KEY

    # Step 2 — run this pipeline
    python data_pipeline.py --mode all
    python data_pipeline.py --mode splits          # just re-split
    python data_pipeline.py --mode combine         # rebuild combined manifests
    python data_pipeline.py --mode stats           # print dataset statistics
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Paths ───────────────────────────────────────────────────────────────────
DATA_DIRS = {
    "real": Path("data/real"),
    "simulated": Path("data/simulated"),
    "combined": Path("data/combined"),
}
SEED = 42


# ─── Normalisation (matches last year's prep_ground_truth.py) ────────────────

def normalize_transcription(text: str) -> str:
    """
    Normalise transcription for WER computation.
    Rules match last year's implementation:
      • digit-by-digit expansion  (e.g. "16" → "one six")
      • lowercase
      • strip punctuation
      • collapse whitespace
    """
    import re
    import inflect  # pip install inflect

    if not isinstance(text, str) or not text.strip():
        return ""

    p = inflect.engine()

    def digits_to_words(match: re.Match) -> str:
        return " ".join(p.number_to_words(d) for d in match.group(0))

    text = re.sub(r"\d+", digits_to_words, text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─── Manifest Loading ────────────────────────────────────────────────────────

def load_manifest(manifest_path: Path) -> list[dict]:
    """Load a JSON or CSV manifest produced by label_studio_export.py."""
    if not manifest_path.exists():
        return []
    if manifest_path.suffix == ".csv":
        return pd.read_csv(manifest_path).to_dict(orient="records")
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def validate_manifest(records: list[dict], data_root: Path) -> list[dict]:
    """
    Validate records: drop entries with missing audio or empty transcription.

    Accepts two audio_file formats:
      • Local path  – resolved relative to data_root, checked with os.path.exists
      • Remote URL  – accepted as-is (https://...) — no local check needed
    """
    valid         = []
    missing_audio = 0
    empty_text    = 0

    for r in records:
        transcription = r.get("transcription", "").strip()
        if not transcription:
            empty_text += 1
            continue

        audio_file = r.get("audio_file", "")
        if not audio_file:
            missing_audio += 1
            continue

        # ── Remote URL (real dataset streamed from Azure) ─────────────────
        if str(audio_file).startswith(("http://", "https://")):
            r["normalized_transcription"] = normalize_transcription(transcription)
            valid.append(r)
            continue

        # ── Local path (simulated dataset downloaded to disk) ─────────────
        audio_path = Path(audio_file)
        if not audio_path.is_absolute():
            audio_path = data_root / audio_file

        if not audio_path.exists():
            audio_path = data_root / "audio" / Path(audio_file).name
            if not audio_path.exists():
                missing_audio += 1
                continue

        r["audio_file"] = str(audio_path)
        r["normalized_transcription"] = normalize_transcription(transcription)
        valid.append(r)

    log.info(
        "Validation: %d valid | %d missing audio | %d empty text",
        len(valid), missing_audio, empty_text,
    )
    return valid


# ─── Duration Helpers ────────────────────────────────────────────────────────

def get_duration(audio_path: str) -> float:
    """Return audio duration in seconds. Handles local paths and URLs."""
    if str(audio_path).startswith(("http://", "https://")):
        # For cloud URLs, load via preprocessing cache and measure length
        try:
            from src.preprocessing import load_audio, TARGET_SR  # noqa: PLC0415
            waveform, sr = load_audio(audio_path)
            return len(waveform) / sr
        except Exception:
            return 0.0
    try:
        import soundfile as sf
        with sf.SoundFile(audio_path) as f:
            return len(f) / f.samplerate
    except Exception:
        return 0.0


def add_durations(records: list[dict]) -> list[dict]:
    """Add duration_s field to each record (needed for stats)."""
    log.info("Computing audio durations...")
    for r in tqdm(records, desc="Durations", unit="file"):
        if "duration_s" not in r:
            r["duration_s"] = get_duration(r["audio_file"])
    return records


# ─── Dataset Splitting ───────────────────────────────────────────────────────

def split_records(
    records: list[dict],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = SEED,
) -> dict[str, list[dict]]:
    """
    Stratified train/val/test split.

    KEY IMPROVEMENT over last year:
    Last year had only ~10 minutes validation, causing wildly unstable WER
    metrics and premature early stopping. With the larger dataset we now get
    a proper validation set that produces reliable metrics.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    random.seed(seed)
    shuffled = records.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train: n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }

    for name, subset in splits.items():
        duration_min = sum(r.get("duration_s", 0) for r in subset) / 60
        log.info(
            "  %-6s: %4d samples  (~%.1f min)",
            name, len(subset), duration_min,
        )

    return splits


def save_splits(
    splits: dict[str, list[dict]],
    output_dir: Path,
) -> None:
    """Write train/val/test JSON manifests to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, records in splits.items():
        out_path = output_dir / f"{split_name}_manifest.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        log.info("Saved %s → %s", split_name, out_path)


# ─── Combined Dataset ────────────────────────────────────────────────────────

def build_combined(
    real_dir: Path = DATA_DIRS["real"],
    sim_dir: Path = DATA_DIRS["simulated"],
    output_dir: Path = DATA_DIRS["combined"],
) -> None:
    """
    Merge real and simulated split manifests into combined manifests.
    Preserves the data_type field so training can apply dataset-aware
    weighting or analysis.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        combined: list[dict] = []
        for source, src_dir in [("real", real_dir), ("simulated", sim_dir)]:
            p = src_dir / f"{split}_manifest.json"
            if p.exists():
                records = load_manifest(p)
                for r in records:
                    r["data_type"] = source  # ensure tag is present
                combined.extend(records)
                log.info("  [%s/%s] loaded %d records", source, split, len(records))

        out_path = output_dir / f"{split}_manifest.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        log.info("Combined %s: %d total → %s", split, len(combined), out_path)


# ─── Dataset Statistics ──────────────────────────────────────────────────────

def compute_stats(records: list[dict], label: str) -> dict:
    """Compute and print summary statistics for a record list."""
    if not records:
        log.warning("[%s] No records found.", label)
        return {}

    durations = [r.get("duration_s", 0.0) for r in records]
    texts = [r.get("normalized_transcription", r.get("transcription", "")) for r in records]
    word_counts = [len(t.split()) for t in texts if t]

    stats = {
        "n_samples": len(records),
        "total_minutes": round(sum(durations) / 60, 1),
        "mean_duration_s": round(float(np.mean(durations)), 2) if durations else 0,
        "mean_words": round(float(np.mean(word_counts)), 1) if word_counts else 0,
        "vocab_size": len(set(" ".join(texts).split())),
        "data_types": dict(Counter(r.get("data_type", "unknown") for r in records)),
    }

    print(f"\n{'─'*50}")
    print(f"  Dataset: {label}")
    print(f"  Samples      : {stats['n_samples']}")
    print(f"  Total audio  : {stats['total_minutes']} min")
    print(f"  Mean duration: {stats['mean_duration_s']} s")
    print(f"  Mean words   : {stats['mean_words']}")
    print(f"  Vocabulary   : {stats['vocab_size']} unique words")
    print(f"  Data types   : {stats['data_types']}")
    print(f"{'─'*50}")

    return stats


def print_all_stats(dirs: dict | None = None) -> None:
    """Print statistics for all datasets and splits."""
    _dirs = dirs if dirs is not None else DATA_DIRS
    for ds_name, ds_dir in _dirs.items():
        for split in ("train", "val", "test"):
            manifest = Path(ds_dir) / f"{split}_manifest.json"
            if manifest.exists():
                records = load_manifest(manifest)
                compute_stats(records, f"{ds_name}/{split}")


# ─── Full Pipeline ───────────────────────────────────────────────────────────

def run_full_pipeline(cfg_path: str = "configs/config.yaml") -> None:
    """
    End-to-end data preparation assuming Label Studio export already ran.

    Steps:
      1. Load manifest.json from data/real/ and data/simulated/
      2. Validate records (check audio files exist, transcriptions non-empty)
      3. Normalise transcriptions
      4. Add audio durations
      5. Split into train/val/test
      6. Save split manifests
      7. Build combined manifests
      8. Print statistics
    """
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(cfg_path)

    log.info("═" * 55)
    log.info("Maritime VHF ASR – Data Pipeline")
    log.info("═" * 55)

    # Use config-driven paths so all data on Drive persists across sessions
    cfg_dirs = {
        "real":      Path(cfg.data.real_data_dir),
        "simulated": Path(cfg.data.simulated_data_dir),
        "combined":  Path(cfg.data.combined_data_dir),
    }

    for ds_name in ("real", "simulated"):
        ds_dir = cfg_dirs[ds_name]
        manifest_path = ds_dir / "manifest.json"

        if not manifest_path.exists():
            log.warning(
                "No manifest found for '%s' at %s\n"
                "  → Run: python label_studio_export.py --api-key YOUR_KEY --project %s",
                ds_name, manifest_path, ds_name,
            )
            continue

        log.info("\n── Processing: %s ──", ds_name)
        records = load_manifest(manifest_path)
        log.info("  Loaded %d records from manifest", len(records))

        records = validate_manifest(records, ds_dir)
        records = add_durations(records)

        splits = split_records(
            records,
            train_ratio=cfg.data.train_ratio,
            val_ratio=cfg.data.val_ratio,
            test_ratio=cfg.data.test_ratio,
            seed=cfg.data.random_seed,
        )
        save_splits(splits, ds_dir)

    log.info("\n── Building combined dataset ──")
    build_combined(
        real_dir=cfg_dirs["real"],
        sim_dir=cfg_dirs["simulated"],
        output_dir=cfg_dirs["combined"],
    )

    log.info("\n── Dataset Statistics ──")
    print_all_stats(cfg_dirs)

    log.info("\n✅  Data pipeline complete.")
    log.info("    Next: python main.py --mode train --experiment ef_whisper_large_real")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Maritime VHF ASR data pipeline")
    p.add_argument(
        "--mode",
        choices=["all", "splits", "combine", "stats"],
        default="all",
        help="all=full pipeline | splits=re-split | combine=rebuild combined | stats=print stats",
    )
    p.add_argument("--config", default="configs/config.yaml")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "all":
        run_full_pipeline(args.config)
    elif args.mode == "splits":
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(args.config)
        _cfg_dirs = {
            "real":      Path(cfg.data.real_data_dir),
            "simulated": Path(cfg.data.simulated_data_dir),
        }
        for ds_name in ("real", "simulated"):
            ds_dir = _cfg_dirs[ds_name]
            manifest_path = ds_dir / "manifest.json"
            if manifest_path.exists():
                records = validate_manifest(load_manifest(manifest_path), ds_dir)
                save_splits(split_records(records, seed=cfg.data.random_seed), ds_dir)
    elif args.mode == "combine":
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(args.config)
        build_combined(
            real_dir=Path(cfg.data.real_data_dir),
            sim_dir=Path(cfg.data.simulated_data_dir),
            output_dir=Path(cfg.data.combined_data_dir),
        )
    elif args.mode == "stats":
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(args.config)
        _cfg_dirs = {
            "real":      Path(cfg.data.real_data_dir),
            "simulated": Path(cfg.data.simulated_data_dir),
            "combined":  Path(cfg.data.combined_data_dir),
        }
        print_all_stats(_cfg_dirs)
