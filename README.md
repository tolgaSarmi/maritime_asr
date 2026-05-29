# Maritime VHF ASR — Dissertation Project

**Transformer-Based ASR Fine-Tuning on Real and Simulated Maritime Speech**  
MSc Artificial Intelligence & Machine Learning — University of Limerick

---

## Overview

This project extends last year's dissertation (*Parameter-Efficient Fine-Tuning of ASR Models for Maritime Radio Communications*) with three key improvements:

| Improvement | Last Year | This Year |
|---|---|---|
| Labelled data | 62 min / 754 samples | ~3× more (real + simulated) |
| LoRA implementation | Unstable (8-bit quant, only q/v proj) | Fixed (no quant, all attention layers, systematic LR) |
| Training conditions | Real data only | **Real / Simulated / Combined** |
| Full fine-tuning | Out of scope | Included (now viable with more data) |
| Validation set | ~10 min (too small, caused premature stopping) | Proper 10% split |
| Models | Whisper + Parakeet | Whisper + Wav2Vec2 |

**Core research question:** Can simulated VHF speech data substitute for, or usefully augment, scarce real maritime radio recordings for ASR domain adaptation?

---

## Project Structure

```
.
├── main.py                     ← single entry point for everything
├── data_pipeline.py            ← silence removal, segmentation, splits
├── label_studio_export.py      ← download data from heartex.com
│
├── configs/
│   └── config.yaml             ← all hyperparameters and experiment definitions
│
├── src/
│   ├── preprocessing.py        ← audio loading, normalisation, trimming
│   ├── dataset.py              ← PyTorch Dataset classes + DataCollators
│   ├── augmentation.py         ← VHF channel simulation, SpecAugment
│   ├── train.py                ← encoder freezing / LoRA / full fine-tuning
│   ├── evaluate.py             ← evaluation across all experiments
│   ├── inference.py            ← transcribe audio files
│   ├── metrics.py              ← WER, CER, error analysis
│   ├── visualization.py        ← all dissertation figures
│   └── utils.py                ← config, logging, checkpoints, seeding
│
├── data/
│   ├── real/                   ← Maritime_ASR_Main exports from Label Studio
│   ├── simulated/              ← sim_vhf_dataset exports from Label Studio
│   └── combined/               ← merged manifests (auto-generated)
│
├── checkpoints/                ← saved model checkpoints (auto-created)
├── results/
│   ├── figures/                ← all dissertation plots (auto-generated)
│   └── logs/                   ← TensorBoard logs
│
└── requirements.txt
```

---

## Setup

```bash
# 1. Clone / download the project
cd asr_dissertation

# 2. Create environment
python -m venv venv && source venv/bin/activate   # Linux/Mac
# or: python -m venv venv && venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Label Studio API key
cp .env.example .env
# Edit .env and set LABEL_STUDIO_API_KEY=your_key_here
# Alternatively, export LABEL_STUDIO_API_KEY="your_key_here"

# 5. Confirm the machine is ready
python main.py --mode check
```

The API key is read from `LABEL_STUDIO_API_KEY` or from a local `.env` file.
Get the key from: https://app.heartex.com/user/account

---

## Quickstart

### Step 0 — Check local requirements

```bash
python main.py --mode check
# Reports missing packages, FFmpeg, API key, data directories, and checkpoints.
```

### Step 1 — Download annotated data from Label Studio

```bash
python label_studio_export.py --api-key YOUR_KEY
# If LABEL_STUDIO_API_KEY is set in the environment or .env, --api-key can be omitted.
# Downloads audio + transcriptions for both:
#   • Maritime_ASR_Main  (real data)
#   • sim_vhf_dataset    (simulated data)
```

### Step 2 — Prepare datasets (split + combine)

```bash
python main.py --mode data
# Validates records, normalises text, creates 80/10/10 splits,
# builds combined/ manifests
```

### Step 3 — Run a single experiment

```bash
# Encoder freezing — Whisper-large — trained on real data
python main.py --mode train --experiment ef_whisper_large_real

# LoRA (fixed) — Whisper-large — trained on combined data
python main.py --mode train --experiment lora_whisper_large_combined

# Full fine-tuning — Whisper-large — real data (NEW)
python main.py --mode train --experiment full_ft_whisper_large_real

# Wav2Vec2 — encoder freezing — simulated data (NEW)
python main.py --mode train --experiment ef_wav2vec2_simulated
```

### Step 4 — Run all experiments

```bash
python main.py --mode train_all
# Skips experiments that already have a checkpoint
# Use --force to retrain
```

### Step 5 — Evaluate

```bash
# Evaluate one experiment on both test sets
python main.py --mode eval --experiment lora_whisper_large_combined

# Evaluate all and save results/all_results.json
python main.py --mode eval_all
```

### Step 6 — Generate dissertation figures

```bash
python main.py --mode figures
# Saves all plots to results/figures/
```

### Step 7 — Transcribe a file

```bash
python main.py --mode transcribe \
    --checkpoint checkpoints/lora_whisper_large_combined \
    --audio path/to/vhf_recording.wav
```

---

## Experiment Matrix

The full matrix has 3 data conditions × 4 training methods × 2 models, evaluated on both real and simulated test sets:

| Experiment | Model | Method | Train Data |
|---|---|---|---|
| `baseline_whisper_*` | Whisper small/med/large | Zero-shot | — |
| `baseline_wav2vec2` | Wav2Vec2 | Zero-shot | — |
| `ef_whisper_*_real` | Whisper small/med/large | Encoder Freezing | Real |
| `ef_whisper_*_simulated` | Whisper small/med/large | Encoder Freezing | Simulated |
| `ef_whisper_*_combined` | Whisper small/med/large | Encoder Freezing | Combined |
| `lora_whisper_*_real` | Whisper small/med/large | LoRA (fixed) | Real |
| `lora_whisper_*_simulated` | Whisper small/med/large | LoRA (fixed) | Simulated |
| `lora_whisper_*_combined` | Whisper small/med/large | LoRA (fixed) | Combined |
| `full_ft_whisper_*_real` | Whisper small/med/large | Full Fine-Tuning | Real |
| `full_ft_whisper_large_combined` | Whisper-large | Full Fine-Tuning | Combined |
| `ef_wav2vec2_*` | Wav2Vec2 | Encoder Freezing | Real/Sim/Combined |
| `lora_wav2vec2_*` | Wav2Vec2 | LoRA | Real/Combined |

---

## Key Differences from Last Year's LoRA

Last year's LoRA experiments were unstable. Here is what was fixed:

```yaml
# LAST YEAR (unstable)
target_modules: ["q_proj", "v_proj"]   # only 2 layers
use_8bit_quantisation: true            # caused gradient instability
learning_rate: 5e-5                    # too low — model couldn't learn

# THIS YEAR (fixed)
target_modules: ["q_proj", "k_proj", "v_proj", "out_proj"]  # all attention
use_8bit_quantisation: false           # disabled — dataset fits in memory
learning_rate: 1.0e-3                  # empirically validated last year
early_stopping_patience: 5            # increased from 3 — prevents premature stop
```

---

## Monitoring Training

```bash
# TensorBoard (in a separate terminal)
tensorboard --logdir results/logs

# Then open: http://localhost:6006
```

---

## Generated Figures

After `python main.py --mode figures`:

| File | Description |
|---|---|
| `01_baseline_comparison.png` | Zero-shot WER for all models (replicates last year's Fig 20) |
| `02_peft_comparison_whisper.png` | Baseline vs EF vs LoRA vs Full FT per model size |
| `03_data_condition_comparison.png` | Real vs Simulated vs Combined training |
| `04_cross_domain_heatmap_lora.png` | **Cross-domain generalisation matrix** |
| `05_model_comparison.png` | Whisper vs Wav2Vec2 across conditions |
| `06_error_breakdown.png` | Substitution / Deletion / Insertion rates |
| `07_wer_distribution.png` | Per-sample WER distributions (violin plots) |

---

## Datasets

| Dataset | Source | Type | Notes |
|---|---|---|---|
| `Maritime_ASR_Main` | Label Studio (heartex.com) | Real IRCG VHF recordings | ~1,556 annotated samples |
| `sim_vhf_dataset` | Label Studio (heartex.com) | Simulated VHF speech | ~1,916 annotated samples (100%) |

Both datasets were annotated using Label Studio with domain-expert review.  
Audio was preprocessed using `pydub.silence` (silence_thresh=−35 dBFS, min_silence_len=1500ms) — confirmed optimal in last year's grid search.

---

## What you need to run the project

- **Python 3.10+** with dependencies installed from `requirements.txt`.
- **FFmpeg** available on `PATH` for robust audio decoding through `pydub` and related audio tooling.
- **Heartex/Label Studio API key** stored in `LABEL_STUDIO_API_KEY` or in a local `.env` file copied from `.env.example`.
- **Prepared datasets** under `data/real`, `data/simulated`, and `data/combined`; create these with `label_studio_export.py` followed by `python main.py --mode data`.
- **GPU capacity** matching the experiment size listed below.

Run `python main.py --mode check` at any point to see exactly what is still missing on the current machine.

---

## Hardware Requirements

| Experiment | Minimum GPU |
|---|---|
| Whisper-small | 8 GB VRAM |
| Whisper-medium | 12 GB VRAM |
| Whisper-large (EF / LoRA) | 24 GB VRAM (A100 recommended) |
| Whisper-large (Full FT) | 40 GB VRAM or gradient checkpointing |
| Wav2Vec2-base | 8 GB VRAM |

Mixed-precision (fp16) is enabled by default and halves memory usage.

---

## Results Interpretation

The key research question is answered by comparing the **cross-domain cells** in `04_cross_domain_heatmap`:

- **Train: Simulated → Eval: Real** — if WER is competitive with *Train: Real → Eval: Real*, simulated data can substitute for real data
- **Train: Combined → Eval: Real** — if WER beats *Train: Real → Eval: Real*, simulated data provides useful augmentation
- **Train: Real → Eval: Simulated** — measures how well real-trained models generalise to synthetic speech

---

## Future Work (from last year's recommendations)

- [ ] Annotation of remaining 24+ hours of `Maritime_ASR_Main` audio
- [ ] Contextual biasing (TCPGen) for vessel names, channel numbers, call signs
- [ ] Real-time streaming deployment using Parakeet-TDT
- [ ] Multi-speaker diarisation for multi-channel VHF recordings
