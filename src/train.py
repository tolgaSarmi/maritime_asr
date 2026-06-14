"""
src/train.py
══════════════════════════════════════════════════════════════════════════════
Training module for all experiments.

Supports three PEFT strategies (improved over last year):
  • encoder_freezing  – freeze acoustic encoder, train decoder only
  • lora              – FIXED: no quantisation, expanded target_modules,
                        systematic LR, proper early stopping patience
  • full_finetuning   – NEW: full weight update (viable with larger dataset)

Models supported:
  • OpenAI Whisper (small / medium / large)
  • Facebook Wav2Vec2
══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

from src.dataset import (
    WhisperASRDataset,
    Wav2Vec2ASRDataset,
    WhisperDataCollator,
    Wav2Vec2DataCollator,
)
from src.metrics import compute_wer, compute_cer
from src.utils import count_parameters, format_number, set_seed

log = logging.getLogger(__name__)


# ─── Model Loaders ───────────────────────────────────────────────────────────

def load_whisper(model_name: str, language: str = "english"):
    log.info("Loading Whisper: %s", model_name)
    processor = WhisperProcessor.from_pretrained(
        model_name, language=language, task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []
    params = count_parameters(model)
    log.info(
        "  total=%s  trainable=%s",
        format_number(params["total"]),
        format_number(params["trainable"]),
    )
    return model, processor


def load_wav2vec2(model_name: str, cfg: Any):
    log.info("Loading Wav2Vec2: %s", model_name)
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    w2v = cfg.models.wav2vec2
    model = Wav2Vec2ForCTC.from_pretrained(
        model_name,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=w2v.ctc_zero_infinity,  # was silently ignored before; prevents NaN gradients on fp16 overflow
        pad_token_id=processor.tokenizer.pad_token_id,
        attention_dropout=w2v.attention_dropout,
        hidden_dropout=w2v.hidden_dropout,
        feat_proj_dropout=w2v.feat_proj_dropout,
        mask_time_prob=w2v.mask_time_prob,
        layerdrop=w2v.layerdrop,
    )
    model.freeze_feature_encoder()
    params = count_parameters(model)
    log.info(
        "  total=%s  trainable=%s  (feature encoder frozen)",
        format_number(params["total"]),
        format_number(params["trainable"]),
    )
    # Confirm ctc_zero_infinity landed in the model config (not just the arg)
    log.info(
        "  [DEBUG] cfg.ctc_zero_infinity=%s  model.config.ctc_zero_infinity=%s",
        w2v.ctc_zero_infinity,
        model.config.ctc_zero_infinity,
    )
    return model, processor


# ─── PEFT: Apply Methods ─────────────────────────────────────────────────────

def apply_encoder_freezing_whisper(model) -> None:
    """Freeze Whisper encoder; only decoder trains."""
    for param in model.model.encoder.parameters():
        param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(
        "Encoder frozen — trainable: %s / %s (%.1f%%)",
        format_number(trainable), format_number(total),
        100 * trainable / total,
    )


def apply_encoder_freezing_wav2vec2(model) -> None:
    """Freeze Wav2Vec2 encoder layers; only CTC head trains."""
    for param in model.wav2vec2.encoder.parameters():
        param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    trainable_groups = sorted({
        ".".join(n.split(".")[:2])
        for n, p in model.named_parameters()
        if p.requires_grad
    })
    log.info(
        "Wav2Vec2 encoder frozen — trainable: %s / %s (%.1f%%)\n"
        "  [DEBUG] Trainable groups: %s",
        format_number(trainable), format_number(total),
        100 * trainable / total,
        trainable_groups,
    )


def apply_lora_whisper(model, cfg: Any):
    """
    Apply LoRA to Whisper.

    KEY FIXES over last year:
      1. No 8-bit quantisation — caused instability with small datasets
      2. Expanded target_modules: q/k/v/out instead of just q/v
      3. Learning rate set to 1e-3 (empirically validated last year)
    """
    from peft import LoraConfig, TaskType, get_peft_model

    lora_cfg = cfg.peft.lora
    config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,  # required for Seq2SeqTrainer.generate() in PEFT 0.12+
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        target_modules=list(lora_cfg.target_modules),
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # PeftModelForSeq2SeqLM.forward() has input_ids=None and inputs_embeds=None
    # as its own default parameters and explicitly passes both to self.base_model(),
    # even for audio models like Whisper that use input_features instead.
    #
    # In transformers 4.57.1, WhisperModel.forward() accepts **kwargs and
    # forwards them to self.decoder(). WhisperDecoder already receives
    # inputs_embeds=decoder_inputs_embeds explicitly from WhisperModel line 1137,
    # so the additional inputs_embeds=None arriving via **kwargs causes:
    #   TypeError: WhisperDecoder got multiple values for keyword argument 'inputs_embeds'
    # The same pattern previously caused the same error for input_ids=None.
    #
    # Fix: strip both input_ids=None and inputs_embeds=None at the LoraModel
    # entry point. The hook fires via __call__ (PeftModelForSeq2SeqLM calls
    # self.base_model(...) via __call__), before BaseTuner.forward() propagates
    # the kwargs to WhisperForConditionalGeneration.
    def _drop_spurious_peft_nones(_module, args, kwargs):
        for key in ("input_ids", "inputs_embeds"):
            if kwargs.get(key) is None:
                kwargs.pop(key, None)
        return args, kwargs

    model.base_model.register_forward_pre_hook(
        _drop_spurious_peft_nones, with_kwargs=True
    )

    return model


def apply_lora_wav2vec2(model, cfg: Any):
    """Apply LoRA to Wav2Vec2 attention layers."""
    from peft import LoraConfig, get_peft_model

    lora_cfg = cfg.peft.lora
    config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


# ─── Checkpoint Helpers ──────────────────────────────────────────────────────

def _validate_checkpoint(ckpt: Path) -> str | None:
    """
    Validate one checkpoint-N directory.  Returns None if valid, or a string
    describing the first failure found.

    Checks (in order):
      1. A model weights file exists — covers every training method and model:
           model.safetensors            Whisper EF, full fine-tuning
           pytorch_model.bin            legacy fallback for the above
           model-NNNNN-of-MMMMM.safetensors  sharded Whisper-large
           adapter_model.safetensors    LoRA (PEFT) — all models
           adapter_model.bin            legacy LoRA fallback
      2. trainer_state.json exists — Trainer needs this to restore step/epoch
         and the best-model path before calling optimizer/scheduler state.
      3. Any .safetensors file passes a binary format check (first 8 bytes).
         Guards against the failure mode where Drive fills up mid-write and
         torch.save writes a pickle stream to a .safetensors path, which later
         causes:
             ValueError: could not determine the shape of object type
             'torch.storage.UntypedStorage'
         in safetensors.torch.load_file() during resume.
    """
    import struct

    # 1. Model weights
    model_file: Path | None = None
    for name in (
        "model.safetensors",
        "pytorch_model.bin",
        "adapter_model.safetensors",
        "adapter_model.bin",
    ):
        if (ckpt / name).exists():
            model_file = ckpt / name
            break
    if model_file is None:
        shards = [
            f for f in ckpt.iterdir()
            if f.name.startswith("model-") and f.suffix == ".safetensors"
        ]
        if shards:
            model_file = shards[0]
    if model_file is None:
        return "no model weights file"

    # 2. trainer_state.json
    if not (ckpt / "trainer_state.json").exists():
        return "trainer_state.json missing"

    # 3. Safetensors format check
    if model_file.suffix == ".safetensors":
        try:
            with open(model_file, "rb") as fh:
                first8 = fh.read(8)
            if len(first8) < 8:
                return f"{model_file.name} too small to be valid safetensors"
            header_len = struct.unpack("<Q", first8)[0]
            # Valid safetensors JSON headers are 1 B – ~10 MB.
            # torch.save pickle magic (0x80 0x02/04/05 or PK\x03\x04) produces
            # header_len values in the billions, catching the corrupt-file case.
            if header_len == 0 or header_len > 100_000_000:
                return (
                    f"{model_file.name} failed safetensors header check "
                    f"(header_len={header_len}, bytes={first8.hex()!r}) — "
                    "likely a torch.save pickle written to a .safetensors path"
                )
        except OSError as exc:
            return f"{model_file.name} unreadable: {exc}"

    return None  # all checks passed


def _find_valid_checkpoint(output_dir: str) -> str | None:
    """
    Scan checkpoint-N directories newest → oldest and return the path of the
    first one that passes _validate_checkpoint(), or None to start fresh.
    All validation failures are logged before moving to the next candidate.
    """
    out = Path(output_dir)
    if not out.is_dir():
        return None

    checkpoints = sorted(
        [d for d in out.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1]),
        reverse=True,
    )
    if not checkpoints:
        return None

    for ckpt in checkpoints:
        failure = _validate_checkpoint(ckpt)
        if failure is not None:
            log.warning("Checkpoint %s invalid (%s) — skipping", ckpt.name, failure)
            continue
        # Determine which weights file was found for the log
        for wf in ("model.safetensors", "pytorch_model.bin",
                   "adapter_model.safetensors", "adapter_model.bin"):
            if (ckpt / wf).exists():
                weights_name = wf
                break
        else:
            weights_name = "sharded safetensors"
        if ckpt != checkpoints[0]:
            log.warning(
                "Fell back from %s to %s after validation failures",
                checkpoints[0].name, ckpt.name,
            )
        log.info("Resuming from %s (%s)", ckpt.name, weights_name)
        return str(ckpt)

    log.warning("No valid checkpoint in %s — starting fresh", output_dir)
    return None


def _cleanup_checkpoints(output_dir: str) -> None:
    """
    Delete checkpoint-N subdirs after successful training.

    trainer.save_model() already wrote the final model (weights only, no
    optimizer state) to output_dir root. The checkpoint subdirs are only
    needed for mid-run resume; once training completes they consume Drive
    space (~2 GB+ each for Whisper Medium) with no remaining benefit.
    """
    import shutil
    out = Path(output_dir)
    for ckpt in sorted(out.iterdir()):
        if ckpt.is_dir() and ckpt.name.startswith("checkpoint-"):
            shutil.rmtree(ckpt)
            log.info("Removed checkpoint dir after training: %s", ckpt.name)


# ─── Dataset Helpers ─────────────────────────────────────────────────────────

def _manifest_paths(data_type: str, cfg: Any) -> tuple[str, str, str]:
    """Return (train_manifest, val_manifest, data_root) using config-driven paths."""
    if data_type == "real":
        base = cfg.data.real_data_dir
    elif data_type == "simulated":
        base = cfg.data.simulated_data_dir
    elif data_type == "combined":
        base = cfg.data.combined_data_dir
    else:
        base = f"data/{data_type}"
    return (
        f"{base}/train_manifest.json",
        f"{base}/val_manifest.json",
        base,
    )


def _check_splits_exist(data_type: str, train_m: str, base: str) -> None:
    """Raise a clear error when split manifests are missing but manifest.json exists."""
    if not Path(train_m).exists():
        source_manifest = Path(base) / "manifest.json"
        if source_manifest.exists():
            raise FileNotFoundError(
                f"Split manifests not found for '{data_type}' — manifest.json exists but "
                f"train_manifest.json has not been generated yet.\n"
                f"  Run: python main.py --mode data"
            )
        raise FileNotFoundError(
            f"No manifest found for '{data_type}' at {base}/.\n"
            f"  Run: python label_studio_export.py --api-key YOUR_KEY --project {data_type}\n"
            f"  Then: python main.py --mode data"
        )


def _build_whisper_datasets(data_type: str, processor: Any, cfg: Any):
    train_m, val_m, root = _manifest_paths(data_type, cfg)
    _check_splits_exist(data_type, train_m, root)
    common = dict(
        data_root=root,
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
        max_duration=cfg.data.max_duration,
        min_duration=cfg.data.min_duration,
        normalize=cfg.data.normalize_audio,
    )
    train_ds = WhisperASRDataset(train_m, split="train", **common)
    val_ds = WhisperASRDataset(val_m, split="val", **common)
    return train_ds, val_ds


def _build_wav2vec2_datasets(data_type: str, processor: Any, cfg: Any):
    train_m, val_m, root = _manifest_paths(data_type, cfg)
    _check_splits_exist(data_type, train_m, root)
    common = dict(
        data_root=root,
        processor=processor,
        max_duration=cfg.data.max_duration,
        min_duration=cfg.data.min_duration,
        normalize=cfg.data.normalize_audio,
    )
    train_ds = Wav2Vec2ASRDataset(train_m, split="train", **common)
    val_ds = Wav2Vec2ASRDataset(val_m, split="val", **common)
    return train_ds, val_ds


# ─── Shared Training Args Builder ────────────────────────────────────────────

def _whisper_training_args(
    output_dir: str,
    learning_rate: float,
    cfg: Any,
    is_seq2seq: bool = True,
    fp16: bool | None = None,
) -> Seq2SeqTrainingArguments | TrainingArguments:
    tc = cfg.training
    # fp16 param overrides the global config flag so Whisper and Wav2Vec2
    # can be controlled independently. Wav2Vec2 is pinned to fp32 to prevent
    # NaN logits in the encoder; Whisper uses fp16 to keep activation memory
    # within T4 limits (307M-param Medium model OOMs in fp32 at batch_size=4).
    _fp16 = (fp16 if fp16 is not None else tc.fp16) and torch.cuda.is_available()
    base = dict(
        output_dir=output_dir,
        per_device_train_batch_size=tc.per_device_train_batch_size,
        per_device_eval_batch_size=tc.per_device_eval_batch_size,
        gradient_accumulation_steps=tc.gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=tc.warmup_steps,
        weight_decay=tc.weight_decay,
        max_grad_norm=tc.max_grad_norm,
        num_train_epochs=tc.num_train_epochs,
        lr_scheduler_type=tc.lr_scheduler_type,
        fp16=_fp16,
        eval_strategy=tc.evaluation_strategy,
        eval_steps=tc.eval_steps,
        save_strategy=tc.save_strategy,
        save_steps=tc.save_steps,
        save_total_limit=tc.save_total_limit,
        load_best_model_at_end=tc.load_best_model_at_end,
        metric_for_best_model=tc.metric_for_best_model,
        greater_is_better=tc.greater_is_better,
        logging_steps=tc.logging_steps,
        logging_dir=str(Path(tc.logging_dir) / Path(output_dir).name),
        report_to=list(tc.report_to),
        remove_unused_columns=False,
        label_names=["labels"],
        push_to_hub=False,
        dataloader_num_workers=getattr(tc, "dataloader_num_workers", 2),
    )
    if is_seq2seq:
        return Seq2SeqTrainingArguments(
            **base,
            predict_with_generate=True,
            generation_max_length=128,
        )
    return TrainingArguments(**base)


# ─── Whisper Trainer ─────────────────────────────────────────────────────────

class WhisperTrainer:
    """
    Handles all three training methods for Whisper:
      encoder_freezing | lora | full_finetuning
    """

    def __init__(self, cfg: Any, experiment: dict):
        self.cfg = cfg
        self.experiment = experiment
        self.exp_name = experiment["name"]
        self.method = experiment["method"]
        self.model_size = experiment["model_size"]
        self.data_type = experiment.get("train_data", "real")
        self.output_dir = str(Path(cfg.training.output_dir) / self.exp_name)

        set_seed(cfg.data.random_seed)

    def train(self) -> dict:
        log.info("═" * 60)
        log.info("Experiment : %s", self.exp_name)
        log.info("Method     : %s", self.method)
        log.info("Model      : %s", self.model_size)
        log.info("Data       : %s", self.data_type)
        log.info("═" * 60)

        model, processor = load_whisper(
            self.model_size, self.cfg.models.whisper.language
        )

        # Apply training method
        if self.method == "encoder_freezing":
            apply_encoder_freezing_whisper(model)
            lr = self.cfg.peft.encoder_freezing.learning_rate

        elif self.method == "lora":
            model = apply_lora_whisper(model, self.cfg)
            lr = self.cfg.peft.lora.learning_rate

        elif self.method == "full_finetuning":
            # All parameters trainable — no freezing
            lr = self.cfg.peft.full_finetuning.learning_rate
            log.info(
                "Full fine-tuning — all %s parameters trainable",
                format_number(sum(p.numel() for p in model.parameters())),
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

        train_ds, val_ds = _build_whisper_datasets(
            self.data_type, processor, self.cfg
        )
        collator = WhisperDataCollator(tokenizer=processor.tokenizer)
        training_args = _whisper_training_args(self.output_dir, lr, self.cfg, fp16=True)

        # FIXED early stopping patience (5 vs last year's 3)
        callbacks = [
            EarlyStoppingCallback(
                early_stopping_patience=self.cfg.training.early_stopping_patience,
                early_stopping_threshold=self.cfg.training.early_stopping_threshold,
            )
        ]

        def compute_metrics(eval_preds):
            pred_ids, label_ids = eval_preds
            label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
            pred_str = processor.tokenizer.batch_decode(
                pred_ids, skip_special_tokens=True
            )
            label_str = processor.tokenizer.batch_decode(
                label_ids, skip_special_tokens=True
            )
            wer = compute_wer(label_str, pred_str)["wer"]
            cer = compute_cer(label_str, pred_str)
            return {"eval_wer": wer, "eval_cer": cer}

        model.config.use_cache = False

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator,
            processing_class=processor,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )

        log.info("Starting training (%d train / %d val samples)...",
                 len(train_ds), len(val_ds))

        last_checkpoint = _find_valid_checkpoint(self.output_dir)
        result = trainer.train(resume_from_checkpoint=last_checkpoint)

        trainer.save_model(self.output_dir)
        processor.save_pretrained(self.output_dir)
        trainer.save_metrics("train", result.metrics)
        trainer.save_state()
        # Remove checkpoint-N subdirs now that the final model is saved.
        # Each subdir holds optimizer state (~1–4 GB) that is only needed
        # for mid-run resume; keeping it after completion wastes Drive space.
        _cleanup_checkpoints(self.output_dir)

        log.info(
            "Done. train_loss=%.4f", result.metrics.get("train_loss", 0)
        )
        return result.metrics


# ─── Wav2Vec2 Trainer ────────────────────────────────────────────────────────

class Wav2Vec2Trainer:
    """
    Handles encoder_freezing and lora for Wav2Vec2.
    """

    def __init__(self, cfg: Any, experiment: dict):
        self.cfg = cfg
        self.experiment = experiment
        self.exp_name = experiment["name"]
        self.method = experiment["method"]
        self.model_size = experiment["model_size"]
        self.data_type = experiment.get("train_data", "real")
        self.output_dir = str(Path(cfg.training.output_dir) / self.exp_name)

        set_seed(cfg.data.random_seed)

    def train(self) -> dict:
        log.info("═" * 60)
        log.info("Experiment : %s", self.exp_name)
        log.info("Method     : %s", self.method)
        log.info("Model      : %s", self.model_size)
        log.info("Data       : %s", self.data_type)
        log.info("═" * 60)

        model, processor = load_wav2vec2(self.model_size, self.cfg)

        if self.method == "encoder_freezing":
            apply_encoder_freezing_wav2vec2(model)
            lr = self.cfg.peft.encoder_freezing.learning_rate
        elif self.method == "lora":
            model = apply_lora_wav2vec2(model, self.cfg)
            lr = self.cfg.peft.lora.learning_rate
        else:
            raise ValueError(f"Unknown method: {self.method}")

        train_ds, val_ds = _build_wav2vec2_datasets(
            self.data_type, processor, self.cfg
        )
        collator = Wav2Vec2DataCollator(processor=processor)
        training_args = _whisper_training_args(
            self.output_dir, lr, self.cfg, is_seq2seq=False, fp16=False
        )
        callbacks = [
            EarlyStoppingCallback(
                early_stopping_patience=self.cfg.training.early_stopping_patience,
                early_stopping_threshold=self.cfg.training.early_stopping_threshold,
            )
        ]

        def compute_metrics(eval_preds):
            import numpy as np
            pred_logits, label_ids = eval_preds
            pred_ids = np.argmax(pred_logits, axis=-1)
            label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
            pred_str = processor.batch_decode(pred_ids)
            label_str = processor.tokenizer.batch_decode(
                label_ids, group_tokens=False
            )
            wer = compute_wer(label_str, pred_str)["wer"]
            cer = compute_cer(label_str, pred_str)
            return {"eval_wer": wer, "eval_cer": cer}

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )

        last_checkpoint = _find_valid_checkpoint(self.output_dir)
        result = trainer.train(resume_from_checkpoint=last_checkpoint)

        trainer.save_model(self.output_dir)
        processor.save_pretrained(self.output_dir)
        trainer.save_metrics("train", result.metrics)
        trainer.save_state()
        _cleanup_checkpoints(self.output_dir)

        log.info("Done. train_loss=%.4f", result.metrics.get("train_loss", 0))
        return result.metrics


# ─── Factory ─────────────────────────────────────────────────────────────────

def build_trainer(cfg: Any, experiment: dict) -> WhisperTrainer | Wav2Vec2Trainer:
    model_type = experiment.get("model", "whisper")
    if model_type == "whisper":
        return WhisperTrainer(cfg, experiment)
    elif model_type == "wav2vec2":
        return Wav2Vec2Trainer(cfg, experiment)
    raise ValueError(f"Unknown model type: {model_type!r}")
