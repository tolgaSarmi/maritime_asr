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
from transformers.trainer_utils import get_last_checkpoint

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


def _install_whisper_lora_debug(peft_model):
    """Print-only instrumentation tracing kwargs through the PEFT→Whisper stack.

    Call chain under training:
      PeftModelForSeq2SeqLM.__call__
        → pre-hook fires        (Stage 1: LoraModel entry, after fix hook)
        → BaseTuner.forward(*args, **kwargs)
            → WhisperForCG.forward()   ← .forward() direct; wrapped at Stage 2
                → WhisperModel.__call__
                    → pre-hook fires   (Stage 3)
                    → WhisperModel.forward()
                        → WhisperDecoder.__call__
                            → pre-hook fires  (Stage 4)

    Returns (handles, ctr). Pass ctr to _WhisperLoraDebugTrainer so step
    numbers stay consistent across the batch log and each stage hook.
    """
    _KEYS = [
        "input_features", "labels", "decoder_input_ids",
        "decoder_attention_mask", "input_ids", "inputs_embeds",
        "decoder_inputs_embeds", "cache_position",
    ]
    ctr = {"n": 0}
    handles = []

    def _active():
        return ctr["n"] < 2

    def _log(stage, kwargs):
        print(f"\n[WHISPER LORA DBG step={ctr['n']}] {stage}", flush=True)
        print(f"  all keys: {sorted(kwargs.keys())}", flush=True)
        for k in _KEYS:
            if k in kwargs:
                v = kwargs[k]
                if v is None:
                    desc = "None"
                elif hasattr(v, "shape"):
                    desc = f"tensor{tuple(v.shape)} dtype={v.dtype}"
                else:
                    desc = repr(v)
                print(f"  {k}: {desc}", flush=True)

    # Stage 1: LoraModel entry — fires AFTER existing _drop_spurious_input_ids
    def _lora_model_hook(m, args, kwargs):
        if _active():
            _log("Stage1 → LoraModel.forward [entry, after fix hook]", kwargs)
        return args, kwargs

    handles.append(
        peft_model.base_model.register_forward_pre_hook(
            _lora_model_hook, with_kwargs=True
        )
    )

    # Stage 2: WhisperForConditionalGeneration.forward
    # BaseTuner calls self.model.forward() directly (not via __call__), so
    # register_forward_pre_hook would not fire. Wrap the instance method instead.
    wfcg = peft_model.base_model.model
    _orig_wfcg_fwd = wfcg.forward

    def _wfcg_debug_fwd(*args, **kwargs):
        if _active():
            _log("Stage2 → WhisperForConditionalGeneration.forward [from BaseTuner]", kwargs)
        return _orig_wfcg_fwd(*args, **kwargs)

    wfcg.forward = _wfcg_debug_fwd

    # Stage 3: WhisperModel (called via __call__ from WhisperForConditionalGeneration)
    try:
        whisper_model = peft_model.base_model.model.model

        def _wm_hook(m, args, kwargs):
            if _active():
                _log("Stage3 → WhisperModel [from WhisperForConditionalGeneration]", kwargs)
            return args, kwargs

        handles.append(
            whisper_model.register_forward_pre_hook(_wm_hook, with_kwargs=True)
        )
    except AttributeError:
        log.warning("[WHISPER LORA DBG] .base_model.model.model not found — no Stage3 hook")

    # Stage 4: WhisperDecoder (called via __call__ from WhisperModel.forward)
    try:
        decoder = peft_model.base_model.model.model.decoder

        def _wd_hook(m, args, kwargs):
            if _active():
                _log("Stage4 → WhisperDecoder [from WhisperModel]", kwargs)
            return args, kwargs

        handles.append(
            decoder.register_forward_pre_hook(_wd_hook, with_kwargs=True)
        )
    except AttributeError:
        log.warning("[WHISPER LORA DBG] .base_model.model.model.decoder not found — no Stage4 hook")

    return handles, ctr


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


# ─── Dataset Helpers ─────────────────────────────────────────────────────────

def _manifest_paths(data_type: str) -> tuple[str, str, str]:
    """Return (train, val, data_root) paths for a given data_type."""
    base = f"data/{data_type}" if data_type != "combined" else "data/combined"
    return (
        f"{base}/train_manifest.json",
        f"{base}/val_manifest.json",
        base,
    )


def _check_splits_exist(data_type: str, train_m: str) -> None:
    """Raise a clear error when split manifests are missing but manifest.json exists."""
    if not Path(train_m).exists():
        base = f"data/{data_type}" if data_type != "combined" else "data/combined"
        source_manifest = Path(base) / "manifest.json"
        if source_manifest.exists():
            raise FileNotFoundError(
                f"Split manifests not found for '{data_type}' — manifest.json exists but "
                f"train_manifest.json has not been generated yet.\n"
                f"  Run Step 6:  python main.py --mode data"
            )
        raise FileNotFoundError(
            f"No manifest found for '{data_type}' at {base}/.\n"
            f"  Run Step 5:  python label_studio_export.py --api-key YOUR_KEY --project {data_type}\n"
            f"  Then Step 6: python main.py --mode data"
        )


def _build_whisper_datasets(data_type: str, processor: Any, cfg: Any):
    train_m, val_m, root = _manifest_paths(data_type)
    _check_splits_exist(data_type, train_m)
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
    train_m, val_m, root = _manifest_paths(data_type)
    _check_splits_exist(data_type, train_m)
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
) -> Seq2SeqTrainingArguments | TrainingArguments:
    tc = cfg.training
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
        fp16=tc.fp16 and torch.cuda.is_available(),
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


# ─── Whisper LoRA Debug Trainer ──────────────────────────────────────────────

class _WhisperLoraDebugTrainer(Seq2SeqTrainer):
    """Debug-only Seq2SeqTrainer.

    Logs the Trainer batch keys before each forward pass for the first 2 steps,
    coordinated with the stage hooks installed by _install_whisper_lora_debug().
    The shared ctr dict keeps step numbers consistent across the log stages.
    """

    def __init__(self, *args, dbg_ctr=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._ctr = dbg_ctr if dbg_ctr is not None else {"n": 0}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        n = self._ctr["n"]
        if n < 2:
            _KEYS = [
                "input_features", "labels", "decoder_input_ids",
                "decoder_attention_mask", "input_ids", "inputs_embeds",
                "decoder_inputs_embeds", "cache_position",
            ]
            print(
                f"\n[WHISPER LORA DBG step={n}] Trainer batch — keys: "
                f"{sorted(inputs.keys())}",
                flush=True,
            )
            for k in _KEYS:
                if k in inputs:
                    v = inputs[k]
                    if v is None:
                        desc = "None"
                    elif hasattr(v, "shape"):
                        desc = f"tensor{tuple(v.shape)} dtype={v.dtype}"
                    else:
                        desc = repr(v)
                    print(f"  {k}: {desc}", flush=True)
        result = super().compute_loss(
            model, inputs, return_outputs=return_outputs, **kwargs
        )
        if n < 2:
            self._ctr["n"] += 1
        return result


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
            _dbg_handles, _dbg_ctr = _install_whisper_lora_debug(model)

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
        training_args = _whisper_training_args(self.output_dir, lr, self.cfg)

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

        _trainer_cls = _WhisperLoraDebugTrainer if self.method == "lora" else Seq2SeqTrainer
        _trainer_kwargs = dict(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator,
            processing_class=processor,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )
        if self.method == "lora":
            _trainer_kwargs["dbg_ctr"] = _dbg_ctr
        trainer = _trainer_cls(**_trainer_kwargs)

        log.info("Starting training (%d train / %d val samples)...",
                 len(train_ds), len(val_ds))

        # Detect and resume from latest checkpoint if available
        last_checkpoint = get_last_checkpoint(self.output_dir)
        if last_checkpoint is not None:
            log.info("Resuming from checkpoint: %s", last_checkpoint)
            result = trainer.train(resume_from_checkpoint=last_checkpoint)
        else:
            log.info("No checkpoint found — starting fresh")
            result = trainer.train()

        trainer.save_model(self.output_dir)
        processor.save_pretrained(self.output_dir)
        trainer.save_metrics("train", result.metrics)
        trainer.save_state()

        log.info(
            "Done. train_loss=%.4f", result.metrics.get("train_loss", 0)
        )
        return result.metrics


# ─── Wav2Vec2 Debug Trainer ──────────────────────────────────────────────────

class _Wav2Vec2DebugTrainer(Trainer):
    """
    Trainer subclass that instruments the first few training steps to reveal
    exactly where NaN enters the computation graph.

    Strategy: register a forward hook on model.lm_head to capture logits
    in-situ (with the same fp16/autocast context as real training), then
    compute CTC input_lengths and target_lengths independently and print
    the full pre-CTC diagnostic table.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dbg_steps = 0

    # ------------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self._dbg_steps < 3:
            return self._compute_loss_with_debug(
                model, inputs, return_outputs, **kwargs
            )
        return super().compute_loss(
            model, inputs, return_outputs=return_outputs, **kwargs
        )

    # ------------------------------------------------------------------
    def _compute_loss_with_debug(self, model, inputs, return_outputs, **kwargs):
        """Run the real forward pass, hooking lm_head to capture logits."""
        captured: dict = {}

        def _lm_head_hook(_module, _inp, out):
            # Called synchronously during model.forward(); out is the logits
            # tensor in whatever dtype autocast chose (fp16 or fp32).
            captured["logits"] = out.detach()

        # Attach hook to the CTC projection layer
        try:
            hook_layer = model.lm_head
        except AttributeError:
            # PEFT wrapper: delegate to base model
            hook_layer = model.base_model.model.lm_head

        handle = hook_layer.register_forward_hook(_lm_head_hook)
        try:
            result = super().compute_loss(
                model, inputs, return_outputs=True, **kwargs
            )
        finally:
            handle.remove()

        loss = result[0]
        self._dbg_steps += 1

        # ── Gather tensors ────────────────────────────────────────────
        logits = captured.get("logits")          # (B, T, vocab) or None
        labels = inputs.get("labels")            # (B, L) with -100 padding
        attn   = inputs.get("attention_mask")    # (B, raw_audio_len)

        # input_lengths: raw audio samples → CNN output frames
        if attn is not None:
            try:
                input_lengths = model._get_feat_extract_output_lengths(
                    attn.sum(-1)
                ).to(torch.long)
            except AttributeError:
                try:
                    input_lengths = (
                        model.base_model.model
                        ._get_feat_extract_output_lengths(attn.sum(-1))
                        .to(torch.long)
                    )
                except Exception:
                    # Fallback: apply wav2vec2 standard CNN arithmetic
                    raw = attn.sum(-1).float()
                    for k, s in zip(
                        [10, 3, 3, 3, 3, 2, 2], [5, 2, 2, 2, 2, 2, 2]
                    ):
                        raw = torch.floor((raw - k) / s) + 1
                    input_lengths = raw.long()
        elif logits is not None:
            B, T = logits.shape[:2]
            input_lengths = torch.full((B,), T, dtype=torch.long,
                                       device=logits.device)
        else:
            input_lengths = None

        # target_lengths: number of non-(-100) tokens per sample
        if labels is not None:
            target_lengths = (labels != -100).sum(dim=-1).to(torch.long)
            labels_flat = labels[labels != -100]
            unk_count = int((labels_flat == 3).sum())
            consec_repeats = 0
            for i in range(labels.shape[0]):
                row = labels[i][labels[i] != -100]
                if len(row) > 1:
                    consec_repeats += int((row[1:] == row[:-1]).sum())
        else:
            target_lengths = None
            unk_count = consec_repeats = -1

        infeasible = (
            int((target_lengths > input_lengths).sum())
            if (target_lengths is not None and input_lengths is not None)
            else -1
        )

        # ── Print ─────────────────────────────────────────────────────
        print(
            f"\n=== [W2V DEBUG] Step {self._dbg_steps} — pre-CTC diagnostics ===",
            flush=True,
        )
        if logits is not None:
            lnan = torch.isnan(logits).any().item()
            linf = torch.isinf(logits).any().item()
            print(f"  logits.shape:             {tuple(logits.shape)}", flush=True)
            print(f"  logits.dtype:             {logits.dtype}", flush=True)
            print(f"  logits nan:               {lnan}", flush=True)
            print(f"  logits inf:               {linf}", flush=True)
            if not lnan and not linf:
                print(
                    f"  logits min/max:           "
                    f"{logits.min().item():.4f} / {logits.max().item():.4f}",
                    flush=True,
                )
        else:
            print("  logits:                   NOT CAPTURED (hook missed)", flush=True)

        if input_lengths is not None:
            print(f"  input_lengths[:10]:       {input_lengths[:10].tolist()}", flush=True)
            print(
                f"  input_lengths min/max:    "
                f"{input_lengths.min().item()} / {input_lengths.max().item()}",
                flush=True,
            )
        if target_lengths is not None:
            print(f"  target_lengths[:10]:      {target_lengths[:10].tolist()}", flush=True)
            print(
                f"  target_lengths min/max:   "
                f"{target_lengths.min().item()} / {target_lengths.max().item()}",
                flush=True,
            )
        print(
            f"  target > input (infeasible): {infeasible}",
            flush=True,
        )
        print(f"  <unk> tokens in batch:    {unk_count}", flush=True)
        print(f"  consecutive repeats:      {consec_repeats}", flush=True)
        print(
            f"  loss: {loss.item():.6f}  "
            f"finite={torch.isfinite(loss).item()}  dtype={loss.dtype}",
            flush=True,
        )
        print("=== [W2V DEBUG] end ===\n", flush=True)

        if not return_outputs:
            return loss
        return result


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
            self.output_dir, lr, self.cfg, is_seq2seq=False
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

        # Pre-training diagnostics
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        trainable_names = [
            n for n, p in model.named_parameters() if p.requires_grad
        ]
        print(
            f"\n=== [W2V DEBUG] Pre-training config ===\n"
            f"  model.config.ctc_zero_infinity = {model.config.ctc_zero_infinity}\n"
            f"  training_args.fp16             = {training_args.fp16}\n"
            f"  training_args.bf16             = {training_args.bf16}\n"
            f"  trainable params               = {n_trainable:,} / {n_total:,} "
            f"({100*n_trainable/n_total:.2f}%)\n"
            f"  trainable param names ({len(trainable_names)} total):\n"
            + "\n".join(f"    {n}" for n in trainable_names[:20])
            + ("\n    ..." if len(trainable_names) > 20 else "")
            + "\n=== [W2V DEBUG] end ===\n",
            flush=True,
        )

        trainer = _Wav2Vec2DebugTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )

        # Detect and resume from latest checkpoint if available
        last_checkpoint = get_last_checkpoint(self.output_dir)
        if last_checkpoint is not None:
            log.info("Resuming from checkpoint: %s", last_checkpoint)
            result = trainer.train(resume_from_checkpoint=last_checkpoint)
        else:
            log.info("No checkpoint found — starting fresh")
            result = trainer.train()

        trainer.save_model(self.output_dir)
        trainer.save_metrics("train", result.metrics)
        trainer.save_state()

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
