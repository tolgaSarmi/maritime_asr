# ══════════════════════════════════════════════════════════════════════════════
# Maritime VHF ASR — Makefile
# ══════════════════════════════════════════════════════════════════════════════
# Usage (Mac terminal):
#   make export-real        → fetch real dataset from Label Studio
#   make export-sim         → download simulated audio
#   make export             → fetch both datasets
#   make data               → run data pipeline (splits + validation)
#   make train EXP=ef_whisper_small_real   → train one experiment
#   make train-all          → train all T4-compatible experiments
#   make eval               → evaluate all trained experiments
#   make figures            → generate dissertation figures
#   make results            → print results table
#   make save               → save checkpoints + manifests to Drive
#   make restore            → restore checkpoints + manifests from Drive
#   make setup-mac          → install dependencies on Mac (CPU only)
#   make setup-colab        → install dependencies in Colab (GPU)
#   make clean              → remove generated data and checkpoints
# ══════════════════════════════════════════════════════════════════════════════

PYTHON     = python3
API_KEY    = $(LABEL_STUDIO_API_KEY)
DRIVE      = /content/drive/MyDrive/ASR_Dissertation
EXP        = ef_whisper_small_real

# ── Setup ─────────────────────────────────────────────────────────────────────
setup-mac:
	pip install -r requirements-mac.txt

setup-colab:
	pip install -q "scipy>=1.14.0" "scikit-learn>=1.6.0" \
		transformers peft datasets accelerate evaluate \
		jiwer librosa soundfile omegaconf rich inflect \
		tensorboard seaborn

# ── Data Export ───────────────────────────────────────────────────────────────
export-real:
	$(PYTHON) label_studio_export.py --api-key $(API_KEY) --project real

export-sim:
	$(PYTHON) label_studio_export.py --api-key $(API_KEY) --project simulated

export:
	$(PYTHON) label_studio_export.py --api-key $(API_KEY) --project both

# ── Data Pipeline ─────────────────────────────────────────────────────────────
data:
	$(PYTHON) main.py --mode data

# ── Training ──────────────────────────────────────────────────────────────────
train:
	$(PYTHON) main.py --mode train --experiment $(EXP)

train-all:
	$(PYTHON) main.py --mode train --experiment ef_whisper_small_real
	$(PYTHON) main.py --mode train --experiment ef_whisper_medium_real
	$(PYTHON) main.py --mode train --experiment ef_wav2vec2_real
	$(PYTHON) main.py --mode train --experiment lora_whisper_small_real
	$(PYTHON) main.py --mode train --experiment lora_whisper_medium_real

train-simulated:
	$(PYTHON) main.py --mode train --experiment ef_whisper_small_simulated
	$(PYTHON) main.py --mode train --experiment ef_whisper_medium_simulated
	$(PYTHON) main.py --mode train --experiment lora_whisper_small_simulated
	$(PYTHON) main.py --mode train --experiment lora_whisper_medium_simulated

train-combined:
	$(PYTHON) main.py --mode train --experiment ef_whisper_small_combined
	$(PYTHON) main.py --mode train --experiment ef_whisper_medium_combined
	$(PYTHON) main.py --mode train --experiment lora_whisper_small_combined
	$(PYTHON) main.py --mode train --experiment lora_whisper_medium_combined

# ── Evaluation ────────────────────────────────────────────────────────────────
eval:
	$(PYTHON) main.py --mode eval_all

figures:
	$(PYTHON) main.py --mode figures

results:
	$(PYTHON) main.py --mode results

# ── Drive Backup / Restore ────────────────────────────────────────────────────
save:
	@echo "Saving checkpoints to Drive..."
	@mkdir -p $(DRIVE)/checkpoints $(DRIVE)/data/real $(DRIVE)/data/simulated
	@cp -r checkpoints/. $(DRIVE)/checkpoints/ 2>/dev/null || true
	@cp data/real/manifest.json $(DRIVE)/data/real/ 2>/dev/null || true
	@cp data/simulated/manifest.json $(DRIVE)/data/simulated/ 2>/dev/null || true
	@echo "Saved to $(DRIVE)"

restore:
	@echo "Restoring from Drive..."
	@mkdir -p checkpoints data/real data/simulated
	@cp -r $(DRIVE)/checkpoints/. checkpoints/ 2>/dev/null || true
	@cp $(DRIVE)/data/real/manifest.json data/real/ 2>/dev/null || true
	@cp $(DRIVE)/data/simulated/manifest.json data/simulated/ 2>/dev/null || true
	@echo "Restored from $(DRIVE)"

save-manifests:
	@mkdir -p $(DRIVE)/data/real $(DRIVE)/data/simulated $(DRIVE)/data/combined
	@find data -name "*.json" -exec sh -c 'mkdir -p $(DRIVE)/$$(dirname {}); cp {} $(DRIVE)/{}' \;
	@echo "All manifests saved to Drive"

restore-manifests:
	@find $(DRIVE)/data -name "*.json" 2>/dev/null | while read f; do \
		rel=$$(echo $$f | sed 's|$(DRIVE)/||'); \
		mkdir -p $$(dirname $$rel); \
		cp $$f $$rel; \
		echo "Restored: $$rel"; \
	done
	@echo "Manifests restored"

# ── Git ───────────────────────────────────────────────────────────────────────
push:
	git add -A
	git commit -m "update $(shell date '+%Y-%m-%d %H:%M')"
	git push

pull:
	git pull

# ── Utilities ────────────────────────────────────────────────────────────────
list:
	$(PYTHON) main.py --mode list

gpu:
	@nvidia-smi 2>/dev/null || echo "No GPU (expected on Mac)"

status:
	@echo "=== Manifests ==="
	@for f in data/real/manifest.json data/simulated/manifest.json; do \
		if [ -f $$f ]; then \
			echo "  $$f: $$(python3 -c 'import json; print(len(json.load(open("'$$f'"))))', "records")"; \
		else \
			echo "  $$f: NOT FOUND"; \
		fi; \
	done
	@echo ""
	@echo "=== Checkpoints ==="
	@ls checkpoints/ 2>/dev/null || echo "  No checkpoints yet"

clean:
	rm -rf data/real/train_manifest.json data/real/val_manifest.json data/real/test_manifest.json
	rm -rf data/simulated/train_manifest.json data/simulated/val_manifest.json data/simulated/test_manifest.json
	rm -rf data/combined/
	rm -rf results/
	@echo "Cleaned generated files (manifests and checkpoints preserved)"

clean-all: clean
	rm -rf checkpoints/
	rm -rf data/real/manifest.json data/simulated/manifest.json
	@echo "Full clean complete — re-run make export && make data to start fresh"

.PHONY: setup-mac setup-colab export-real export-sim export data \
        train train-all train-simulated train-combined \
        eval figures results save restore save-manifests restore-manifests \
        push pull list gpu status clean clean-all
