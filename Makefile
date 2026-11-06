# Usage: make <target> [CONFIG=configs/cpu_small.yaml] [PYTHON=python]
CONFIG ?= configs/cpu_small.yaml
PYTHON ?= python
export PYTHONPATH := $(CURDIR)/src:$(PYTHONPATH)

.PHONY: install install-gpu data baseline sft dpo-pairs dpo eval-sft eval-dpo groq report pipeline test lint format axolotl-export clean

install:          ## CPU install
	$(PYTHON) -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt
install-gpu:      ## GPU install (CUDA torch from PyPI + bitsandbytes)
	$(PYTHON) -m pip install -r requirements.txt -r requirements-gpu.txt -r requirements-dev.txt
data:             ## download glaive-function-calling-v2 (pinned revision) and build splits
	$(PYTHON) -m toolcall_ft.prepare_data --config $(CONFIG)
baseline:         ## evaluate the base model with zero-shot and few-shot prompts
	$(PYTHON) -m toolcall_ft.evaluate --config $(CONFIG) --stage baseline_zeroshot
	$(PYTHON) -m toolcall_ft.evaluate --config $(CONFIG) --stage baseline_fewshot
sft:
	$(PYTHON) -m toolcall_ft.train_sft --config $(CONFIG)
eval-sft:
	$(PYTHON) -m toolcall_ft.evaluate --config $(CONFIG) --stage sft
dpo-pairs:
	$(PYTHON) -m toolcall_ft.build_dpo_pairs --config $(CONFIG)
dpo:
	$(PYTHON) -m toolcall_ft.train_dpo --config $(CONFIG)
eval-dpo:
	$(PYTHON) -m toolcall_ft.evaluate --config $(CONFIG) --stage sft_dpo
groq:             ## optional strong-model reference (needs GROQ_API_KEY)
	$(PYTHON) -m toolcall_ft.groq_reference --config $(CONFIG) --full
report:           ## plots + results table
	$(PYTHON) -m toolcall_ft.report --config $(CONFIG)
pipeline:         ## everything, in order
	PYTHON=$(PYTHON) bash scripts/run_pipeline.sh $(CONFIG)
axolotl-export:
	$(PYTHON) -m toolcall_ft.export_axolotl
test:
	$(PYTHON) -m pytest
lint:
	$(PYTHON) -m ruff check src tests
format:
	$(PYTHON) -m ruff format src tests
clean:
	rm -rf outputs .pytest_cache .ruff_cache
