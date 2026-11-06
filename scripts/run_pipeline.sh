#!/usr/bin/env bash
# Full pipeline for one config: baseline -> SFT -> eval -> DPO pairs -> DPO -> eval -> report.
# Usage: scripts/run_pipeline.sh configs/cpu_small.yaml [--skip-zeroshot]
set -euo pipefail
CFG=${1:-configs/cpu_small.yaml}
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
PY=${PYTHON:-python}
step() { echo; echo "=== [$(date +%H:%M:%S)] $* ==="; }
RES=$($PY -c "import yaml,sys;print(yaml.safe_load(open('$CFG'))['results_dir'])")
OUT=$($PY -c "import yaml,sys;print(yaml.safe_load(open('$CFG'))['output_dir'])")
# Resumable: a stage is skipped when its artefact already exists (FORCE=1 to rerun everything).
have() { [[ -z "${FORCE:-}" && -e "$1" ]]; }
if [[ "${2:-}" != "--skip-zeroshot" ]]; then
  have "$RES/metrics_baseline_zeroshot.json" || { step "baseline (zero-shot prompt)"; $PY -m toolcall_ft.evaluate --config "$CFG" --stage baseline_zeroshot; }
fi
have "$RES/metrics_baseline_fewshot.json" || { step "baseline (engineered few-shot prompt)"; $PY -m toolcall_ft.evaluate --config "$CFG" --stage baseline_fewshot; }
have "$OUT/sft_adapter/adapter_config.json" || { step "phase 1: SFT"; $PY -m toolcall_ft.train_sft --config "$CFG"; }
have "$RES/metrics_sft.json" || { step "eval SFT"; $PY -m toolcall_ft.evaluate --config "$CFG" --stage sft; }
have "$OUT/dpo_pairs.jsonl" || { step "phase 2a: sample + auto-label DPO pairs"; $PY -m toolcall_ft.build_dpo_pairs --config "$CFG"; }
have "$OUT/dpo_adapter/adapter_config.json" || { step "phase 2b: DPO on top of SFT"; $PY -m toolcall_ft.train_dpo --config "$CFG"; }
have "$RES/metrics_sft_dpo.json" || { step "eval SFT+DPO"; $PY -m toolcall_ft.evaluate --config "$CFG" --stage sft_dpo; }
step "report";                            $PY -m toolcall_ft.report --config "$CFG"
step "done"
