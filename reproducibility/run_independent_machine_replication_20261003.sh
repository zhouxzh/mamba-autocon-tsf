#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 /absolute/path/to/validated/python" >&2
  exit 2
fi

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON_BIN=$1
OUT=artifacts/reproducibility/independent_machine_replication_20261003
SEED=5612

cd "$REPO_ROOT"
sha256sum -c reproducibility/clean_install_replication_checksums_20261003.txt
"$PYTHON_BIN" scripts/verify_clean_install_environment.py \
  --output "$OUT/environment_check.json"

run_guarded() {
  local name=$1
  shift
  env AUTOCON_CPU_THREAD_LIMIT=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 NUMEXPR_THREADS=4 "$PYTHON_BIN" scripts/run_gpu_guarded.py \
    --temperature-limit-c 78 --fan-limit-percent 85 --interval-sec 5 \
    --telemetry-csv "$OUT/${name}_telemetry.csv" \
    --event-log "$OUT/${name}_event.json" -- "$@"
}

common_pretrain=(
  scripts/pretrain_mambatsv3_core_autocon.py
  --source PEMS07:dataset/PEMS07/PEMS07.npz:npz
  --source PEMS08:dataset/PEMS08/PEMS08.npz:npz
  --source PEMS04:dataset/PEMS04/PEMS04.npz:npz
  --initial-checkpoint artifacts/paper_final/pems_mambatsv3_d24_test_release_20260926/candidate/checkpoints/long_term_forecast_plan_confirmatory_pems_d24_candidate_test_pems03_MambaTSv3AutoCon_h12_s2131_l0p001_ca786fbe2493c_MambaTSv3AutoCon_pems03_ftM_sl96_ll48_pl12_dm2_h00dcc565bb3086f9/checkpoint.pth
  --initial-checkpoint artifacts/paper_final/pems_mambatsv3_d24_test_release_20260926/candidate/checkpoints/long_term_forecast_plan_confirmatory_pems_d24_candidate_test_pems04_MambaTSv3AutoCon_h12_s2131_l0p001_c22b7150b652d_MambaTSv3AutoCon_pems04_ftM_sl96_ll48_pl12_dm2_h179e4641a67e8f06/checkpoint.pth
  --seq-len 96 --patch-len 24 --patch-stride 24 --source-pred-len 192
  --d-model 24 --d-ff 48 --mamba-d-state 8 --mamba-d-conv 2 --mamba-expand 1
  --context-rank 8 --train-fraction .6 --batch-size 4 --epochs 3 --max-windows 64
  --relation-pair-policy contrastive_cluster --forecast-contrastive-pairing channel_coverage
  --min-channel-positive-support .5 --reconstruction-weight 0 --predictive-weight 1
  --prediction-target last_value_residual --pretrain-task next_patch
  --relation-view forecast_contrastive --learning-rate .0003 --seed "$SEED"
)

run_guarded active_pretrain "$PYTHON_BIN" "${common_pretrain[@]}" \
  --output "$OUT/active_pretrain.pt" --mamba-lambda .1 --relation-loss-fraction .1
run_guarded null_pretrain "$PYTHON_BIN" "${common_pretrain[@]}" \
  --output "$OUT/null_pretrain.pt" --mamba-lambda 0 --relation-loss-fraction 0

common_finetune=(
  scripts/finetune_mambatsv3_dlinear_hybrid.py
  --dataset dataset/PEMS-BAY/PEMS-BAY.csv --seq-len 96 --pred-len 192
  --patch-len 24 --patch-stride 24 --d-model 24 --d-ff 48
  --mamba-d-state 8 --mamba-d-conv 2 --mamba-expand 1 --context-rank 16
  --channel-agnostic-scan --transfer-prediction-head --prediction-head-transfer-mode exact
  --linear-model NLinear --mamba-correction-output --few-shot-fraction .1
  --train-fraction .6 --val-fraction .2 --batch-size 4 --epochs 5
  --max-train-windows 64 --freeze-mamba-epochs 1 --seed "$SEED"
)

run_guarded active_finetune "$PYTHON_BIN" "${common_finetune[@]}" \
  --source-checkpoint "$OUT/active_pretrain.pt" --output "$OUT/active_hybrid.pt"
run_guarded null_finetune "$PYTHON_BIN" "${common_finetune[@]}" \
  --source-checkpoint "$OUT/null_pretrain.pt" --output "$OUT/null_hybrid.pt"

ACTIVE_VAL=$("$PYTHON_BIN" -c "import json; print(json.load(open('$OUT/active_hybrid.json'))['best_validation_mse'])")
NULL_VAL=$("$PYTHON_BIN" -c "import json; print(json.load(open('$OUT/null_hybrid.json'))['best_validation_mse'])")
if ! "$PYTHON_BIN" -c "raise SystemExit(0 if float('$ACTIVE_VAL') <= float('$NULL_VAL') else 1)"; then
  echo "Validation gate failed; target test remains closed." >&2
  exit 1
fi

for arm in active null; do
  run_guarded "${arm}_test" "$PYTHON_BIN" scripts/evaluate_lowdata_checkpoint_test.py \
    --dataset dataset/PEMS-BAY/PEMS-BAY.csv --checkpoint "$OUT/${arm}_hybrid.pt" \
    --output "$OUT/${arm}_test.json" --kind hybrid --batch-size 4 --seed "$SEED"
done
