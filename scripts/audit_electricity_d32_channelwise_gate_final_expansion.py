#!/usr/bin/env python3
"""Final ten-seed Electricity d32 channelwise-gate versus PatchTST audit."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/exploratory/electricity_d32_channelwise_residual_gate_replication_rerun_20261006"
BASE_PATCH = ROOT / "artifacts/exploratory/official_patchtst_electricity_d32_channelwise_gate_20261006"
EXP = ROOT / "artifacts/exploratory/electricity_d32_channelwise_gate_expansion_20261006"
OUTPUT = ROOT / "artifacts/exploratory/electricity_d32_channelwise_gate_final_expansion_20261007"
PRIOR_SEEDS = (6047, 6048, 6049, 6050, 6051)
FRESH_SEEDS = (6094, 6095, 6096, 6097, 6098)


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def prior_paths(seed):
    root = BASE if seed < 6050 else EXP
    patch_root = BASE_PATCH if seed < 6050 else EXP
    return root / f"s{seed}_channelwise_gate.json", patch_root / f"electricity_h96_s{seed}_official_patchtst.json"


def main():
    plan = load(OUTPUT / "plan.json")
    if plan.get("stop_rule") is None or len(plan.get("prior_seeds", [])) != 5: raise ValueError("final expansion registration failure")
    rows = []
    for seed in PRIOR_SEEDS:
        candidate_path, patch_path = prior_paths(seed); candidate, patch = load(candidate_path), load(patch_path)
        rows.append({"seed": seed, "channelwise_gate": float(candidate["best_validation_mse"]), "official_patchtst": float(patch["best_validation_mse"]), "cohort": "prior"})
    for seed in FRESH_SEEDS:
        candidate = load(OUTPUT / f"s{seed}_channelwise_gate.json"); patch = load(OUTPUT / f"electricity_h96_s{seed}_official_patchtst.json"); event = load(OUTPUT / f"electricity_h96_s{seed}_official_patchtst_patch_event.json")
        if candidate.get("test_metrics_produced") or patch.get("test_metrics_produced"): raise ValueError("test boundary")
        if not candidate["config"].get("mamba_channelwise_residual_scale"): raise ValueError("candidate gate missing")
        if event.get("status") != "completed" or event.get("trigger") is not None: raise ValueError("fresh PatchTST guard failure")
        rows.append({"seed": seed, "channelwise_gate": float(candidate["best_validation_mse"]), "official_patchtst": float(patch["best_validation_mse"]), "cohort": "fresh"})
    candidate = np.asarray([row["channelwise_gate"] for row in rows]); patch = np.asarray([row["official_patchtst"] for row in rows]); ratio = candidate / patch
    result = {"validation_only": True, "test_metrics_produced": False, "final_precision_expansion": True, "rows": rows, "channelwise_gate_vs_official_patchtst": {"candidate_mean_mse": float(candidate.mean()), "official_patchtst_mean_mse": float(patch.mean()), "percent": float((candidate.mean() / patch.mean() - 1) * 100), "wins": int((candidate < patch).sum()), "ratio_upper_1se": float(ratio.mean() + ratio.std(ddof=1) / math.sqrt(len(ratio))), "noninferior_at_one_se": bool(ratio.mean() + ratio.std(ddof=1) / math.sqrt(len(ratio)) <= 1)}, "submission_ready": False}
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); print(json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()
