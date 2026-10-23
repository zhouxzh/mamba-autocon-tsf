#!/usr/bin/env python3
"""Audit the three-seed Electricity strong target-calibration relation probe."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/mambatsv3_electricity_mamba1_strong_calibration_relation_probe_20261006"
SEEDS = (5948, 5949, 5950)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    rows = []
    for seed in SEEDS:
        active = _load(OUTPUT / f"s{seed}_active_frozen_core.json")
        null = _load(OUTPUT / f"s{seed}_null_frozen_core.json")
        source = _load(OUTPUT / f"s{seed}_active_pretrain.json")
        if active.get("test_metrics_produced") or null.get("test_metrics_produced"):
            raise ValueError("probe opened test metrics")
        cfg = active["config"]
        if (cfg.get("mamba_ts_mixer_type"), int(cfg["patch_len"]), int(cfg["patch_stride"]), bool(cfg["freeze_pretrained_core"])) != ("mamba1", 24, 8, True):
            raise ValueError(f"target contract mismatch: {seed}")
        roles = {item["name"]: item for item in source["sources"]}
        if roles["Traffic"]["relation_loss_fraction"] != .1 or roles["Electricity"]["relation_loss_fraction"] != .3:
            raise ValueError(f"relation schedule mismatch: {seed}")
        if roles["Electricity"]["role"] != "target_calibration" or roles["Electricity"]["train_prefix_fraction"] != .15:
            raise ValueError(f"calibration prefix contract mismatch: {seed}")
        if not all(record["global_relation_active_fraction"] > 0.0 for record in source["history"]):
            raise ValueError(f"active relation path did not execute: {seed}")
        candidate, control = float(active["best_validation_mse"]), float(null["best_validation_mse"])
        rows.append({"seed": seed, "active_validation_mse": candidate, "null_validation_mse": control,
                     "ratio": candidate / control, "active_better": candidate < control})
    active = np.asarray([row["active_validation_mse"] for row in rows])
    null = np.asarray([row["null_validation_mse"] for row in rows])
    ratios = active / null
    result = {
        "validation_only": True, "test_metrics_produced": False, "rows": rows,
        "active_mean_validation_mse": float(active.mean()), "null_mean_validation_mse": float(null.mean()),
        "active_vs_null_percent": float((active.mean() / null.mean() - 1) * 100),
        "active_wins": int((active < null).sum()),
        "ratio_upper_1se": float(ratios.mean() + ratios.std(ddof=1) / math.sqrt(len(ratios))),
        "noninferior_at_one_se": bool(ratios.mean() + ratios.std(ddof=1) / math.sqrt(len(ratios)) <= 1.0),
        "protocol": "Mamba-1 24/8, frozen core, Traffic source at relation fraction 0.1 plus Electricity train-prefix calibration at 0.3",
        "external_patchtst_validation_mean_mse": None,
        "external_baseline_status": "not part of this causal audit; validation comparator is lower than candidate and test remains closed",
        "submission_ready": False,
    }
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
