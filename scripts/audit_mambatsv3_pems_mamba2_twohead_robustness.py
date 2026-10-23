#!/usr/bin/env python3
"""Audit paired validation stresses for frozen Mamba-2 two-head transfer."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = ROOT / "artifacts/exploratory/mambatsv3_pems_mamba2_twohead_probe_20261006"
RESULT_ROOT = CHECKPOINT_ROOT / "robustness_validation"
SEEDS = (5880, 5881, 5882)
CONDITIONS = (
    "in_distribution", "late_validation_shift", "missing_channels_10",
    "missing_channels_30", "noise_0p05", "noise_0p10",
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    rows = {condition: [] for condition in CONDITIONS}
    for seed in SEEDS:
        checkpoint = _load(CHECKPOINT_ROOT / f"s{seed}_active_frozen_core.json")
        config = checkpoint["config"]
        contract = (
            config.get("mamba_ts_mixer_type"), int(config.get("mamba_headdim")),
            int(config.get("patch_len")), int(config.get("patch_stride")),
            bool(config.get("freeze_pretrained_core")), bool(checkpoint.get("test_metrics_produced")),
        )
        if contract != ("mamba2", 12, 24, 8, True, False):
            raise ValueError(f"checkpoint {seed} violates Mamba-2 frozen-core contract")
        active = _load(RESULT_ROOT / f"s{seed}_active.json")
        null = _load(RESULT_ROOT / f"s{seed}_null.json")
        if active.get("test_metrics_produced") or null.get("test_metrics_produced"):
            raise ValueError("robustness evaluation opened test metrics")
        if set(active["conditions"]) != set(CONDITIONS) or set(null["conditions"]) != set(CONDITIONS):
            raise ValueError("stress condition sets do not match")
        for condition in CONDITIONS:
            left, right = active["conditions"][condition], null["conditions"][condition]
            for key in ("windows", "mask_fraction", "noise_scale"):
                if left[key] != right[key]:
                    raise ValueError(f"condition {condition} differs between paired arms")
            rows[condition].append({
                "seed": seed, "active_mse": float(left["mse"]), "null_mse": float(right["mse"]),
                "ratio": float(left["mse"]) / float(right["mse"]),
                "active_better": bool(float(left["mse"]) < float(right["mse"])),
                "windows": int(left["windows"]), "mask_fraction": float(left["mask_fraction"]),
                "noise_scale": float(left["noise_scale"]),
            })
    summary = {}
    for condition, values in rows.items():
        active = np.asarray([item["active_mse"] for item in values])
        null = np.asarray([item["null_mse"] for item in values])
        ratios = active / null
        upper = float(ratios.mean() + ratios.std(ddof=1) / math.sqrt(len(ratios)))
        summary[condition] = {
            "mean_active_mse": float(active.mean()), "mean_null_mse": float(null.mean()),
            "active_vs_null_percent": float((active.mean() / null.mean() - 1.0) * 100.0),
            "active_wins": int((active < null).sum()), "ratio_mean": float(ratios.mean()),
            "ratio_upper_1se": upper, "noninferior_at_one_se": bool(upper <= 1.0),
            "windows": values[0]["windows"], "mask_fraction": values[0]["mask_fraction"],
            "noise_scale": values[0]["noise_scale"],
        }
    result = {
        "validation_only": True, "test_metrics_produced": False,
        "mamba_contract": {"mixer": "mamba2", "headdim": 12, "patch_len": 24, "patch_stride": 8,
                           "target_core_frozen": True},
        "rows": rows, "summary": summary,
        "scope": "same-seed paired validation stresses; not an independent architecture confirmation",
        "selection_status": "architecture selected adaptively on PEMS-BAY validation before robustness readout",
        "submission_ready": False,
    }
    output = RESULT_ROOT / "audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
