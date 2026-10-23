#!/usr/bin/env python3
"""Audit paired validation stresses for selected-source Mamba-2 METR-LA."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = ROOT / "artifacts/exploratory/mambatsv3_metr_mamba2_selected_sources_probe_20261006"
RESULT_ROOT = CHECKPOINT_ROOT / "robustness_validation"
SEEDS = (5891, 5892, 5893)
CONDITIONS = ("in_distribution", "late_validation_shift", "missing_channels_10",
              "missing_channels_30", "noise_0p05", "noise_0p10")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    rows = {name: [] for name in CONDITIONS}
    for seed in SEEDS:
        checkpoint = _load(CHECKPOINT_ROOT / f"s{seed}_active_frozen_core.json")
        cfg = checkpoint["config"]
        if (cfg.get("mamba_ts_mixer_type"), int(cfg["mamba_headdim"]), int(cfg["patch_len"]),
                int(cfg["patch_stride"]), bool(cfg["freeze_pretrained_core"]),
                bool(checkpoint["test_metrics_produced"])) != ("mamba2", 12, 24, 8, True, False):
            raise ValueError(f"checkpoint contract mismatch: {seed}")
        active, null = (_load(RESULT_ROOT / f"s{seed}_active.json"),
                        _load(RESULT_ROOT / f"s{seed}_null.json"))
        if active["test_metrics_produced"] or null["test_metrics_produced"]:
            raise ValueError("evaluation opened test metrics")
        for condition in CONDITIONS:
            left, right = active["conditions"][condition], null["conditions"][condition]
            for key in ("windows", "mask_fraction", "noise_scale"):
                if left[key] != right[key]:
                    raise ValueError(f"paired condition mismatch: {condition}")
            rows[condition].append({"seed": seed, "active_mse": float(left["mse"]),
                                    "null_mse": float(right["mse"]),
                                    "ratio": float(left["mse"]) / float(right["mse"]),
                                    "active_better": bool(float(left["mse"]) < float(right["mse"]))})
    summary = {}
    for condition, cells in rows.items():
        active = np.asarray([cell["active_mse"] for cell in cells])
        null = np.asarray([cell["null_mse"] for cell in cells])
        ratios = active / null
        upper = float(ratios.mean() + ratios.std(ddof=1) / math.sqrt(len(ratios)))
        summary[condition] = {"mean_active_mse": float(active.mean()), "mean_null_mse": float(null.mean()),
                              "active_vs_null_percent": float((active.mean() / null.mean() - 1) * 100),
                              "active_wins": int((active < null).sum()), "ratio_upper_1se": upper,
                              "noninferior_at_one_se": bool(upper <= 1.0)}
    result = {"validation_only": True, "test_metrics_produced": False, "rows": rows,
              "summary": summary, "submission_ready": False,
              "scope": "same-seed paired validation stresses after adaptive source selection"}
    (RESULT_ROOT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
