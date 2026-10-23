#!/usr/bin/env python3
"""Audit PEMS-BAY 10%-label frozen Mamba-1 active/null/PatchTST validation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "artifacts/exploratory/mambatsv3_pems_mamba1_10pct_replication_20261006"
BASELINE = ROOT / "artifacts/exploratory/official_patchtst_pems_mamba1_10pct_20261006"
SEEDS = (5960, 5961, 5962)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    rows = []
    for seed in SEEDS:
        active = _load(ACTIVE / f"s{seed}_active_frozen_core.json")
        null = _load(ACTIVE / f"s{seed}_null_frozen_core.json")
        patch = _load(BASELINE / f"pems_bay_h192_s{seed}_official_patchtst.json")
        if any(item.get("test_metrics_produced") for item in (active, null, patch)):
            raise ValueError("validation matrix opened test metrics")
        cfg, external = active["config"], patch["config"]
        if (cfg.get("mamba_ts_mixer_type"), int(cfg["patch_len"]), int(cfg["patch_stride"]),
                float(cfg["few_shot_fraction"]), bool(cfg["freeze_pretrained_core"])) != ("mamba1", 24, 8, .1, True):
            raise ValueError(f"candidate contract mismatch: {seed}")
        if (int(external["patch_len"]), int(external["patch_stride"]), float(external["few_shot_fraction"])) != (24, 8, .1):
            raise ValueError(f"PatchTST contract mismatch: {seed}")
        candidate, control, baseline = (float(active["best_validation_mse"]), float(null["best_validation_mse"]), float(patch["best_validation_mse"]))
        rows.append({"seed": seed, "active_mse": candidate, "null_mse": control,
                     "official_patchtst_mse": baseline, "active_better_null": candidate < control,
                     "active_better_patchtst": candidate < baseline, "patchtst_ratio": candidate / baseline})
    active = np.asarray([row["active_mse"] for row in rows])
    null = np.asarray([row["null_mse"] for row in rows])
    patch = np.asarray([row["official_patchtst_mse"] for row in rows])
    ratios = active / patch
    result = {
        "validation_only": True, "test_metrics_produced": False, "rows": rows,
        "active_mean_validation_mse": float(active.mean()), "null_mean_validation_mse": float(null.mean()),
        "official_patchtst_mean_validation_mse": float(patch.mean()),
        "active_vs_null_percent": float((active.mean() / null.mean() - 1) * 100),
        "active_vs_null_wins": int((active < null).sum()),
        "active_vs_patchtst_percent": float((active.mean() / patch.mean() - 1) * 100),
        "active_vs_patchtst_wins": int((active < patch).sum()),
        "ratio_upper_1se": float(ratios.mean() + ratios.std(ddof=1) / math.sqrt(len(ratios))),
        "noninferior_at_one_se": bool(ratios.mean() + ratios.std(ddof=1) / math.sqrt(len(ratios)) <= 1.0),
        "submission_ready": False,
    }
    (ACTIVE / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
