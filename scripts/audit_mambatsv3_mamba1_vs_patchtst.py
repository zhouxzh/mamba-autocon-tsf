#!/usr/bin/env python3
"""Audit validation-only Mamba-1 frozen-core transfer versus official PatchTST."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-stem", default="pems_bay_h192")
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds = tuple(args.seed)
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seed values must be unique")
    rows = []
    for seed in seeds:
        active = _load(args.active / f"s{seed}_active_frozen_core.json")
        null = _load(args.active / f"s{seed}_null_frozen_core.json")
        baseline = _load(args.baseline / f"{args.baseline_stem}_s{seed}_official_patchtst.json")
        if any(value.get("test_metrics_produced") for value in (active, null, baseline)):
            raise ValueError("comparison opened test metrics")
        cfg, reference = active["config"], baseline["config"]
        if (cfg.get("mamba_ts_mixer_type"), int(cfg["patch_len"]), int(cfg["patch_stride"]),
                bool(cfg["freeze_pretrained_core"])) != ("mamba1", 24, 8, True):
            raise ValueError(f"Mamba-1 contract mismatch: {seed}")
        if (int(reference["patch_len"]), int(reference["patch_stride"])) != (24, 8):
            raise ValueError(f"PatchTST geometry mismatch: {seed}")
        candidate, control, external = (float(active["best_validation_mse"]),
                                        float(null["best_validation_mse"]),
                                        float(baseline["best_validation_mse"]))
        rows.append({"seed": seed, "active_mse": candidate, "null_mse": control,
                     "official_patchtst_mse": external, "active_vs_null_ratio": candidate / control,
                     "active_vs_patchtst_ratio": candidate / external,
                     "active_beats_null": candidate < control,
                     "active_beats_patchtst": candidate < external})
    active = np.asarray([row["active_mse"] for row in rows])
    null = np.asarray([row["null_mse"] for row in rows])
    baseline = np.asarray([row["official_patchtst_mse"] for row in rows])
    ratios = active / baseline
    upper = float(ratios.mean() + ratios.std(ddof=1) / math.sqrt(len(ratios)))
    result = {
        "validation_only": True, "test_metrics_produced": False, "rows": rows,
        "mean_mamba1_active": float(active.mean()), "mean_mamba1_null": float(null.mean()),
        "mean_official_patchtst": float(baseline.mean()),
        "active_vs_null_percent": float((active.mean() / null.mean() - 1) * 100),
        "active_vs_null_wins": int((active < null).sum()),
        "active_vs_patchtst_percent": float((active.mean() / baseline.mean() - 1) * 100),
        "active_vs_patchtst_wins": int((active < baseline).sum()),
        "active_vs_patchtst_ratio_upper_1se": upper,
        "active_vs_patchtst_noninferior_at_one_se": bool(upper <= 1.0),
        "mamba_contract": {"mixer": "mamba1", "patch_len": 24, "patch_stride": 8,
                           "target_core_frozen": True},
        "submission_ready": False,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
