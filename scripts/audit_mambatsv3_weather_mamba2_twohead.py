#!/usr/bin/env python3
"""Audit Weather Mamba-2 two-head transfer against paired null and PatchTST."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "artifacts/exploratory/mambatsv3_weather_mamba2_twohead_probe_20261006"
BASELINE = ROOT / "artifacts/exploratory/official_patchtst_weather_mamba2_patch24_comparator_20261006"
SEEDS = (5900, 5901, 5902)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _upper(values: np.ndarray) -> float:
    return float(values.mean() + values.std(ddof=1) / math.sqrt(len(values)))


def main() -> int:
    rows = []
    for seed in SEEDS:
        active = _load(ACTIVE / f"s{seed}_active_frozen_core.json")
        null = _load(ACTIVE / f"s{seed}_null_frozen_core.json")
        patchtst = _load(BASELINE / f"weather_h192_s{seed}_official_patchtst.json")
        if any(item.get("test_metrics_produced") for item in (active, null, patchtst)):
            raise ValueError("comparison opened test metrics")
        cfg = active["config"]
        required = (cfg.get("mamba_ts_mixer_type"), int(cfg["mamba_headdim"]),
                    int(cfg["patch_len"]), int(cfg["patch_stride"]), bool(cfg["freeze_pretrained_core"]))
        if required != ("mamba2", 12, 24, 8, True):
            raise ValueError(f"Mamba-2 contract mismatch for seed {seed}")
        patch_cfg = patchtst["config"]
        if (int(patch_cfg["patch_len"]), int(patch_cfg["patch_stride"]),
                float(patch_cfg["few_shot_fraction"])) != (24, 8, .1):
            raise ValueError(f"PatchTST geometry mismatch for seed {seed}")
        left, middle, right = (float(active["best_validation_mse"]),
                               float(null["best_validation_mse"]),
                               float(patchtst["best_validation_mse"]))
        rows.append({"seed": seed, "active_mse": left, "null_mse": middle,
                     "official_patchtst_mse": right, "active_vs_null_ratio": left / middle,
                     "active_vs_patchtst_ratio": left / right,
                     "active_beats_null": bool(left < middle),
                     "active_beats_patchtst": bool(left < right)})
    active = np.asarray([row["active_mse"] for row in rows])
    null = np.asarray([row["null_mse"] for row in rows])
    patchtst = np.asarray([row["official_patchtst_mse"] for row in rows])
    patch_ratios = active / patchtst
    result = {
        "validation_only": True, "test_metrics_produced": False,
        "rows": rows,
        "mean_active_mse": float(active.mean()), "mean_null_mse": float(null.mean()),
        "mean_official_patchtst_mse": float(patchtst.mean()),
        "active_vs_null_percent": float((active.mean() / null.mean() - 1) * 100),
        "active_vs_null_wins": int((active < null).sum()),
        "active_vs_patchtst_percent": float((active.mean() / patchtst.mean() - 1) * 100),
        "active_vs_patchtst_wins": int((active < patchtst).sum()),
        "active_vs_patchtst_ratio_upper_1se": _upper(patch_ratios),
        "active_vs_patchtst_noninferior_at_one_se": bool(_upper(patch_ratios) <= 1.0),
        "mamba_contract": {"mixer": "mamba2", "headdim": 12, "patch_len": 24,
                           "patch_stride": 8, "target_core_frozen": True},
        "selection_status": "exploratory: seeds 5901-5902 were added after seed 5900",
        "submission_ready": False,
    }
    (ACTIVE / "official_patchtst_weather_mamba2_validation_audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
