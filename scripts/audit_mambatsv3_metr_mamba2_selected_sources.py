#!/usr/bin/env python3
"""Audit ACF-selected-source Mamba-2 METR-LA transfer versus PatchTST."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "artifacts/exploratory/mambatsv3_metr_mamba2_selected_sources_probe_20261006"
BASELINE = ROOT / "artifacts/exploratory/official_patchtst_metr_mamba2_patch24_comparator_20261006"
SEEDS = (5891, 5892, 5893)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    rows = []
    for seed in SEEDS:
        active = _load(ACTIVE / f"s{seed}_active_frozen_core.json")
        null = _load(ACTIVE / f"s{seed}_null_frozen_core.json")
        patchtst = _load(BASELINE / f"metr_la_h192_s{seed}_official_patchtst.json")
        if any(value.get("test_metrics_produced") for value in (active, null, patchtst)):
            raise ValueError("comparison opened test metrics")
        cfg, reference = active["config"], patchtst["config"]
        if (cfg.get("mamba_ts_mixer_type"), int(cfg["mamba_headdim"]), int(cfg["patch_len"]),
                int(cfg["patch_stride"]), bool(cfg["freeze_pretrained_core"])) != ("mamba2", 12, 24, 8, True):
            raise ValueError(f"Mamba contract mismatch: {seed}")
        if (int(reference["patch_len"]), int(reference["patch_stride"]),
                float(reference["few_shot_fraction"])) != (24, 8, .25):
            raise ValueError(f"PatchTST contract mismatch: {seed}")
        left, control, external = (float(active["best_validation_mse"]),
                                   float(null["best_validation_mse"]),
                                   float(patchtst["best_validation_mse"]))
        rows.append({"seed": seed, "active_mse": left, "null_mse": control,
                     "official_patchtst_mse": external, "active_vs_null_ratio": left / control,
                     "active_vs_patchtst_ratio": left / external,
                     "active_beats_null": bool(left < control),
                     "active_beats_patchtst": bool(left < external)})
    active = np.asarray([row["active_mse"] for row in rows])
    null = np.asarray([row["null_mse"] for row in rows])
    patchtst = np.asarray([row["official_patchtst_mse"] for row in rows])
    ratios = active / patchtst
    result = {
        "validation_only": True, "test_metrics_produced": False, "rows": rows,
        "mean_active_mse": float(active.mean()), "mean_null_mse": float(null.mean()),
        "mean_official_patchtst_mse": float(patchtst.mean()),
        "active_vs_null_percent": float((active.mean() / null.mean() - 1) * 100),
        "active_vs_null_wins": int((active < null).sum()),
        "active_vs_patchtst_percent": float((active.mean() / patchtst.mean() - 1) * 100),
        "active_vs_patchtst_wins": int((active < patchtst).sum()),
        "active_vs_patchtst_ratio_upper_1se": float(ratios.mean() + ratios.std(ddof=1) / math.sqrt(len(ratios))),
        "mamba_contract": {"mixer": "mamba2", "headdim": 12, "patch_len": 24,
                           "patch_stride": 8, "target_core_frozen": True},
        "source_selection": "top-3 target-few-shot train-prefix ACF cosine similarity; target validation/test excluded",
        "selection_status": "exploratory: source pool and later seeds followed validation selection",
        "submission_ready": False,
    }
    result["active_vs_patchtst_noninferior_at_one_se"] = bool(
        result["active_vs_patchtst_ratio_upper_1se"] <= 1.0
    )
    (ACTIVE / "official_patchtst_metr_mamba2_selected_sources_audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
