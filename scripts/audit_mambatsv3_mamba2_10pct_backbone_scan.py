#!/usr/bin/env python3
"""Read-only audit for the frozen 10%-label Mamba-1/Mamba-2 backbone scan."""

from __future__ import annotations

import json
import math
import argparse
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCAN = ROOT / "artifacts/exploratory/mambatsv3_mamba2_10pct_backbone_scan_20261006"
MAMBA1 = {
    "pems_bay": ROOT / "artifacts/exploratory/mambatsv3_pems_mamba1_10pct_replication_20261006",
    "metr_la": ROOT / "artifacts/exploratory/mambatsv3_metr_mamba1_10pct_replication_20261006",
}
PATCHTST = {
    "pems_bay": ROOT / "artifacts/exploratory/official_patchtst_pems_mamba1_10pct_20261006",
    "metr_la": ROOT / "artifacts/exploratory/official_patchtst_metr_mamba1_10pct_20261006",
}
SEEDS = {"pems_bay": (5960, 5961, 5962), "metr_la": (5970, 5971, 5972)}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _patch_path(target: str, seed: int) -> Path:
    label = "pems_bay" if target == "pems_bay" else "metr_la"
    return PATCHTST[target] / f"{label}_h192_s{seed}_official_patchtst.json"


def _summary(values: np.ndarray, comparator: np.ndarray) -> dict:
    ratios = values / comparator
    return {
        "mean_mse": float(values.mean()),
        "percent": float((values.mean() / comparator.mean() - 1.0) * 100.0),
        "wins": int((values < comparator).sum()),
        "ratio_upper_1se": float(ratios.mean() + ratios.std(ddof=1) / math.sqrt(len(ratios))),
        "noninferior_at_one_se": bool(ratios.mean() + ratios.std(ddof=1) / math.sqrt(len(ratios)) <= 1.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=sorted(MAMBA1), action="append", default=None)
    args = parser.parse_args()
    targets = tuple(args.target or sorted(MAMBA1))
    result = {"validation_only": True, "test_metrics_produced": False, "targets": {}, "submission_ready": False}
    for target in targets:
        rows = []
        for seed in SEEDS[target]:
            active2 = _load(SCAN / target / f"s{seed}_active_frozen_core.json")
            null2 = _load(SCAN / target / f"s{seed}_null_frozen_core.json")
            active1 = _load(MAMBA1[target] / f"s{seed}_active_frozen_core.json")
            patch = _load(_patch_path(target, seed))
            if any(item.get("test_metrics_produced") for item in (active2, null2, active1, patch)):
                raise ValueError("backbone scan opened test metrics")
            config = active2["config"]
            contract = (config.get("mamba_ts_mixer_type"), int(config["mamba_headdim"]), int(config["patch_len"]),
                        int(config["patch_stride"]), float(config["few_shot_fraction"]), bool(config["freeze_pretrained_core"]))
            if contract != ("mamba2", 12, 24, 8, .1, True):
                raise ValueError(f"Mamba-2 contract mismatch: {target}/{seed}: {contract}")
            if active1["config"].get("mamba_ts_mixer_type") != "mamba1":
                raise ValueError(f"Mamba-1 contract mismatch: {target}/{seed}")
            rows.append({"seed": seed, "mamba2_active": float(active2["best_validation_mse"]),
                         "mamba2_null": float(null2["best_validation_mse"]),
                         "mamba1_active": float(active1["best_validation_mse"]),
                         "official_patchtst": float(patch["best_validation_mse"])})
        m2 = np.asarray([row["mamba2_active"] for row in rows])
        null = np.asarray([row["mamba2_null"] for row in rows])
        m1 = np.asarray([row["mamba1_active"] for row in rows])
        patch = np.asarray([row["official_patchtst"] for row in rows])
        result["targets"][target] = {
            "rows": rows,
            "mamba2_vs_null": _summary(m2, null),
            "mamba2_vs_mamba1": _summary(m2, m1),
            "mamba2_vs_official_patchtst": _summary(m2, patch),
        }
    suffix = "" if len(targets) == len(MAMBA1) else "_" + "_".join(targets)
    (SCAN / f"audit{suffix}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
