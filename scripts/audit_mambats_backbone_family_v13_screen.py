#!/usr/bin/env python3
"""Read-only validation audit for the registered v3/v13 backbone-family pilot."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/mambats_backbone_family_v13_screen_20261006"
SEEDS = {"pems_bay": (6021, 6022), "metr_la": (6023, 6024)}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary(left: np.ndarray, right: np.ndarray) -> dict:
    ratios = left / right
    return {"left_mean_mse": float(left.mean()), "right_mean_mse": float(right.mean()),
            "percent": float((left.mean() / right.mean() - 1.0) * 100.0),
            "wins": int((left < right).sum()),
            "ratio_upper_1se": float(ratios.mean() + ratios.std(ddof=1) / math.sqrt(len(ratios)))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=sorted(SEEDS), action="append", default=None)
    args = parser.parse_args()
    result = {"validation_only": True, "test_metrics_produced": False, "targets": {}, "submission_ready": False}
    for target in args.target or sorted(SEEDS):
        rows = []
        for seed in SEEDS[target]:
            records = {arm: _load(OUTPUT / target / f"s{seed}_{arm}_frozen_core.json") for arm in ("v3_active", "v13_null", "v13_active")}
            if any(record.get("test_metrics_produced") for record in records.values()):
                raise ValueError("test boundary violated")
            for arm, expected_variant in (("v3_active", "v3"), ("v13_null", "v13"), ("v13_active", "v13")):
                config = records[arm]["config"]
                contract = (config.get("mamba_ts_variant"), config.get("mamba_ts_mixer_type"),
                            int(config["patch_len"]), int(config["patch_stride"]),
                            float(config["few_shot_fraction"]), bool(config["freeze_pretrained_core"]))
                if contract != (expected_variant, "mamba1", 24, 8, .1, True):
                    raise ValueError(f"contract mismatch: {target}/{seed}/{arm}: {contract}")
            rows.append({"seed": seed, **{arm: float(record["best_validation_mse"]) for arm, record in records.items()}})
        v3 = np.asarray([row["v3_active"] for row in rows])
        null = np.asarray([row["v13_null"] for row in rows])
        active = np.asarray([row["v13_active"] for row in rows])
        result["targets"][target] = {"rows": rows, "v13_active_vs_v3_active": _summary(active, v3),
                                      "v13_active_vs_v13_null": _summary(active, null)}
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
