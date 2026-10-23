#!/usr/bin/env python3
"""Read-only audit for Electricity set-pooling strong-calibration probe."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/exploratory/mambatsv3_electricity_mamba1_set_pooling_strong_calibration_probe_20261006"
SEED = 5988


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    values = {}
    for arm in ("plain_active", "pool_active", "pool_null"):
        prefix = ARTIFACT / f"s{SEED}_{arm}"
        pretrain = _load(prefix.with_name(prefix.name + "_pretrain.json"))
        finetune = _load(prefix.with_name(prefix.name + "_frozen_core.json"))
        if pretrain.get("test_metrics_produced") or finetune.get("test_metrics_produced"):
            raise ValueError("test boundary violation")
        config = finetune["config"]
        contract = (config.get("mamba_ts_mixer_type"), int(config["patch_len"]),
                    int(config["patch_stride"]), float(config["few_shot_fraction"]),
                    bool(config["freeze_pretrained_core"]))
        if contract != ("mamba1", 24, 8, .25, True):
            raise ValueError(f"transfer contract mismatch: {arm}")
        expected_pool = arm.startswith("pool_")
        if bool(pretrain["config"].get("mamba_ts_channel_set_pooling")) != expected_pool:
            raise ValueError(f"pooling flag mismatch: {arm}")
        if expected_pool and not any("ts_channel_set_pooling_score.weight" in key for key in pretrain["shared_keys"]):
            raise ValueError(f"pooling key missing: {arm}")
        if arm != "pool_null" and min(float(row.get("global_relation_active_fraction", 0.0)) for row in pretrain["history"]) < .8:
            raise ValueError(f"relation coverage insufficient: {arm}")
        values[arm] = float(finetune["best_validation_mse"])
    result = {
        "validation_only": True, "test_metrics_produced": False, "seed": SEED,
        "mse": values,
        "pool_active_vs_pool_null_percent": (values["pool_active"] / values["pool_null"] - 1.0) * 100.0,
        "pool_active_vs_plain_active_percent": (values["pool_active"] / values["plain_active"] - 1.0) * 100.0,
        "pool_active_beats_null": values["pool_active"] <= values["pool_null"],
        "pool_active_beats_plain": values["pool_active"] <= values["plain_active"],
        "submission_ready": False,
    }
    result["selection_outcome"] = (
        "highdim_positive_signal"
        if result["pool_active_beats_null"] and result["pool_active_beats_plain"]
        else "stop_no_highdim_gain"
    )
    (ARTIFACT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
