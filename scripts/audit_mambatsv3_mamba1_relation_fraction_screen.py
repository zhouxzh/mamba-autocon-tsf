#!/usr/bin/env python3
"""Read-only audit for source-only Mamba-1 relation-fraction selection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / "artifacts/exploratory/mambatsv3_mamba1_relation_fraction_screen_20261006"
FRACTIONS = (0.05, 0.10, 0.20, 0.30)
TARGETS = ("pems_bay", "metr_la")
SEED = 5980


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _tag(fraction: float) -> str:
    return "r" + str(fraction).replace(".", "p")


def _mse(target: str, tag: str) -> tuple[float, dict, dict]:
    payload = _load(SCREEN / target / f"s{SEED}_{tag}_frozen_core.json")
    pretrain = _load(SCREEN / target / f"s{SEED}_{tag}_pretrain.json")
    if payload.get("test_metrics_produced") or pretrain.get("test_metrics_produced"):
        raise ValueError(f"test boundary violation: {target}/{tag}")
    config = payload["config"]
    common = (config.get("mamba_ts_mixer_type"), int(config["patch_len"]), int(config["patch_stride"]),
              float(config["few_shot_fraction"]), bool(config["freeze_pretrained_core"]))
    if common != ("mamba1", 24, 8, .1, True):
        raise ValueError(f"candidate contract mismatch: {target}/{tag}: {common}")
    return float(payload["best_validation_mse"]), config, pretrain["config"]


def main() -> int:
    plan = _load(SCREEN / "plan.json")
    if plan.get("validation_only") is not True or plan.get("test_metrics_produced"):
        raise ValueError("invalid plan boundary")
    result = {"validation_only": True, "test_metrics_produced": False, "seed": SEED, "targets": {}, "fractions": {}}
    by_fraction = {fraction: [] for fraction in FRACTIONS}
    for target in TARGETS:
        null, null_config, null_pretrain_config = _mse(target, "null")
        if float(null_pretrain_config.get("mamba_lambda", -1)) != 0 or float(null_pretrain_config.get("relation_loss_fraction", -1)) != 0:
            raise ValueError(f"null relation contract mismatch: {target}")
        target_rows = {"null_mse": null, "fractions": {}}
        for fraction in FRACTIONS:
            active, config, pretrain_config = _mse(target, _tag(fraction))
            if (float(pretrain_config.get("mamba_lambda", -1)), float(pretrain_config.get("relation_loss_fraction", -1))) != (.1, fraction):
                raise ValueError(f"active relation contract mismatch: {target}/{fraction}")
            target_rows["fractions"][str(fraction)] = {
                "active_mse": active,
                "vs_null_percent": (active / null - 1.0) * 100.0,
                "beats_null": active <= null,
            }
            by_fraction[fraction].append(active)
        result["targets"][target] = target_rows
    baseline = np.asarray(by_fraction[.1])
    for fraction, values in by_fraction.items():
        values = np.asarray(values)
        per_target = [result["targets"][target]["fractions"][str(fraction)] for target in TARGETS]
        result["fractions"][str(fraction)] = {
            "mean_active_mse": float(values.mean()),
            "mean_vs_fraction_0.1_percent": float((values.mean() / baseline.mean() - 1.0) * 100.0),
            "beats_or_ties_null_on_both_targets": bool(all(row["beats_null"] for row in per_target)),
            "improves_or_ties_fraction_0.1_on_both_targets": bool(np.all(values <= baseline)),
        }
    eligible = [fraction for fraction in FRACTIONS if (
        result["fractions"][str(fraction)]["beats_or_ties_null_on_both_targets"]
        and result["fractions"][str(fraction)]["improves_or_ties_fraction_0.1_on_both_targets"]
        and fraction != .1
    )]
    result["replication_eligible_fractions"] = eligible
    result["selection_outcome"] = "eligible" if eligible else "stop_no_cross_domain_fraction"
    result["submission_ready"] = False
    (SCREEN / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
