#!/usr/bin/env python3
"""Read-only audit for the ACF-stability-weighted Mamba-1 probe."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/exploratory/mambatsv3_mamba1_acf_stability_probe_20261006"
TARGETS, ARMS, SEED = ("pems_bay", "metr_la"), ("null", "plain", "stability"), 5982


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    result = {"validation_only": True, "test_metrics_produced": False, "targets": {}, "submission_ready": False}
    for target in TARGETS:
        values = {}
        for arm in ARMS:
            prefix = ARTIFACT / target / f"s{SEED}_{arm}"
            pretrain, transfer = _load(prefix.with_name(prefix.name + "_pretrain.json")), _load(prefix.with_name(prefix.name + "_frozen_core.json"))
            if pretrain.get("test_metrics_produced") or transfer.get("test_metrics_produced"):
                raise ValueError(f"test boundary violation: {target}/{arm}")
            cfg = transfer["config"]
            if (cfg.get("mamba_ts_mixer_type"), int(cfg["patch_len"]), int(cfg["patch_stride"]), float(cfg["few_shot_fraction"]), bool(cfg["freeze_pretrained_core"])) != ("mamba1", 24, 8, .1, True):
                raise ValueError(f"transfer contract mismatch: {target}/{arm}")
            pre_cfg = pretrain["config"]
            expected = {"null": (0.0, False), "plain": (.1, False), "stability": (.1, True)}[arm]
            actual = (float(pre_cfg.get("relation_loss_fraction", -1)), bool(pre_cfg.get("forecast_contrastive_stability_weighting", False)))
            if actual != expected:
                raise ValueError(f"pretrain contract mismatch: {target}/{arm}: {actual}")
            if arm != "null" and min(float(row.get("global_relation_active_fraction", 0.0)) for row in pretrain["history"]) < .8:
                raise ValueError(f"relation coverage failed: {target}/{arm}")
            values[arm] = float(transfer["best_validation_mse"])
        result["targets"][target] = {
            "mse": values,
            "stability_vs_null_percent": (values["stability"] / values["null"] - 1.0) * 100.0,
            "plain_vs_null_percent": (values["plain"] / values["null"] - 1.0) * 100.0,
            "stability_vs_plain_percent": (values["stability"] / values["plain"] - 1.0) * 100.0,
            "stability_beats_plain": values["stability"] <= values["plain"],
        }
    stability = np.asarray([result["targets"][target]["mse"]["stability"] for target in TARGETS])
    plain = np.asarray([result["targets"][target]["mse"]["plain"] for target in TARGETS])
    result["mean_stability_vs_plain_percent"] = float((stability.mean() / plain.mean() - 1.0) * 100.0)
    result["cross_domain_gate"] = bool(all(result["targets"][target]["stability_beats_plain"] for target in TARGETS))
    result["selection_outcome"] = "positive_mechanism_signal" if result["cross_domain_gate"] else "stop_no_cross_domain_gain"
    (ARTIFACT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
