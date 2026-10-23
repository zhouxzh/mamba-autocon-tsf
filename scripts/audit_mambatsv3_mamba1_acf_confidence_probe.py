#!/usr/bin/env python3
"""Read-only audit for ACF-confidence-weighted relation pretraining."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ROOT_ARTIFACT = ROOT / "artifacts/exploratory/mambatsv3_mamba1_acf_confidence_probe_20261006"
TARGETS = ("pems_bay", "metr_la")
ARMS = ("null", "plain", "confidence")
SEED = 5981


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    result = {"validation_only": True, "test_metrics_produced": False, "targets": {}, "submission_ready": False}
    for target in TARGETS:
        rows = {}
        for arm in ARMS:
            prefix = ROOT_ARTIFACT / target / f"s{SEED}_{arm}"
            train = _load(prefix.with_name(prefix.name + "_pretrain.json"))
            transfer = _load(prefix.with_name(prefix.name + "_frozen_core.json"))
            if train.get("test_metrics_produced") or transfer.get("test_metrics_produced"):
                raise ValueError(f"test boundary violation: {target}/{arm}")
            cfg = transfer["config"]
            contract = (cfg.get("mamba_ts_mixer_type"), int(cfg["patch_len"]), int(cfg["patch_stride"]),
                        float(cfg["few_shot_fraction"]), bool(cfg["freeze_pretrained_core"]))
            if contract != ("mamba1", 24, 8, .1, True):
                raise ValueError(f"transfer contract mismatch: {target}/{arm}: {contract}")
            pre_cfg = train["config"]
            expected = {"null": (0.0, False), "plain": (0.1, False), "confidence": (0.1, True)}[arm]
            actual = (float(pre_cfg.get("relation_loss_fraction", -1)), bool(pre_cfg.get("forecast_contrastive_acf_weighting", False)))
            if actual != expected:
                raise ValueError(f"pretrain contract mismatch: {target}/{arm}: {actual}")
            history = train.get("history", [])
            if arm != "null" and (not history or min(float(row.get("global_relation_active_fraction", 0.0)) for row in history) < .8):
                raise ValueError(f"insufficient relation coverage: {target}/{arm}")
            rows[arm] = float(transfer["best_validation_mse"])
        null, plain, confidence = (rows["null"], rows["plain"], rows["confidence"])
        result["targets"][target] = {
            "mse": rows,
            "confidence_vs_null_percent": (confidence / null - 1.0) * 100.0,
            "plain_vs_null_percent": (plain / null - 1.0) * 100.0,
            "confidence_vs_plain_percent": (confidence / plain - 1.0) * 100.0,
            "confidence_beats_null": confidence <= null,
            "confidence_beats_plain": confidence <= plain,
        }
    confidence_values = np.asarray([result["targets"][target]["mse"]["confidence"] for target in TARGETS])
    plain_values = np.asarray([result["targets"][target]["mse"]["plain"] for target in TARGETS])
    result["mean_confidence_vs_plain_percent"] = float((confidence_values.mean() / plain_values.mean() - 1.0) * 100.0)
    result["cross_domain_gate"] = bool(all(result["targets"][target]["confidence_beats_plain"] for target in TARGETS))
    result["selection_outcome"] = "positive_mechanism_signal" if result["cross_domain_gate"] else "stop_no_cross_domain_gain"
    (ROOT_ARTIFACT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
