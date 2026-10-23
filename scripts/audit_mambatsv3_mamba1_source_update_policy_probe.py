#!/usr/bin/env python3
"""Read-only audit for source-domain optimizer-order probe."""

from __future__ import annotations
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/exploratory/mambatsv3_mamba1_source_update_policy_probe_20261006"
SEED = 5983
TARGETS = ("pems_bay", "metr_la")

def load(p): return json.loads(p.read_text(encoding="utf-8"))
def main():
    result = {"validation_only": True, "test_metrics_produced": False, "targets": {}, "submission_ready": False}
    for target in TARGETS:
        values = {}
        for arm in ("null_sequential", "active_sequential", "null_cyclic", "active_cyclic"):
            prefix = ART / target / f"s{SEED}_{arm}"
            pre, ft = load(prefix.with_name(prefix.name + "_pretrain.json")), load(prefix.with_name(prefix.name + "_frozen_core.json"))
            if pre.get("test_metrics_produced") or ft.get("test_metrics_produced"): raise ValueError(f"test boundary: {target}/{arm}")
            cfg = ft["config"]
            if (cfg.get("mamba_ts_mixer_type"), int(cfg["patch_len"]), int(cfg["patch_stride"]), float(cfg["few_shot_fraction"]), bool(cfg["freeze_pretrained_core"])) != ("mamba1", 24, 8, .1, True): raise ValueError(f"transfer contract: {target}/{arm}")
            values[arm] = float(ft["best_validation_mse"])
            expected_policy = "cyclic" if arm.endswith("cyclic") else "sequential"
            if pre["config"].get("source_update_policy") != expected_policy: raise ValueError(f"policy mismatch: {target}/{arm}")
            orders = [row.get("source_update_order") for row in pre.get("history", [])]
            if len(orders) != 3: raise ValueError(f"missing source order history: {target}/{arm}")
            if expected_policy == "cyclic" and len(set(tuple(order) for order in orders)) != 3: raise ValueError(f"cyclic order did not rotate: {target}/{arm}")
        result["targets"][target] = {"mse": values, "active_sequential_vs_null_percent": (values["active_sequential"] / values["null_sequential"] - 1) * 100, "active_cyclic_vs_null_percent": (values["active_cyclic"] / values["null_cyclic"] - 1) * 100, "cyclic_vs_sequential_active_percent": (values["active_cyclic"] / values["active_sequential"] - 1) * 100, "cyclic_beats_sequential_active": values["active_cyclic"] <= values["active_sequential"]}
    result["cross_domain_gate"] = all(item["cyclic_beats_sequential_active"] for item in result["targets"].values())
    result["selection_outcome"] = "positive_mechanism_signal" if result["cross_domain_gate"] else "stop_no_cross_domain_gain"
    (ART / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); print(json.dumps(result, sort_keys=True)); return 0
if __name__ == "__main__": raise SystemExit(main())
