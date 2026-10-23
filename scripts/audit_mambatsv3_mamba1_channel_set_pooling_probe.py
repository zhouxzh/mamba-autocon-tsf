#!/usr/bin/env python3
"""Read-only audit for learned channel-set pooling transfer."""
from __future__ import annotations
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/exploratory/mambatsv3_mamba1_channel_set_pooling_probe_20261006"
TARGETS = ("pems_bay", "metr_la"); SEED = 5984
def load(path): return json.loads(path.read_text(encoding="utf-8"))
def main():
    result = {"validation_only": True, "test_metrics_produced": False, "targets": {}, "submission_ready": False}
    for target in TARGETS:
        values = {}
        for arm in ("active", "null"):
            prefix = ART / target / f"s{SEED}_{arm}"; pre, ft = load(prefix.with_name(prefix.name + "_pretrain.json")), load(prefix.with_name(prefix.name + "_frozen_core.json"))
            if pre.get("test_metrics_produced") or ft.get("test_metrics_produced"): raise ValueError(f"test boundary: {target}/{arm}")
            cfg = ft["config"]
            if (cfg.get("mamba_ts_mixer_type"), int(cfg["patch_len"]), int(cfg["patch_stride"]), float(cfg["few_shot_fraction"]), bool(cfg["freeze_pretrained_core"])) != ("mamba1", 24, 8, .1, True): raise ValueError(f"transfer contract: {target}/{arm}")
            if not bool(pre["config"].get("mamba_ts_channel_set_pooling")): raise ValueError(f"pooling flag missing: {target}/{arm}")
            if not any("ts_channel_set_pooling_score.weight" in key for key in pre.get("shared_keys", [])): raise ValueError(f"pooling core was not transferred: {target}/{arm}")
            if arm == "active" and min(float(row.get("global_relation_active_fraction", 0.0)) for row in pre["history"]) < .8: raise ValueError(f"relation coverage: {target}")
            values[arm] = float(ft["best_validation_mse"])
        result["targets"][target] = {"active_mse": values["active"], "null_mse": values["null"], "active_vs_null_percent": (values["active"] / values["null"] - 1) * 100, "active_beats_null": values["active"] <= values["null"]}
    result["cross_domain_gate"] = all(row["active_beats_null"] for row in result["targets"].values()); result["selection_outcome"] = "positive_mechanism_signal" if result["cross_domain_gate"] else "stop_no_cross_domain_gain"
    (ART / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); print(json.dumps(result, sort_keys=True)); return 0
if __name__ == "__main__": raise SystemExit(main())
