#!/usr/bin/env python3
"""Read-only audit for the frozen learned channel-set pooling replication."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/exploratory/mambatsv3_mamba1_channel_set_pooling_replication_20261006"
SEEDS = (5985, 5986, 5987); TARGETS = ("pems_bay", "metr_la")
def load(p): return json.loads(p.read_text(encoding="utf-8"))
def summary(left, right):
    ratio = left / right
    return {"left_mean_mse": float(left.mean()), "right_mean_mse": float(right.mean()), "percent": float((left.mean()/right.mean()-1)*100), "wins": int((left < right).sum()), "ratio_upper_1se": float(ratio.mean()+ratio.std(ddof=1)/math.sqrt(len(ratio))), "noninferior_at_one_se": bool(ratio.mean()+ratio.std(ddof=1)/math.sqrt(len(ratio)) <= 1)}
def main():
    result = {"validation_only": True, "test_metrics_produced": False, "targets": {}, "submission_ready": False}
    for target in TARGETS:
        rows=[]
        for seed in SEEDS:
            values={}
            for arm in ("pool_active", "pool_null", "plain_active"):
                prefix=ART/target/f"s{seed}_{arm}"; pre,ft=load(prefix.with_name(prefix.name+"_pretrain.json")),load(prefix.with_name(prefix.name+"_frozen_core.json"))
                if pre.get("test_metrics_produced") or ft.get("test_metrics_produced"): raise ValueError(f"test boundary: {target}/{seed}/{arm}")
                cfg=ft["config"]
                if (cfg.get("mamba_ts_mixer_type"),int(cfg["patch_len"]),int(cfg["patch_stride"]),float(cfg["few_shot_fraction"]),bool(cfg["freeze_pretrained_core"])) != ("mamba1",24,8,.1,True): raise ValueError(f"transfer contract: {target}/{seed}/{arm}")
                expected_pool = arm.startswith("pool_")
                if bool(pre["config"].get("mamba_ts_channel_set_pooling")) != expected_pool: raise ValueError(f"pooling flag: {target}/{seed}/{arm}")
                if expected_pool and not any("ts_channel_set_pooling_score.weight" in key for key in pre["shared_keys"]): raise ValueError(f"pooling key: {target}/{seed}/{arm}")
                if arm != "pool_null" and min(float(h.get("global_relation_active_fraction",0)) for h in pre["history"]) < .8: raise ValueError(f"relation coverage: {target}/{seed}/{arm}")
                values[arm]=float(ft["best_validation_mse"])
            rows.append({"seed":seed,**values})
        active=np.asarray([r["pool_active"] for r in rows]); null=np.asarray([r["pool_null"] for r in rows]); plain=np.asarray([r["plain_active"] for r in rows])
        result["targets"][target]={"rows":rows,"pool_active_vs_null":summary(active,null),"pool_active_vs_plain_active":summary(active,plain)}
    result["cross_domain_gate"] = all(v["pool_active_vs_null"]["noninferior_at_one_se"] and v["pool_active_vs_plain_active"]["noninferior_at_one_se"] and v["pool_active_vs_plain_active"]["percent"] <= 0 for v in result["targets"].values())
    result["selection_outcome"] = "replicated_positive_mechanism" if result["cross_domain_gate"] else "stop_no_cross_domain_replication"
    (ART/"audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8"); print(json.dumps(result,sort_keys=True)); return 0
if __name__ == "__main__": raise SystemExit(main())
