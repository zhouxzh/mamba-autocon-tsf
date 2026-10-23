#!/usr/bin/env python3
"""Read-only audit for Electricity prototype-set-pooling probe."""
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/"artifacts/exploratory/mambatsv3_electricity_mamba1_prototype_pooling_probe_20261006"; SEED=5989
def load(p): return json.loads(p.read_text(encoding="utf-8"))
def main():
    values={}
    for arm in ("plain_active","prototype_active","prototype_null"):
        prefix=ART/f"s{SEED}_{arm}"; pre,ft=load(prefix.with_name(prefix.name+"_pretrain.json")),load(prefix.with_name(prefix.name+"_frozen_core.json"))
        if pre.get("test_metrics_produced") or ft.get("test_metrics_produced"): raise ValueError("test boundary")
        cfg=ft["config"]
        if (cfg.get("mamba_ts_mixer_type"),int(cfg["patch_len"]),int(cfg["patch_stride"]),float(cfg["few_shot_fraction"]),bool(cfg["freeze_pretrained_core"])) != ("mamba1",24,8,.25,True): raise ValueError(f"transfer contract: {arm}")
        proto=arm.startswith("prototype_")
        if bool(pre["config"].get("mamba_ts_channel_prototype_pooling")) != proto: raise ValueError(f"prototype flag: {arm}")
        if proto and not any("ts_channel_prototype_queries" in k for k in pre["shared_keys"]): raise ValueError(f"prototype key: {arm}")
        if arm!="prototype_null" and min(float(h.get("global_relation_active_fraction",0)) for h in pre["history"])<.8: raise ValueError(f"coverage: {arm}")
        values[arm]=float(ft["best_validation_mse"])
    result={"validation_only":True,"test_metrics_produced":False,"seed":SEED,"mse":values,"prototype_active_vs_null_percent":(values["prototype_active"]/values["prototype_null"]-1)*100,"prototype_active_vs_plain_percent":(values["prototype_active"]/values["plain_active"]-1)*100,"prototype_active_beats_null":values["prototype_active"]<=values["prototype_null"],"prototype_active_beats_plain":values["prototype_active"]<=values["plain_active"],"submission_ready":False}
    result["selection_outcome"]="highdim_positive_signal" if result["prototype_active_beats_null"] and result["prototype_active_beats_plain"] else "stop_no_highdim_gain"
    (ART/"audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps(result,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
