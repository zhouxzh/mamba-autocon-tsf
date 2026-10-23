#!/usr/bin/env python3
"""Read-only fresh-seed audit for post-scan mixer replication."""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"artifacts/exploratory/traffic_d32_postscan_mixer_replication_20261007"
SEEDS=(6099,6100,6101)
def load(path): return json.loads(path.read_text(encoding="utf-8"))
def summary(candidate, baseline):
    ratio=candidate/baseline
    return {"percent":float((candidate.mean()/baseline.mean()-1)*100),"wins":int((candidate<baseline).sum()),"ratio_upper_1se":float(ratio.mean()+ratio.std(ddof=1)/math.sqrt(len(ratio))),"noninferior_at_one_se":bool(ratio.mean()+ratio.std(ddof=1)/math.sqrt(len(ratio))<=1)}
def main():
    rows=[]
    for seed in SEEDS:
        values={}
        for arm in ("forecast_calibration","channel_patch","postscan_channel_patch"):
            pre=load(OUTPUT/f"s{seed}_{arm}_pretrain.json"); ft=load(OUTPUT/f"s{seed}_{arm}_frozen_core.json")
            if pre.get("test_metrics_produced") or ft.get("test_metrics_produced"): raise ValueError("test boundary")
            if arm=="postscan_channel_patch":
                if not pre["config"].get("mamba_ts_shared_postscan_mixer"): raise ValueError("mixer config missing")
                if not any(key.startswith("ts_shared_postscan_mixer.") for key in pre["shared_keys"]): raise ValueError("mixer key missing")
                if not all(row["mamba_core_frozen"] for row in ft["history"]): raise ValueError("target core unfrozen")
            values[arm]=float(ft["best_validation_mse"])
        rows.append({"seed":seed,**values})
    forecast=np.asarray([row["forecast_calibration"] for row in rows]); channel=np.asarray([row["channel_patch"] for row in rows]); candidate=np.asarray([row["postscan_channel_patch"] for row in rows])
    vs_forecast=summary(candidate,forecast); vs_channel=summary(candidate,channel)
    result={"validation_only":True,"test_metrics_produced":False,"rows":rows,"postscan_vs_forecast":vs_forecast,"postscan_vs_channel_patch":vs_channel,"replication_passes":bool(vs_forecast["wins"]==len(SEEDS) and vs_channel["wins"]==len(SEEDS) and vs_forecast["noninferior_at_one_se"] and vs_channel["noninferior_at_one_se"]),"submission_ready":False}
    (OUTPUT/"audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8"); print(json.dumps(result,sort_keys=True))
if __name__=="__main__": main()
