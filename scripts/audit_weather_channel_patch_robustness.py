#!/usr/bin/env python3
"""Read-only external robustness audit for Weather channel-patch SSL."""

from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "artifacts/exploratory/weather_channel_patch_replication_20261006/robustness_validation"
BASELINE = ROOT / "artifacts/exploratory/mambatsv3_weather_mamba1_10pct_independent_replication_20261006/robustness_validation"
PATCH = ROOT / "artifacts/exploratory/official_patchtst_weather_mamba1_10pct_independent_replication_20261006/robustness_validation"
SEEDS = (5991, 5992, 5993, 5994, 5995)
CONDITIONS = ("in_distribution", "late_validation_shift", "missing_channels_10", "missing_channels_30", "noise_0p05", "noise_0p10")
def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def summary(left, right):
    ratio = left / right
    return {"left_mean_mse": float(left.mean()), "right_mean_mse": float(right.mean()), "percent": float((left.mean()/right.mean()-1)*100), "wins": int((left<right).sum()), "ratio_upper_1se": float(ratio.mean()+ratio.std(ddof=1)/math.sqrt(len(ratio))), "noninferior_at_one_se": bool(ratio.mean()+ratio.std(ddof=1)/math.sqrt(len(ratio))<=1)}
def main():
    result={"validation_only":True,"test_metrics_produced":False,"conditions":{},"submission_ready":False}
    for condition in CONDITIONS:
        rows=[]
        for seed in SEEDS:
            candidate=load(CANDIDATE/f"s{seed}_channel_patch.json");forecast=load(BASELINE/f"s{seed}_active.json");null=load(BASELINE/f"s{seed}_null.json");patch=load(PATCH/f"s{seed}.json")
            if any(item.get("test_metrics_produced") for item in (candidate,forecast,null,patch)): raise ValueError("test boundary violated")
            values={"channel_patch":candidate["conditions"][condition],"forecast":forecast["conditions"][condition],"null":null["conditions"][condition],"official_patchtst":patch["conditions"][condition]}
            first=next(iter(values.values()))
            if any((item["windows"],item["mask_fraction"],item["noise_scale"])!=(first["windows"],first["mask_fraction"],first["noise_scale"]) for item in values.values()): raise ValueError("condition mismatch")
            rows.append({"seed":seed,**{key:float(item["mse"]) for key,item in values.items()}})
        values={key:np.asarray([row[key] for row in rows]) for key in ("channel_patch","forecast","null","official_patchtst")}
        result["conditions"][condition]={"rows":rows,"channel_patch_vs_forecast":summary(values["channel_patch"],values["forecast"]),"channel_patch_vs_null":summary(values["channel_patch"],values["null"]),"channel_patch_vs_official_patchtst":summary(values["channel_patch"],values["official_patchtst"])}
    result["all_external_noninferior"]=all(item["channel_patch_vs_official_patchtst"]["noninferior_at_one_se"] for item in result["conditions"].values())
    (CANDIDATE/"audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps(result,sort_keys=True))
if __name__=="__main__":main()
