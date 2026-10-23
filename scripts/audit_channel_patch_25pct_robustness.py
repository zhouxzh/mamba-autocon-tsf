#!/usr/bin/env python3
"""Read-only external robustness audit for 25% channel-patch SSL."""
from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];RESULTS=ROOT/"artifacts/exploratory/channel_patch_25pct_replication_20261006/robustness_validation";SEEDS={"pems_bay":(6036,6037,6038),"metr_la":(6039,6040,6041)};CONDITIONS=("in_distribution","late_validation_shift","missing_channels_10","missing_channels_30","noise_0p05","noise_0p10")
def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def summary(left,right):
 ratio=left/right;return {"left_mean_mse":float(left.mean()),"right_mean_mse":float(right.mean()),"percent":float((left.mean()/right.mean()-1)*100),"wins":int((left<right).sum()),"ratio_upper_1se":float(ratio.mean()+ratio.std(ddof=1)/math.sqrt(len(ratio))),"noninferior_at_one_se":bool(ratio.mean()+ratio.std(ddof=1)/math.sqrt(len(ratio))<=1)}
def main():
 result={"validation_only":True,"test_metrics_produced":False,"targets":{},"submission_ready":False}
 for target,seeds in SEEDS.items():
  conditions={}
  for condition in CONDITIONS:
   rows=[]
   for seed in seeds:
    candidate=load(RESULTS/f"{target}_s{seed}_channel_patch.json");patch=load(RESULTS/f"{target}_s{seed}_official_patchtst.json")
    if candidate.get("test_metrics_produced") or patch.get("test_metrics_produced"):raise ValueError("test boundary violated")
    left,right=candidate["conditions"][condition],patch["conditions"][condition]
    if (left["windows"],left["mask_fraction"],left["noise_scale"])!=(right["windows"],right["mask_fraction"],right["noise_scale"]):raise ValueError("paired condition mismatch")
    rows.append({"seed":seed,"channel_patch":float(left["mse"]),"official_patchtst":float(right["mse"])})
   left=np.asarray([row["channel_patch"] for row in rows]);right=np.asarray([row["official_patchtst"] for row in rows]);conditions[condition]={"rows":rows,"channel_patch_vs_official_patchtst":summary(left,right)}
  result["targets"][target]={"conditions":conditions,"all_external_noninferior":all(item["channel_patch_vs_official_patchtst"]["noninferior_at_one_se"] for item in conditions.values())}
 result["all_external_noninferior"]=all(item["all_external_noninferior"] for item in result["targets"].values())
 (RESULTS/"audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps(result,sort_keys=True))
if __name__=="__main__":main()
