#!/usr/bin/env python3
"""Read-only H336 channel-patch external robustness audit."""
from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/"artifacts/exploratory/pems_h336_channel_patch_replication_20261006/robustness_validation";SEEDS=(6052,6053,6054)
CONDITIONS=("in_distribution","late_validation_shift","missing_channels_10","missing_channels_30","noise_0p05","noise_0p10")
def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def summary(left,right):
 ratio=left/right;return {"left_mean_mse":float(left.mean()),"right_mean_mse":float(right.mean()),"percent":float((left.mean()/right.mean()-1)*100),"wins":int((left<right).sum()),"ratio_upper_1se":float(ratio.mean()+ratio.std(ddof=1)/math.sqrt(len(ratio))),"noninferior_at_one_se":bool(ratio.mean()+ratio.std(ddof=1)/math.sqrt(len(ratio))<=1)}
def main():
 result={"validation_only":True,"test_metrics_produced":False,"conditions":{},"submission_ready":False}
 for condition in CONDITIONS:
  rows=[]
  for seed in SEEDS:
   values={arm:load(RESULTS/f"s{seed}_{arm}.json") for arm in ("null","forecast","channel_patch","official_patchtst")}
   if any(item.get("test_metrics_produced") for item in values.values()):raise ValueError("test boundary")
   selected={arm:values[arm]["conditions"][condition] for arm in values};first=next(iter(selected.values()))
   if any((item["windows"],item["mask_fraction"],item["noise_scale"])!=(first["windows"],first["mask_fraction"],first["noise_scale"]) for item in selected.values()):raise ValueError("condition mismatch")
   rows.append({"seed":seed,**{arm:float(item["mse"]) for arm,item in selected.items()}})
  values={arm:np.asarray([row[arm] for row in rows]) for arm in ("null","forecast","channel_patch","official_patchtst")}
  result["conditions"][condition]={"rows":rows,"channel_patch_vs_null":summary(values["channel_patch"],values["null"]),"channel_patch_vs_forecast":summary(values["channel_patch"],values["forecast"]),"channel_patch_vs_official_patchtst":summary(values["channel_patch"],values["official_patchtst"])}
 result["all_external_noninferior"]=all(row["channel_patch_vs_official_patchtst"]["noninferior_at_one_se"] for row in result["conditions"].values())
 (RESULTS/"audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps(result,sort_keys=True))
if __name__=="__main__":main()
