#!/usr/bin/env python3
"""Read-only audit for current 24/8 PEMS H336 channel-patch replication."""
from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUTPUT=ROOT/"artifacts/exploratory/pems_h336_channel_patch_replication_20261006";PATCH=ROOT/"artifacts/exploratory/official_patchtst_pems_h336_channel_patch_replication_20261006";SEEDS=(6052,6053,6054)
def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def summary(left,right):
 ratio=left/right;return {"left_mean_mse":float(left.mean()),"right_mean_mse":float(right.mean()),"percent":float((left.mean()/right.mean()-1)*100),"wins":int((left<right).sum()),"ratio_upper_1se":float(ratio.mean()+ratio.std(ddof=1)/math.sqrt(len(ratio))),"noninferior_at_one_se":bool(ratio.mean()+ratio.std(ddof=1)/math.sqrt(len(ratio))<=1)}
def main():
 rows=[]
 for seed in SEEDS:
  records={arm:load(OUTPUT/f"s{seed}_{arm}_frozen_core.json") for arm in ("null","forecast","channel_patch")};patch=load(PATCH/f"pems_bay_h336_s{seed}_official_patchtst.json");source=load(OUTPUT/f"s{seed}_channel_patch_pretrain.json")
  if any(item.get("test_metrics_produced") for item in (*records.values(),patch,source)):raise ValueError("test boundary")
  if any(int(item["config"]["pred_len"])!=336 for item in (*records.values(),patch)):raise ValueError("horizon mismatch")
  if any(int(item["config"]["patch_stride"])!=8 for item in (*records.values(),patch)):raise ValueError("patch contract mismatch")
  if not all(float(item.get("channel_patch_relation_active_fraction",0))>0 for item in source["history"]):raise ValueError("inactive patch anchors")
  rows.append({"seed":seed,**{arm:float(item["best_validation_mse"]) for arm,item in records.items()},"official_patchtst":float(patch["best_validation_mse"])})
 values={name:np.asarray([row[name] for row in rows]) for name in ("null","forecast","channel_patch","official_patchtst")}
 result={"validation_only":True,"test_metrics_produced":False,"rows":rows,"channel_patch_vs_forecast":summary(values["channel_patch"],values["forecast"]),"channel_patch_vs_null":summary(values["channel_patch"],values["null"]),"channel_patch_vs_official_patchtst":summary(values["channel_patch"],values["official_patchtst"]),"submission_ready":False}
 result["robustness_eligible"]=bool(result["channel_patch_vs_forecast"]["wins"]>=2 and result["channel_patch_vs_forecast"]["noninferior_at_one_se"] and result["channel_patch_vs_official_patchtst"]["noninferior_at_one_se"])
 (OUTPUT/"audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps(result,sort_keys=True))
if __name__=="__main__":main()
