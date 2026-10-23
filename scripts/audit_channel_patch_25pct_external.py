#!/usr/bin/env python3
"""Read-only official PatchTST audit for 25% channel-patch SSL replication."""
from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];CANDIDATE=ROOT/"artifacts/exploratory/channel_patch_25pct_replication_20261006";PATCH=ROOT/"artifacts/exploratory/official_patchtst_channel_patch_25pct_replication_20261006";SEEDS={"pems_bay":(6036,6037,6038),"metr_la":(6039,6040,6041)}
def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def summary(left,right):
 ratio=left/right;return {"candidate_mean_mse":float(left.mean()),"official_patchtst_mean_mse":float(right.mean()),"percent":float((left.mean()/right.mean()-1)*100),"wins":int((left<right).sum()),"ratio_upper_1se":float(ratio.mean()+ratio.std(ddof=1)/math.sqrt(len(ratio))),"noninferior_at_one_se":bool(ratio.mean()+ratio.std(ddof=1)/math.sqrt(len(ratio))<=1)}
def main():
 result={"validation_only":True,"test_metrics_produced":False,"targets":{},"submission_ready":False}
 for target,seeds in SEEDS.items():
  rows=[]
  for seed in seeds:
   candidate=load(CANDIDATE/target/f"s{seed}_channel_patch_frozen_core.json");patch=load(PATCH/target/f"{target}_h192_s{seed}_official_patchtst.json")
   if candidate.get("test_metrics_produced") or patch.get("test_metrics_produced"):raise ValueError("test boundary violated")
   if float(candidate["config"]["few_shot_fraction"])!=.25 or float(patch["config"]["few_shot_fraction"])!=.25:raise ValueError("label fraction mismatch")
   rows.append({"seed":seed,"channel_patch":float(candidate["best_validation_mse"]),"official_patchtst":float(patch["best_validation_mse"])})
  left=np.asarray([row["channel_patch"] for row in rows]);right=np.asarray([row["official_patchtst"] for row in rows]);result["targets"][target]={"rows":rows,"channel_patch_vs_official_patchtst":summary(left,right)}
 result["robustness_eligible"]=bool(all(item["channel_patch_vs_official_patchtst"]["noninferior_at_one_se"] for item in result["targets"].values()))
 (CANDIDATE/"external_patchtst_25pct_audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps(result,sort_keys=True))
if __name__=="__main__":main()
