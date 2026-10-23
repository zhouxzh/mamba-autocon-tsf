#!/usr/bin/env python3
"""Pair frozen Mamba-1 10% robustness results with official PatchTST stresses."""
from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PATCH=ROOT/"artifacts/exploratory/official_patchtst_mamba1_10pct_robustness_20261006"
MAMBA={"pems_bay":(ROOT/"artifacts/exploratory/mambatsv3_pems_mamba1_10pct_replication_20261006/robustness_validation",(5960,5961,5962)),"metr_la":(ROOT/"artifacts/exploratory/mambatsv3_metr_mamba1_10pct_replication_20261006/robustness_validation",(5970,5971,5972)),"weather":(ROOT/"artifacts/exploratory/mambatsv3_mamba1_crossdomain_replication_20261006/weather/robustness_validation",(5930,5931,5932))}
CONDITIONS=("in_distribution","late_validation_shift","missing_channels_10","missing_channels_30","noise_0p05","noise_0p10")
def load(p):return json.loads(p.read_text(encoding="utf-8"))
def main():
 result={"validation_only":True,"test_metrics_produced":False,"targets":{},"submission_ready":False}
 for target,(mamba_root,seeds) in MAMBA.items():
  summary={}
  for condition in CONDITIONS:
   active=[];patch=[];rows=[]
   for seed in seeds:
    m=load(mamba_root/f"s{seed}_active.json");p=load(PATCH/target/f"s{seed}.json")
    if m.get("test_metrics_produced") or p.get("test_metrics_produced"):raise ValueError(f"test boundary: {target}/{seed}")
    a=float(m["conditions"][condition]["mse"]);b=float(p["conditions"][condition]["mse"]);active.append(a);patch.append(b);rows.append({"seed":seed,"mamba1_active":a,"official_patchtst":b})
   active=np.asarray(active);patch=np.asarray(patch);ratio=active/patch
   summary[condition]={"rows":rows,"active_mean_mse":float(active.mean()),"official_patchtst_mean_mse":float(patch.mean()),"active_vs_patchtst_percent":float((active.mean()/patch.mean()-1)*100),"active_wins":int((active<patch).sum()),"ratio_upper_1se":float(ratio.mean()+ratio.std(ddof=1)/math.sqrt(len(ratio))),"noninferior_at_one_se":bool(ratio.mean()+ratio.std(ddof=1)/math.sqrt(len(ratio))<=1)}
  result["targets"][target]=summary
 result["all_conditions_noninferior"]=all(row["noninferior_at_one_se"] for target in result["targets"].values() for row in target.values())
 result["scope"]="post-hoc validation-only external robustness extension; candidate and PatchTST checkpoints were trained before stress evaluation"
 (PATCH/"audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps(result,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
