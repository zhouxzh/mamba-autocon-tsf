#!/usr/bin/env python3
"""Read-only five-seed Weather external robustness audit."""
from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];MAMBA=ROOT/"artifacts/exploratory/mambatsv3_weather_mamba1_10pct_independent_replication_20261006/robustness_validation";PATCH=ROOT/"artifacts/exploratory/official_patchtst_weather_mamba1_10pct_independent_replication_20261006/robustness_validation";SEEDS=(5991,5992,5993,5994,5995);CONDITIONS=("in_distribution","late_validation_shift","missing_channels_10","missing_channels_30","noise_0p05","noise_0p10")
def load(p):return json.loads(p.read_text(encoding="utf-8"))
def main():
 result={"validation_only":True,"test_metrics_produced":False,"seeds":list(SEEDS),"conditions":{},"submission_ready":False}
 for condition in CONDITIONS:
  active=[];null=[];patch=[];rows=[]
  for seed in SEEDS:
   a,n,p=load(MAMBA/f"s{seed}_active.json"),load(MAMBA/f"s{seed}_null.json"),load(PATCH/f"s{seed}.json")
   if a.get("test_metrics_produced") or n.get("test_metrics_produced") or p.get("test_metrics_produced"):raise ValueError("test boundary")
   av=float(a["conditions"][condition]["mse"]);nv=float(n["conditions"][condition]["mse"]);pv=float(p["conditions"][condition]["mse"]);active.append(av);null.append(nv);patch.append(pv);rows.append({"seed":seed,"active":av,"null":nv,"official_patchtst":pv})
  active=np.asarray(active);null=np.asarray(null);patch=np.asarray(patch);ratio=active/patch
  result["conditions"][condition]={"rows":rows,"active_mean_mse":float(active.mean()),"null_mean_mse":float(null.mean()),"official_patchtst_mean_mse":float(patch.mean()),"active_vs_null_percent":float((active.mean()/null.mean()-1)*100),"active_vs_patchtst_percent":float((active.mean()/patch.mean()-1)*100),"active_wins_vs_null":int((active<null).sum()),"active_wins_vs_patchtst":int((active<patch).sum()),"ratio_upper_1se":float(ratio.mean()+ratio.std(ddof=1)/math.sqrt(len(ratio))),"noninferior_at_one_se":bool(ratio.mean()+ratio.std(ddof=1)/math.sqrt(len(ratio))<=1)}
 result["all_conditions_noninferior"]=all(row["noninferior_at_one_se"] for row in result["conditions"].values())
 (MAMBA/"external_patchtst_audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps(result,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
