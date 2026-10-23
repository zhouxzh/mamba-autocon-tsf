#!/usr/bin/env python3
"""Read-only audit for d32/24 Electricity SSL replication."""
from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/"artifacts/exploratory/mambatsv3_electricity_mamba1_d32_stride24_transfer_replication_20261006";SEEDS=(5997,5998,5999)
def load(p):return json.loads(p.read_text(encoding="utf-8"))
def main():
 rows=[]
 for seed in SEEDS:
  a,n=load(ART/f"s{seed}_active_frozen_core.json"),load(ART/f"s{seed}_null_frozen_core.json")
  if a.get("test_metrics_produced") or n.get("test_metrics_produced"):raise ValueError("test boundary")
  for arm,x in (("active",a),("null",n)):
   cfg=x["config"]
   if (cfg.get("mamba_ts_mixer_type"),int(cfg["patch_len"]),int(cfg["patch_stride"]),int(cfg["d_model"]),int(cfg["d_ff"]),float(cfg["few_shot_fraction"]),bool(cfg["freeze_pretrained_core"])) != ("mamba1",24,24,32,64,.25,True):raise ValueError(f"contract: {seed}/{arm}")
  rows.append({"seed":seed,"active":float(a["best_validation_mse"]),"null":float(n["best_validation_mse"])})
 active=np.asarray([r["active"] for r in rows]);null=np.asarray([r["null"] for r in rows]);ratio=active/null
 result={"validation_only":True,"test_metrics_produced":False,"seeds":list(SEEDS),"rows":rows,"active_mean_mse":float(active.mean()),"null_mean_mse":float(null.mean()),"active_vs_null_percent":float((active.mean()/null.mean()-1)*100),"active_wins":int((active<null).sum()),"ratio_upper_1se":float(ratio.mean()+ratio.std(ddof=1)/math.sqrt(len(ratio))),"noninferior_at_one_se":bool(ratio.mean()+ratio.std(ddof=1)/math.sqrt(len(ratio))<=1),"submission_ready":False}
 result["external_baseline_eligible"]=bool(result["noninferior_at_one_se"] and result["active_wins"]>=2)
 (ART/"audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps(result,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
