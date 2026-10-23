#!/usr/bin/env python3
"""Read-only PEMS-BAY five-seed Mamba-1 versus official TimesNet audit."""
from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];MAMBA=ROOT/"artifacts/exploratory/mambatsv3_pems_mamba1_10pct_independent_replication_20261006";TIMES=ROOT/"artifacts/exploratory/official_timesnet_pems_mamba1_10pct_independent_replication_20261006";SEEDS=(6001,6002,6003,6004,6005)
def load(p):return json.loads(p.read_text(encoding="utf-8"))
def main():
 rows=[]
 for seed in SEEDS:
  a,t=load(MAMBA/f"s{seed}_active_frozen_core.json"),load(TIMES/f"pems_bay_h192_s{seed}_official_timesnet.json")
  if a.get("test_metrics_produced") or t.get("test_metrics_produced"):raise ValueError("test boundary")
  cfg=a["config"];tc=t["config"]
  if (cfg.get("mamba_ts_mixer_type"),int(cfg["patch_len"]),int(cfg["patch_stride"]),float(cfg["few_shot_fraction"]),bool(cfg["freeze_pretrained_core"])) != ("mamba1",24,8,.1,True):raise ValueError(f"candidate contract: {seed}")
  if (int(tc["seq_len"]),int(tc["pred_len"]),float(tc["few_shot_fraction"]),int(tc["epochs"]),int(tc["max_train_windows"]),int(tc["d_model"]),int(tc["d_ff"])) != (96,192,.1,5,64,24,48):raise ValueError(f"TimesNet contract: {seed}")
  rows.append({"seed":seed,"mamba_active":float(a["best_validation_mse"]),"official_timesnet":float(t["best_validation_mse"])})
 active=np.asarray([r["mamba_active"] for r in rows]);times=np.asarray([r["official_timesnet"] for r in rows]);ratio=active/times
 result={"validation_only":True,"test_metrics_produced":False,"rows":rows,"active_mean_mse":float(active.mean()),"official_timesnet_mean_mse":float(times.mean()),"active_vs_timesnet_percent":float((active.mean()/times.mean()-1)*100),"active_wins":int((active<times).sum()),"ratio_upper_1se":float(ratio.mean()+ratio.std(ddof=1)/math.sqrt(len(ratio))),"noninferior_at_one_se":bool(ratio.mean()+ratio.std(ddof=1)/math.sqrt(len(ratio))<=1),"submission_ready":False}
 (TIMES/"audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps(result,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
