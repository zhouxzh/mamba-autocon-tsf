#!/usr/bin/env python3
"""Read-only paired audit for d32/24 Electricity Mamba versus PatchTST."""
from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];MAMBA=ROOT/"artifacts/exploratory/mambatsv3_electricity_mamba1_d32_stride24_transfer_replication_20261006";PATCH=ROOT/"artifacts/exploratory/official_patchtst_electricity_d32_stride24_comparator_20261006";SEEDS=(5997,5998,5999)
def load(p):return json.loads(p.read_text(encoding="utf-8"))
def main():
 rows=[]
 for seed in SEEDS:
  a=load(MAMBA/f"s{seed}_active_frozen_core.json");p=load(PATCH/f"electricity_h96_s{seed}_official_patchtst.json")
  if a.get("test_metrics_produced") or p.get("test_metrics_produced"):raise ValueError("test boundary")
  cfg=a["config"];pc=p["config"]
  if (cfg.get("mamba_ts_mixer_type"),int(cfg["patch_len"]),int(cfg["patch_stride"]),int(cfg["d_model"]),int(cfg["d_ff"]),float(cfg["few_shot_fraction"]),bool(cfg["freeze_pretrained_core"])) != ("mamba1",24,24,32,64,.25,True):raise ValueError(f"candidate contract: {seed}")
  if (int(pc["patch_len"]),int(pc["patch_stride"]),int(pc["d_model"]),int(pc["d_ff"]),float(pc["few_shot_fraction"])) != (24,24,32,64,.25):raise ValueError(f"PatchTST contract: {seed}")
  rows.append({"seed":seed,"mamba_active":float(a["best_validation_mse"]),"official_patchtst":float(p["best_validation_mse"])})
 active=np.asarray([r["mamba_active"] for r in rows]);patch=np.asarray([r["official_patchtst"] for r in rows]);ratio=active/patch
 result={"validation_only":True,"test_metrics_produced":False,"rows":rows,"active_mean_mse":float(active.mean()),"official_patchtst_mean_mse":float(patch.mean()),"active_vs_patchtst_percent":float((active.mean()/patch.mean()-1)*100),"active_wins":int((active<patch).sum()),"ratio_upper_1se":float(ratio.mean()+ratio.std(ddof=1)/math.sqrt(len(ratio))),"noninferior_at_one_se":bool(ratio.mean()+ratio.std(ddof=1)/math.sqrt(len(ratio))<=1),"submission_ready":False}
 (PATCH/"audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps(result,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
