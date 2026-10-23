#!/usr/bin/env python3
"""Validation stresses for independent five-seed PEMS-BAY Mamba-1 replication."""
from __future__ import annotations
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CHECKPOINTS=ROOT/"artifacts/exploratory/mambatsv3_pems_mamba1_10pct_independent_replication_20261006";OUTPUT=CHECKPOINTS/"robustness_validation";SEEDS=(6001,6002,6003,6004,6005)
def guard(c,p):subprocess.run([sys.executable,str(ROOT/"scripts/run_gpu_guarded.py"),"--temperature-limit-c","78","--fan-limit-percent","85","--interval-sec","5","--telemetry-csv",str(p.with_name(p.name+"_telemetry.csv")),"--event-log",str(p.with_name(p.name+"_event.json")),"--",*c],cwd=ROOT,check=True)
def main():
 OUTPUT.mkdir(parents=True,exist_ok=True);(OUTPUT/"plan.json").write_text(json.dumps({"status":"registered_before_evaluation","validation_only":True,"test_metrics_produced":False,"target":"PEMS-BAY","horizon":192,"few_shot_fraction":.1,"seeds":list(SEEDS),"conditions":["in_distribution","late_validation_shift","missing_channels_10","missing_channels_30","noise_0p05","noise_0p10"]},indent=2)+"\n",encoding="utf-8")
 for seed in SEEDS:
  for arm in ("active","null"):
   checkpoint=CHECKPOINTS/f"s{seed}_{arm}_frozen_core.pt";output=OUTPUT/f"s{seed}_{arm}.json"
   if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
   guard([sys.executable,str(ROOT/"scripts/evaluate_mambatsv3_lowdata_robustness.py"),"--dataset","dataset/PEMS-BAY/PEMS-BAY.csv","--checkpoint",str(checkpoint),"--output",str(output),"--batch-size","4","--seed",str(seed),"--hybrid"],OUTPUT/f"s{seed}_{arm}")
 return 0
if __name__=="__main__":raise SystemExit(main())
