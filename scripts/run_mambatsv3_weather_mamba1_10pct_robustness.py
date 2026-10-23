#!/usr/bin/env python3
"""Validation stress matrix for frozen Weather 10%-label Mamba-1 checkpoints."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CHECKPOINTS=ROOT/"artifacts/exploratory/mambatsv3_mamba1_crossdomain_replication_20261006/weather"
OUTPUT=CHECKPOINTS/"robustness_validation"
SEEDS=(5930,5931,5932)
def guard(command,prefix):subprocess.run([sys.executable,str(ROOT/"scripts/run_gpu_guarded.py"),"--temperature-limit-c","78","--fan-limit-percent","85","--interval-sec","5","--telemetry-csv",str(prefix.with_name(prefix.name+"_telemetry.csv")),"--event-log",str(prefix.with_name(prefix.name+"_event.json")),"--",*command],cwd=ROOT,check=True)
def main():
 OUTPUT.mkdir(parents=True,exist_ok=True)
 (OUTPUT/"plan.json").write_text(json.dumps({"status":"registered_before_evaluation","validation_only":True,"test_metrics_produced":False,"target":"Weather","horizon":192,"few_shot_fraction":.1,"seeds":list(SEEDS),"conditions":["in_distribution","late_validation_shift","missing_channels_10","missing_channels_30","noise_0p05","noise_0p10"],"paired":"same seed and deterministic stress realization for active/null"},indent=2)+"\n",encoding="utf-8")
 for seed in SEEDS:
  for arm in ("active","null"):
   checkpoint=CHECKPOINTS/f"s{seed}_{arm}_frozen_core.pt";output=OUTPUT/f"s{seed}_{arm}.json"
   if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
   guard([sys.executable,str(ROOT/"scripts/evaluate_mambatsv3_lowdata_robustness.py"),"--dataset","dataset/weather/weather.csv","--checkpoint",str(checkpoint),"--output",str(output),"--batch-size","4","--seed",str(seed),"--hybrid"],OUTPUT/f"s{seed}_{arm}")
 return 0
if __name__=="__main__":raise SystemExit(main())
