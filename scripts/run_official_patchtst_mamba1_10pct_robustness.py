#!/usr/bin/env python3
"""Evaluate locked official PatchTST 10% checkpoints under candidate stresses."""
from __future__ import annotations
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SPECS={"pems_bay":("dataset/PEMS-BAY/PEMS-BAY.csv",ROOT/"artifacts/exploratory/official_patchtst_pems_mamba1_10pct_20261006",(5960,5961,5962)),"metr_la":("dataset/METR-LA/METR-LA.csv",ROOT/"artifacts/exploratory/official_patchtst_metr_mamba1_10pct_20261006",(5970,5971,5972)),"weather":("dataset/weather/weather.csv",ROOT/"artifacts/exploratory/official_patchtst_mamba1_crossdomain_replication_20261006/weather",(5930,5931,5932))}
def guard(command,prefix):subprocess.run([sys.executable,str(ROOT/"scripts/run_gpu_guarded.py"),"--temperature-limit-c","78","--fan-limit-percent","85","--interval-sec","5","--telemetry-csv",str(prefix.with_name(prefix.name+"_telemetry.csv")),"--event-log",str(prefix.with_name(prefix.name+"_event.json")),"--",*command],cwd=ROOT,check=True)
def main():
 import argparse
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--target",choices=sorted(SPECS),action="append",default=None);parser.add_argument("--resume",action="store_true");args=parser.parse_args()
 targets=tuple(args.target or sorted(SPECS))
 out=ROOT/"artifacts/exploratory/official_patchtst_mamba1_10pct_robustness_20261006";out.mkdir(parents=True,exist_ok=True)
 (out/"plan.json").write_text(json.dumps({"status":"registered_before_evaluation","validation_only":True,"test_metrics_produced":False,"targets":list(SPECS),"conditions":["in_distribution","late_validation_shift","missing_channels_10","missing_channels_30","noise_0p05","noise_0p10"],"comparison":"same normalization, validation origins, deterministic perturbations, and seeds as Mamba-1 10% robustness artifacts"},indent=2)+"\n",encoding="utf-8")
 for target in targets:
  dataset,checkpoint_root,seeds=SPECS[target]
  target_out=out/target;target_out.mkdir(parents=True,exist_ok=True)
  for seed in seeds:
   checkpoint=checkpoint_root/f"{target}_h192_s{seed}_official_patchtst.pt";output=target_out/f"s{seed}.json"
   if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
   if args.resume and output.is_file():continue
   guard([sys.executable,str(ROOT/"scripts/evaluate_official_patchtst_robustness.py"),"--dataset",dataset,"--checkpoint",str(checkpoint),"--output",str(output),"--batch-size","4","--seed",str(seed)],target_out/f"s{seed}")
 return 0
if __name__=="__main__":raise SystemExit(main())
