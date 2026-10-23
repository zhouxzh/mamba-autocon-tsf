#!/usr/bin/env python3
"""Read-only six-condition validation stresses for 25% channel-patch SSL."""
from __future__ import annotations
import json,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];CANDIDATE=ROOT/"artifacts/exploratory/channel_patch_25pct_replication_20261006";PATCH=ROOT/"artifacts/exploratory/official_patchtst_channel_patch_25pct_replication_20261006";OUTPUT=CANDIDATE/"robustness_validation";SEEDS={"pems_bay":(6036,6037,6038),"metr_la":(6039,6040,6041)};DATASETS={"pems_bay":"dataset/PEMS-BAY/PEMS-BAY.csv","metr_la":"dataset/METR-LA/METR-LA.csv"};CONDITIONS=["in_distribution","late_validation_shift","missing_channels_10","missing_channels_30","noise_0p05","noise_0p10"]
def guard(command,prefix):subprocess.run([sys.executable,str(ROOT/"scripts/run_gpu_guarded.py"),"--temperature-limit-c","78","--fan-limit-percent","85","--interval-sec","5","--telemetry-csv",str(prefix.with_name(prefix.name+"_telemetry.csv")),"--event-log",str(prefix.with_name(prefix.name+"_event.json")),"--",*command],cwd=ROOT,check=True)
def complete(path):
 return Path(path).is_file() and json.loads(Path(path).read_text(encoding="utf-8")).get("status")=="completed" and json.loads(Path(path).read_text(encoding="utf-8")).get("trigger") is None
def main():
 import argparse
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--target",choices=sorted(SEEDS),action="append",default=None);parser.add_argument("--resume",action="store_true");args=parser.parse_args();targets=tuple(args.target or sorted(SEEDS))
 OUTPUT.mkdir(parents=True,exist_ok=True);(OUTPUT/"plan.json").write_text(json.dumps({"status":"registered_before_evaluation","validation_only":True,"test_metrics_produced":False,"targets":{key:list(SEEDS[key]) for key in targets},"conditions":CONDITIONS,"candidate":"channel-patch ACF SSL","external_baseline":"same-seed official PatchTST","few_shot_fraction":.25},indent=2)+"\n",encoding="utf-8")
 for target in targets:
  seeds=SEEDS[target]
  for seed in seeds:
   checkpoint=CANDIDATE/target/f"s{seed}_channel_patch_frozen_core.pt";prefix=OUTPUT/f"{target}_s{seed}_channel_patch"
   if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
   if not(args.resume and prefix.with_suffix(".json").is_file() and complete(prefix.with_name(prefix.name+"_event.json"))):guard([sys.executable,str(ROOT/"scripts/evaluate_mambatsv3_lowdata_robustness.py"),"--dataset",DATASETS[target],"--checkpoint",str(checkpoint),"--output",str(prefix.with_suffix(".json")),"--batch-size","4","--seed",str(seed),"--hybrid"],prefix)
   checkpoint=PATCH/target/f"{target}_h192_s{seed}_official_patchtst.pt";prefix=OUTPUT/f"{target}_s{seed}_official_patchtst"
   if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
   if not(args.resume and prefix.with_suffix(".json").is_file() and complete(prefix.with_name(prefix.name+"_event.json"))):guard([sys.executable,str(ROOT/"scripts/evaluate_official_patchtst_robustness.py"),"--dataset",DATASETS[target],"--checkpoint",str(checkpoint),"--output",str(prefix.with_suffix(".json")),"--batch-size","4","--seed",str(seed)],prefix)
 return 0
if __name__=="__main__":raise SystemExit(main())
