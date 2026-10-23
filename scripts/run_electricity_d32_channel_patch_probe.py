#!/usr/bin/env python3
"""Validation-only d32/24 channel-patch representation gate for Electricity."""
from __future__ import annotations
import json,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/"artifacts/exploratory/mambatsv3_electricity_mamba1_d32_stride24_transfer_probe_20261006"
OUTPUT=ROOT/"artifacts/exploratory/electricity_d32_channel_patch_probe_20261006"
SEED=6043
def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def replace(command,flag,value):out=list(command);out[out.index(flag)+1]=value;return out
def guard(command,prefix,stage):subprocess.run([sys.executable,str(ROOT/"scripts/run_gpu_guarded.py"),"--temperature-limit-c","78","--fan-limit-percent","85","--interval-sec","5","--telemetry-csv",str(prefix.with_name(prefix.name+f"_{stage}_telemetry.csv")),"--event-log",str(prefix.with_name(prefix.name+f"_{stage}_event.json")),"--",*command],cwd=ROOT,check=True)
def main():
 OUTPUT.mkdir(parents=True,exist_ok=True)
 (OUTPUT/"plan.json").write_text(json.dumps({"status":"registered_before_training","validation_only":True,"test_metrics_produced":False,"target":"Electricity","horizon":96,"few_shot_fraction":.25,"seed":SEED,"question":"Does channel-patch ACF representation improve the positive d32/24 strong-calibration bridge?","arms":{"forecast":"frozen d32 forecast-InfoNCE protocol","channel_patch":"same protocol plus channel patch relation=.05"},"fixed_contract":{"mixer":"mamba1","patch":"24/24","d_model":32,"d_ff":64,"target_core_frozen":True,"mamba_scan_count":1,"Traffic_relation":.1,"Electricity_calibration_relation":.3}},indent=2)+"\n",encoding="utf-8")
 for arm in ("forecast","channel_patch"):
  ref=REFERENCE/"s5996_active";prefix=OUTPUT/f"s{SEED}_{arm}";pre=load(ref.with_name(ref.name+"_pretrain_event.json"))["command"];ft=load(ref.with_name(ref.name+"_finetune_event.json"))["command"];pre[0]=sys.executable;ft[0]=sys.executable
  if arm=="channel_patch":pre += ["--channel-patch-relation-fraction",".05"]
  pre=replace(pre,"--output",str(prefix.with_name(prefix.name+"_pretrain.pt")));pre=replace(pre,"--seed",str(SEED));ft=replace(ft,"--source-checkpoint",str(prefix.with_name(prefix.name+"_pretrain.pt")));ft=replace(ft,"--output",str(prefix.with_name(prefix.name+"_frozen_core.pt")));ft=replace(ft,"--seed",str(SEED))
  if "--evaluate-test" in pre+ft:raise ValueError("d32 channel-patch probe cannot read test rows")
  guard(pre,prefix,"pretrain");guard(ft,prefix,"finetune")
if __name__=="__main__":main()
