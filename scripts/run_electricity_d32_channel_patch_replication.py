#!/usr/bin/env python3
"""Independent d32/24 channel-patch Electricity validation replication."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REFERENCE=ROOT/"artifacts/exploratory/electricity_d32_channel_patch_probe_20261006";OUTPUT=ROOT/"artifacts/exploratory/electricity_d32_channel_patch_replication_20261006";SEEDS=(6044,6045,6046)
def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def replace(command,flag,value):out=list(command);out[out.index(flag)+1]=value;return out
def complete(path):return Path(path).is_file() and load(path).get("status")=="completed" and load(path).get("trigger") is None
def guard(command,prefix,stage):subprocess.run([sys.executable,str(ROOT/"scripts/run_gpu_guarded.py"),"--temperature-limit-c","78","--fan-limit-percent","85","--interval-sec","5","--telemetry-csv",str(prefix.with_name(prefix.name+f"_{stage}_telemetry.csv")),"--event-log",str(prefix.with_name(prefix.name+f"_{stage}_event.json")),"--",*command],cwd=ROOT,check=True)
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--resume",action="store_true");args=parser.parse_args();OUTPUT.mkdir(parents=True,exist_ok=True)
 (OUTPUT/"plan.json").write_text(json.dumps({"status":"registered_before_training","validation_only":True,"test_metrics_produced":False,"target":"Electricity","horizon":96,"few_shot_fraction":.25,"seeds":list(SEEDS),"question":"Does the d32/24 channel-patch direction replicate beyond its weak one-seed gate?","frozen_gate":str(REFERENCE),"arms":["forecast","channel_patch"],"fixed_contract":{"mixer":"mamba1","patch":"24/24","d_model":32,"d_ff":64,"target_core_frozen":True,"mamba_scan_count":1,"channel_patch_relation_fraction":.05},"no_result_combination":"fresh seeds separate from 6043 gate"},indent=2)+"\n",encoding="utf-8")
 for seed in SEEDS:
  for arm in ("forecast","channel_patch"):
   ref=REFERENCE/f"s6043_{arm}";prefix=OUTPUT/f"s{seed}_{arm}";pre=load(ref.with_name(ref.name+"_pretrain_event.json"))["command"];ft=load(ref.with_name(ref.name+"_finetune_event.json"))["command"];pre[0]=sys.executable;ft[0]=sys.executable
   pre=replace(pre,"--output",str(prefix.with_name(prefix.name+"_pretrain.pt")));pre=replace(pre,"--seed",str(seed));ft=replace(ft,"--source-checkpoint",str(prefix.with_name(prefix.name+"_pretrain.pt")));ft=replace(ft,"--output",str(prefix.with_name(prefix.name+"_frozen_core.pt")));ft=replace(ft,"--seed",str(seed))
   if "--evaluate-test" in pre+ft:raise ValueError("d32 replication cannot read test rows")
   pre_event=prefix.with_name(prefix.name+"_pretrain_event.json");ft_event=prefix.with_name(prefix.name+"_finetune_event.json")
   if not(args.resume and prefix.with_name(prefix.name+"_pretrain.pt").is_file() and complete(pre_event)):guard(pre,prefix,"pretrain")
   if not(args.resume and prefix.with_name(prefix.name+"_frozen_core.pt").is_file() and complete(ft_event)):guard(ft,prefix,"finetune")
if __name__=="__main__":main()
