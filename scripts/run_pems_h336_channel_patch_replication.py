#!/usr/bin/env python3
"""Current 24/8 PEMS-BAY H336 channel-patch SSL validation replication."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CANDIDATE_REFERENCE=ROOT/"artifacts/exploratory/mambats_channel_patch_replication_20261006/pems_bay"
PATCH_REFERENCE=ROOT/"artifacts/exploratory/official_patchtst_channel_patch_replication_20261006/pems_bay"
OUTPUT=ROOT/"artifacts/exploratory/pems_h336_channel_patch_replication_20261006";PATCH_OUTPUT=ROOT/"artifacts/exploratory/official_patchtst_pems_h336_channel_patch_replication_20261006";SEEDS=(6052,6053,6054)
def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def replace(command,flag,value):out=list(command);out[out.index(flag)+1]=value;return out
def complete(path):return Path(path).is_file() and load(path).get("status")=="completed" and load(path).get("trigger") is None
def guard(command,prefix,stage):subprocess.run([sys.executable,str(ROOT/"scripts/run_gpu_guarded.py"),"--temperature-limit-c","78","--fan-limit-percent","85","--interval-sec","5","--telemetry-csv",str(prefix.with_name(prefix.name+f"_{stage}_telemetry.csv")),"--event-log",str(prefix.with_name(prefix.name+f"_{stage}_event.json")),"--",*command],cwd=ROOT,check=True)
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--resume",action="store_true");args=parser.parse_args();OUTPUT.mkdir(parents=True,exist_ok=True);PATCH_OUTPUT.mkdir(parents=True,exist_ok=True)
 plan={"status":"registered_before_training","validation_only":True,"test_metrics_produced":False,"target":"PEMS-BAY","horizon":336,"few_shot_fraction":.1,"seeds":list(SEEDS),"question":"Does channel-patch ACF SSL generalize from H192 to H336 under the current 24/8 transfer contract?","source_pool":["PEMS07","PEMS08","PEMS04"],"arms":["null","forecast","channel_patch","official_patchtst"],"fixed_contract":{"mixer":"mamba1","patch":"24/8","d_model":24,"d_ff":48,"target_core_frozen":True,"mamba_scan_count":1,"channel_patch_relation_fraction":.05}}
 (OUTPUT/"plan.json").write_text(json.dumps(plan,indent=2)+"\n",encoding="utf-8");(PATCH_OUTPUT/"plan.json").write_text(json.dumps(plan,indent=2)+"\n",encoding="utf-8")
 for seed in SEEDS:
  for arm in ("null","forecast","channel_patch"):
   source_arm="null" if arm=="null" else "channel_patch" if arm=="channel_patch" else "forecast";ref=CANDIDATE_REFERENCE/f"s6026_{source_arm}";prefix=OUTPUT/f"s{seed}_{arm}";pre=load(ref.with_name(ref.name+"_pretrain_event.json"))["command"];ft=load(ref.with_name(ref.name+"_finetune_event.json"))["command"];pre[0]=sys.executable;ft[0]=sys.executable
   pre=replace(pre,"--output",str(prefix.with_name(prefix.name+"_pretrain.pt")));pre=replace(pre,"--seed",str(seed));pre=replace(pre,"--source-pred-len","336");ft=replace(ft,"--source-checkpoint",str(prefix.with_name(prefix.name+"_pretrain.pt")));ft=replace(ft,"--output",str(prefix.with_name(prefix.name+"_frozen_core.pt")));ft=replace(ft,"--seed",str(seed));ft=replace(ft,"--pred-len","336")
   if "--evaluate-test" in pre+ft:raise ValueError("H336 candidate cannot read test")
   pre_event=prefix.with_name(prefix.name+"_pretrain_event.json");ft_event=prefix.with_name(prefix.name+"_finetune_event.json")
   if not(args.resume and prefix.with_name(prefix.name+"_pretrain.pt").is_file() and complete(pre_event)):guard(pre,prefix,"pretrain")
   if not(args.resume and prefix.with_name(prefix.name+"_frozen_core.pt").is_file() and complete(ft_event)):guard(ft,prefix,"finetune")
  ref=PATCH_REFERENCE/"pems_bay_h192_s6026_official_patchtst_event.json";prefix=PATCH_OUTPUT/f"pems_bay_h336_s{seed}_official_patchtst";patch=load(ref)["command"];patch[0]=sys.executable;patch=replace(patch,"--output",str(prefix.with_suffix(".pt")));patch=replace(patch,"--seed",str(seed));patch=replace(patch,"--pred-len","336")
  if "--evaluate-test" in patch:raise ValueError("H336 PatchTST cannot read test")
  event=prefix.with_name(prefix.name+"_event.json")
  if not(args.resume and prefix.with_suffix(".pt").is_file() and complete(event)):guard(patch,prefix,"patch")
if __name__=="__main__":main()
