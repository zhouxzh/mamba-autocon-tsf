#!/usr/bin/env python3
"""Current-geometry PEMS03 external validation of channel-patch ACF SSL."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/"artifacts/exploratory/mambats_channel_patch_replication_20261006/pems_bay"
PATCH_REF=ROOT/"artifacts/exploratory/official_patchtst_channel_patch_replication_20261006/pems_bay"
# The first attempt used the current PEMS-BAY command wholesale, including a
# historical PEMS03 checkpoint.  Keep that artifact immutable, and put the
# corrected target-excluded replication in a distinct directory.
INITIAL_REFERENCE=ROOT/"artifacts/exploratory/mambatsv3_acf_infonce_local_cluster_pems03_external_confirmation_20261003/s5570_active_pretrain_event.json"
OUTPUT=ROOT/"artifacts/exploratory/pems03_channel_patch_external_replication_rerun_20261007";PATCH_OUTPUT=ROOT/"artifacts/exploratory/official_patchtst_pems03_channel_patch_external_replication_rerun_20261007"
SEEDS=(6055,6056,6057)
ARMS=("null","forecast","channel_patch")
def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def replace(command,flag,value):out=list(command);out[out.index(flag)+1]=value;return out
def values(command,flag):return [command[index+1] for index,item in enumerate(command[:-1]) if item==flag]
def replace_repeated(command,flag,replacements):
 out=list(command); positions=[index for index,item in enumerate(out[:-1]) if item==flag]
 if len(positions)!=len(replacements):raise ValueError(f"{flag} count mismatch")
 for position,value in zip(positions,replacements):out[position+1]=value
 return out
def set_single(command,flag,value):
 matches=values(command,flag)
 if len(matches)>1:raise ValueError(f"{flag} must occur at most once")
 return replace(command,flag,value) if matches else [*command,flag,value]
def replace_sources(command,sources):
 out=[];i=0;seen=0
 while i<len(command):
  if command[i]=="--source":out += ["--source",sources[seen]];seen+=1;i+=2
  else:out.append(command[i]);i+=1
 if seen!=len(sources):raise ValueError("source count mismatch")
 return out
def complete(path):return Path(path).is_file() and load(path).get("status")=="completed" and load(path).get("trigger") is None
def guard(command,prefix,stage):subprocess.run([sys.executable,str(ROOT/"scripts/run_gpu_guarded.py"),"--temperature-limit-c","78","--fan-limit-percent","85","--interval-sec","5","--telemetry-csv",str(prefix.with_name(prefix.name+f"_{stage}_telemetry.csv")),"--event-log",str(prefix.with_name(prefix.name+f"_{stage}_event.json")),"--",*command],cwd=ROOT,check=True)
def assert_target_excluded(command):
 checked=values(command,"--source")+values(command,"--initial-checkpoint")
 if not checked or any("pems03" in item.lower() for item in checked):
  raise ValueError("PEMS03 is forbidden in external source or initialization inputs")
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--resume",action="store_true");args=parser.parse_args();OUTPUT.mkdir(parents=True,exist_ok=True);PATCH_OUTPUT.mkdir(parents=True,exist_ok=True)
 sources=["PEMS04:dataset/PEMS04/PEMS04.npz:npz","PEMS08:dataset/PEMS08/PEMS08.npz:npz","PEMS07:dataset/PEMS07/PEMS07.npz:npz"]
 initial_checkpoints=values(load(INITIAL_REFERENCE)["command"],"--initial-checkpoint")
 if len(initial_checkpoints)!=2 or any("pems03" in item.lower() for item in initial_checkpoints):raise ValueError("historical target-excluded initialization is unavailable")
 plan={"status":"registered_before_training","validation_only":True,"test_metrics_produced":False,"target":"PEMS03","horizon":192,"few_shot_fraction":.1,"seeds":list(SEEDS),"question":"Does current 24/8 channel-patch SSL transfer to PEMS03 excluded from source and initialization?","source_pool":["PEMS04","PEMS08","PEMS07"],"initialization_sources":["PEMS04","PEMS08"],"initialization_checkpoints":initial_checkpoints,"target_excluded_from_source_and_initialization":True,"supersedes_unusable_attempt":"artifacts/exploratory/pems03_channel_patch_external_replication_20261007","arms":["null","forecast","channel_patch","official_patchtst"],"fixed_contract":{"mixer":"mamba1","patch":"24/8","d_model":24,"d_ff":48,"target_core_frozen":True,"mamba_scan_count":1,"channel_patch_relation_fraction":.05}}
 (OUTPUT/"plan.json").write_text(json.dumps(plan,indent=2)+"\n",encoding="utf-8");(PATCH_OUTPUT/"plan.json").write_text(json.dumps(plan,indent=2)+"\n",encoding="utf-8")
 for seed in SEEDS:
  for arm in ARMS:
   source_arm="null" if arm=="null" else "channel_patch" if arm=="channel_patch" else "forecast";ref=REFERENCE/f"s6026_{source_arm}";prefix=OUTPUT/f"s{seed}_{arm}";pre=load(ref.with_name(ref.name+"_pretrain_event.json"))["command"];ft=load(ref.with_name(ref.name+"_finetune_event.json"))["command"];pre[0]=sys.executable;ft[0]=sys.executable;pre=replace_sources(pre,sources);pre=replace_repeated(pre,"--initial-checkpoint",initial_checkpoints);pre=replace(pre,"--output",str(prefix.with_name(prefix.name+"_pretrain.pt")));pre=replace(pre,"--seed",str(seed));ft=replace(ft,"--source-checkpoint",str(prefix.with_name(prefix.name+"_pretrain.pt")));ft=replace(ft,"--output",str(prefix.with_name(prefix.name+"_frozen_core.pt")));ft=replace(ft,"--seed",str(seed));ft=replace(ft,"--dataset","dataset/PEMS03/PEMS03.npz")
   if arm=="channel_patch":pre=set_single(pre,"--channel-patch-relation-fraction",".05")
   assert_target_excluded(pre)
   if "--evaluate-test" in pre+ft:raise ValueError("PEMS03 external replication cannot read test")
   pre_event=prefix.with_name(prefix.name+"_pretrain_event.json");ft_event=prefix.with_name(prefix.name+"_finetune_event.json")
   if not(args.resume and prefix.with_name(prefix.name+"_pretrain.pt").is_file() and complete(pre_event)):guard(pre,prefix,"pretrain")
   if not(args.resume and prefix.with_name(prefix.name+"_frozen_core.pt").is_file() and complete(ft_event)):guard(ft,prefix,"finetune")
  ref=PATCH_REF/f"pems_bay_h192_s6026_official_patchtst_event.json";prefix=PATCH_OUTPUT/f"pems03_h192_s{seed}_official_patchtst";patch=load(ref)["command"];patch[0]=sys.executable;patch=replace(patch,"--output",str(prefix.with_suffix(".pt")));patch=replace(patch,"--seed",str(seed));patch=replace(patch,"--dataset","dataset/PEMS03/PEMS03.npz")
  if "--evaluate-test" in patch:raise ValueError("PatchTST cannot read test")
  event=prefix.with_name(prefix.name+"_event.json")
  if not(args.resume and prefix.with_suffix(".pt").is_file() and complete(event)):guard(patch,prefix,"patch")
if __name__=="__main__":main()
