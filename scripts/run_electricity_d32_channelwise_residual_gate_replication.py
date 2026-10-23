#!/usr/bin/env python3
"""Independent d32/24 Electricity channelwise residual-gate replication."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PRETRAIN_REFERENCE=ROOT/"artifacts/exploratory/electricity_d32_channel_patch_probe_20261006";FINETUNE_REFERENCE=ROOT/"artifacts/exploratory/electricity_d32_channelwise_residual_gate_probe_20261006";OUTPUT=ROOT/"artifacts/exploratory/electricity_d32_channelwise_residual_gate_replication_rerun_20261006";SEEDS=(6047,6048,6049)
def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def replace(command,flag,value):out=list(command);out[out.index(flag)+1]=value;return out
def complete(path):return Path(path).is_file() and load(path).get("status")=="completed" and load(path).get("trigger") is None
def guard(command,prefix):subprocess.run([sys.executable,str(ROOT/"scripts/run_gpu_guarded.py"),"--temperature-limit-c","78","--fan-limit-percent","85","--interval-sec","5","--telemetry-csv",str(prefix.with_name(prefix.name+"_telemetry.csv")),"--event-log",str(prefix.with_name(prefix.name+"_event.json")),"--",*command],cwd=ROOT,check=True)
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--resume",action="store_true");args=parser.parse_args();OUTPUT.mkdir(parents=True,exist_ok=True)
 (OUTPUT/"plan.json").write_text(json.dumps({"status":"registered_before_training","validation_only":True,"test_metrics_produced":False,"target":"Electricity","horizon":96,"few_shot_fraction":.25,"seeds":list(SEEDS),"question":"Does the target-channel residual gate replicate on the d32/24 channel-patch protocol?","pretrain_reference":str(PRETRAIN_REFERENCE),"finetune_reference":str(FINETUNE_REFERENCE),"architecture":{"mixer":"mamba1","patch":"24/24","d_model":32,"d_ff":64,"mamba_scan_count":1},"target_gate":"zero-initialized channelwise Mamba correction scale","paired_design":"each fresh seed shares one channel-patch pretraining checkpoint between global and channelwise target gates","supersedes":"electricity_d32_channelwise_residual_gate_replication_20261006; that run failed before training because its seed-specific pretraining checkpoints did not exist"},indent=2)+"\n",encoding="utf-8")
 for seed in SEEDS:
  pre_prefix=OUTPUT/f"s{seed}_channel_patch";pre=load(PRETRAIN_REFERENCE/"s6043_channel_patch_pretrain_event.json")["command"];pre[0]=sys.executable;pre=replace(pre,"--output",str(pre_prefix.with_name(pre_prefix.name+"_pretrain.pt")));pre=replace(pre,"--seed",str(seed))
  if "--evaluate-test" in pre:raise ValueError("replication cannot read test rows")
  pre_event=pre_prefix.with_name(pre_prefix.name+"_pretrain_event.json")
  if not(args.resume and pre_prefix.with_name(pre_prefix.name+"_pretrain.pt").is_file() and complete(pre_event)):guard(pre,pre_prefix.with_name(pre_prefix.name+"_pretrain"))
  for arm,reference_name in (("global_gate","s6043_channel_patch_finetune_event.json"),("channelwise_gate","s6044_channelwise_gate_event.json")):
   prefix=OUTPUT/f"s{seed}_{arm}";command=load(FINETUNE_REFERENCE/reference_name)["command"] if reference_name.startswith("s6044_channelwise") else load(PRETRAIN_REFERENCE/reference_name)["command"];command[0]=sys.executable;command=replace(command,"--source-checkpoint",str(pre_prefix.with_name(pre_prefix.name+"_pretrain.pt")));command=replace(command,"--output",str(prefix.with_suffix(".pt")));command=replace(command,"--seed",str(seed))
   if "--evaluate-test" in command:raise ValueError("replication cannot read test rows")
   event=prefix.with_name(prefix.name+"_event.json")
   if not(args.resume and prefix.with_suffix(".pt").is_file() and complete(event)):guard(command,prefix)
if __name__=="__main__":main()
