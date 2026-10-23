#!/usr/bin/env python3
"""Fresh 25%-label validation replication of channel-patch ACF SSL."""

from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"artifacts/exploratory/channel_patch_25pct_replication_20261006"
TARGETS={
 "pems_bay": (ROOT/"artifacts/exploratory/mambatsv3_pems_mamba1_10pct_independent_replication_20261006",6001,(6036,6037,6038)),
 "metr_la": (ROOT/"artifacts/exploratory/mambatsv3_metr_mamba1_10pct_independent_replication_20261006",6007,(6039,6040,6041)),
}
ARMS=("null","forecast","channel_patch")
def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def replace(command,flag,value):
 out=list(command);out[out.index(flag)+1]=value;return out
def complete(path): return Path(path).is_file() and load(path).get("status")=="completed" and load(path).get("trigger") is None
def guard(command,prefix,stage):
 subprocess.run([sys.executable,str(ROOT/"scripts/run_gpu_guarded.py"),"--temperature-limit-c","78","--fan-limit-percent","85","--interval-sec","5","--telemetry-csv",str(prefix.with_name(prefix.name+f"_{stage}_telemetry.csv")),"--event-log",str(prefix.with_name(prefix.name+f"_{stage}_event.json")),"--",*command],cwd=ROOT,check=True)
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--target",choices=sorted(TARGETS),action="append",default=None);parser.add_argument("--resume",action="store_true");args=parser.parse_args();targets=tuple(args.target or sorted(TARGETS))
 OUTPUT.mkdir(parents=True,exist_ok=True)
 (OUTPUT/"plan.json").write_text(json.dumps({"status":"registered_before_training","validation_only":True,"test_metrics_produced":False,"targets":{name:{"seeds":list(TARGETS[name][2]),"reference_10pct":str(TARGETS[name][0])} for name in targets},"question":"Does channel-patch ACF SSL preserve gains at 25% labels under current 24/8 transfer?","arms":{"null":"next-patch only","forecast":"forecast-level ACF InfoNCE","channel_patch":"forecast-level plus patch-trajectory ACF InfoNCE"},"fixed_contract":{"mixer":"mamba1","patch":"24/8","d_model":24,"d_ff":48,"few_shot_fraction":.25,"target_core_frozen":True,"mamba_scan_count":1,"channel_patch_relation_fraction":.05}},indent=2)+"\n",encoding="utf-8")
 for target in targets:
  root,reference_seed,seeds=TARGETS[target];(OUTPUT/target).mkdir(parents=True,exist_ok=True)
  for seed in seeds:
   for arm in ARMS:
    source_arm="null" if arm=="null" else "active";reference=root/f"s{reference_seed}_{source_arm}";prefix=OUTPUT/target/f"s{seed}_{arm}"
    pre=load(reference.with_name(reference.name+"_pretrain_event.json"))["command"];ft=load(reference.with_name(reference.name+"_finetune_event.json"))["command"];pre[0]=sys.executable;ft[0]=sys.executable
    if arm=="channel_patch":pre += ["--channel-patch-relation-fraction",".05"]
    pre=replace(pre,"--output",str(prefix.with_name(prefix.name+"_pretrain.pt")));pre=replace(pre,"--seed",str(seed))
    ft=replace(ft,"--source-checkpoint",str(prefix.with_name(prefix.name+"_pretrain.pt")));ft=replace(ft,"--output",str(prefix.with_name(prefix.name+"_frozen_core.pt")));ft=replace(ft,"--seed",str(seed));ft=replace(ft,"--few-shot-fraction",".25")
    if "--evaluate-test" in pre+ft:raise ValueError("25% replication cannot read test rows")
    pre_event=prefix.with_name(prefix.name+"_pretrain_event.json");ft_event=prefix.with_name(prefix.name+"_finetune_event.json")
    if not(args.resume and prefix.with_name(prefix.name+"_pretrain.pt").is_file() and complete(pre_event)):guard(pre,prefix,"pretrain")
    if not(args.resume and prefix.with_name(prefix.name+"_frozen_core.pt").is_file() and complete(ft_event)):guard(ft,prefix,"finetune")
 return 0
if __name__=="__main__":raise SystemExit(main())
