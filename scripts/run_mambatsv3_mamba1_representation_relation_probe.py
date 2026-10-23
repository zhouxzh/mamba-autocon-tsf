#!/usr/bin/env python3
"""Validation-only direct-representation ACF pretraining probe."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"artifacts/exploratory/mambatsv3_mamba1_representation_relation_probe_20261006"
SEED=5990
TARGETS={"pems_bay":(ROOT/"artifacts/exploratory/mambatsv3_pems_mamba1_10pct_replication_20261006",5960,"PEMS-BAY"),"metr_la":(ROOT/"artifacts/exploratory/mambatsv3_metr_mamba1_10pct_replication_20261006",5970,"METR-LA")}
ARMS=("null","plain","representation")
def load(p):return json.loads(p.read_text(encoding="utf-8"))
def replace(c,f,v):out=list(c);out[out.index(f)+1]=v;return out
def append(c,*items):return list(c)+list(items)
def commands(target,arm):
    root,seed,_=TARGETS[target]; ref=root/f"s{seed}_{'null' if arm=='null' else 'active'}"; prefix=OUTPUT/target/f"s{SEED}_{arm}"
    pre=load(ref.with_name(ref.name+"_pretrain_event.json"))["command"];pre[0]=sys.executable
    ft=load(ref.with_name(ref.name+"_finetune_event.json"))["command"];ft[0]=sys.executable
    if arm=="representation":pre=append(pre,"--representation-relation-fraction",".05")
    pre=replace(pre,"--output",str(prefix.with_name(prefix.name+"_pretrain.pt")));pre=replace(pre,"--seed",str(SEED))
    ft=replace(ft,"--source-checkpoint",str(prefix.with_name(prefix.name+"_pretrain.pt")));ft=replace(ft,"--output",str(prefix.with_name(prefix.name+"_frozen_core.pt")));ft=replace(ft,"--seed",str(SEED))
    if "--evaluate-test" in pre+ft:raise ValueError("representation probe cannot read test rows")
    return pre,ft,prefix
def guard(c,p,stage):subprocess.run([sys.executable,str(ROOT/"scripts/run_gpu_guarded.py"),"--temperature-limit-c","78","--fan-limit-percent","85","--interval-sec","5","--telemetry-csv",str(p.with_name(p.name+f"_{stage}_telemetry.csv")),"--event-log",str(p.with_name(p.name+f"_{stage}_event.json")),"--",*c],cwd=ROOT,check=True)
def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/"plan.json").write_text(json.dumps({"status":"registered_before_training","validation_only":True,"test_metrics_produced":False,"seed":SEED,"targets":{k:v[2] for k,v in TARGETS.items()},"arms":list(ARMS),"question":"Does a prediction-scaled ACF objective directly on repr_ improve frozen low-label transfer?","fixed_contract":{"mixer":"mamba1","patch_len":24,"patch_stride":8,"few_shot_fraction":.1,"forecast_relation_fraction":.1,"representation_relation_fraction":.05,"target_core_frozen":True,"mamba_scan_count":1,"inference_change":"none"}},indent=2)+"\n",encoding="utf-8")
    for target in TARGETS:
        (OUTPUT/target).mkdir(parents=True,exist_ok=True)
        for arm in ARMS:
            pre,ft,prefix=commands(target,arm);guard(pre,prefix,"pretrain");guard(ft,prefix,"finetune")
    return 0
if __name__=="__main__":raise SystemExit(main())
