#!/usr/bin/env python3
"""Validation-only Electricity prototype-set-pooling probe."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/"artifacts/exploratory/mambatsv3_electricity_mamba1_strong_calibration_relation_probe_20261006"
OUTPUT=ROOT/"artifacts/exploratory/mambatsv3_electricity_mamba1_prototype_pooling_probe_20261006"
SEED=5989
def load(p): return json.loads(p.read_text(encoding="utf-8"))
def replace(c,f,v): out=list(c); out[out.index(f)+1]=v; return out
def append(c,*items): return list(c)+list(items)
def commands(arm):
    source_arm="null" if arm=="prototype_null" else "active"; ref=REFERENCE/f"s5948_{source_arm}"; prefix=OUTPUT/f"s{SEED}_{arm}"
    pre=load(ref.with_name(ref.name+"_pretrain_event.json"))["command"]; pre[0]=sys.executable
    ft=load(ref.with_name(ref.name+"_finetune_event.json"))["command"]; ft[0]=sys.executable
    if arm.startswith("prototype_"):
        pre=append(pre,"--mamba-ts-channel-prototype-pooling","--mamba-ts-channel-prototype-count","4")
        ft=append(ft,"--mamba-ts-channel-prototype-pooling","--mamba-ts-channel-prototype-count","4")
    pre=replace(pre,"--output",str(prefix.with_name(prefix.name+"_pretrain.pt"))); pre=replace(pre,"--seed",str(SEED))
    ft=replace(ft,"--source-checkpoint",str(prefix.with_name(prefix.name+"_pretrain.pt"))); ft=replace(ft,"--output",str(prefix.with_name(prefix.name+"_frozen_core.pt"))); ft=replace(ft,"--seed",str(SEED))
    if "--evaluate-test" in pre+ft: raise ValueError("probe cannot read test rows")
    return pre,ft,prefix
def guard(c,p,stage): subprocess.run([sys.executable,str(ROOT/"scripts/run_gpu_guarded.py"),"--temperature-limit-c","78","--fan-limit-percent","85","--interval-sec","5","--telemetry-csv",str(p.with_name(p.name+f"_{stage}_telemetry.csv")),"--event-log",str(p.with_name(p.name+f"_{stage}_event.json")),"--",*c],cwd=ROOT,check=True)
def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/"plan.json").write_text(json.dumps({"status":"registered_before_training","validation_only":True,"test_metrics_produced":False,"target":"Electricity","horizon":96,"few_shot_fraction":.25,"seed":SEED,"arms":["plain_active","prototype_active","prototype_null"],"base_protocol":str(REFERENCE),"fixed_contract":{"mixer":"mamba1","patch_len":24,"patch_stride":8,"target_core_frozen":True,"mamba_scan_count":1},"prototype_count":4,"question":"Can multiple channel-set prototypes improve strong-calibration high-dimensional transfer?"},indent=2)+"\n",encoding="utf-8")
    for arm in ("plain_active","prototype_active","prototype_null"):
        pre,ft,prefix=commands(arm); guard(pre,prefix,"pretrain"); guard(ft,prefix,"finetune")
    return 0
if __name__=="__main__": raise SystemExit(main())
