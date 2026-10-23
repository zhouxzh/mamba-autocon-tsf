#!/usr/bin/env python3
"""Validation-only strength probe for variable-token Mamba representation ACF."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/"artifacts/exploratory/variable_mamba_acf_traffic_probe_20261007"
OUTPUT=ROOT/"artifacts/exploratory/variable_mamba_acf_strength_probe_20261007"
SEED=6103
ARMS={"repr_0p05":(".05",".05"),"repr_0p10":(".10",".10")}

def load(path): return json.loads(path.read_text())
def replace(command,flag,value):
    result=list(command);result[result.index(flag)+1]=value;return result
def complete(path): return path.is_file() and load(path).get("status")=="completed" and load(path).get("trigger") is None
def guard(command,prefix,stage):
    subprocess.run([sys.executable,str(ROOT/"scripts/run_gpu_guarded.py"),"--temperature-limit-c","78","--fan-limit-percent","85","--interval-sec","5","--telemetry-csv",str(prefix.with_name(prefix.name+f"_{stage}_telemetry.csv")),"--event-log",str(prefix.with_name(prefix.name+f"_{stage}_event.json")),"--",*command],cwd=ROOT,check=True)

def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    plan={"status":"registered_before_training","validation_only":True,"test_metrics_produced":False,"target":"Traffic","horizon":96,"few_shot_fraction":.25,"seed":SEED,"reference":str(REFERENCE),"arms":ARMS,"question":"Does increasing prediction-scaled representation ACF InfoNCE from .05 to .10 materially improve variable-token Mamba transfer?","fixed_contract":{"backbone":"one variable-axis Mamba scan","d_model":32,"d_ff":64,"target_variable_mamba_frozen":True,"forecast_relation_fractions":{"source":.1,"calibration":.3}},"screen_rule":"expand .10 only if validation MSE is at least .25 percent below .05"}
    (OUTPUT/"plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    reference=REFERENCE/"s6102_representation_acf"
    for arm,(source_fraction,calibration_fraction) in ARMS.items():
        prefix=OUTPUT/f"s{SEED}_{arm}"
        pre=load(reference.with_name(reference.name+"_pretrain_event.json"))["command"]
        ft=load(reference.with_name(reference.name+"_finetune_event.json"))["command"]
        pre[0]=ft[0]=sys.executable
        pre=replace(pre,"--representation-relation-fraction",source_fraction)
        pre=replace(pre,"--calibration-representation-relation-fraction",calibration_fraction)
        pre=replace(pre,"--output",str(prefix.with_name(prefix.name+"_pretrain.pt")))
        pre=replace(pre,"--seed",str(SEED))
        ft=replace(ft,"--source-checkpoint",str(prefix.with_name(prefix.name+"_pretrain.pt")))
        ft=replace(ft,"--output",str(prefix.with_name(prefix.name+"_head.pt")))
        ft=replace(ft,"--seed",str(SEED))
        if "--evaluate-test" in pre+ft: raise ValueError("strength probe cannot read test")
        if not(prefix.with_name(prefix.name+"_pretrain.pt").is_file() and complete(prefix.with_name(prefix.name+"_pretrain_event.json"))):guard(pre,prefix,"pretrain")
        if not(prefix.with_name(prefix.name+"_head.pt").is_file() and complete(prefix.with_name(prefix.name+"_finetune_event.json"))):guard(ft,prefix,"finetune")

if __name__=="__main__": main()
