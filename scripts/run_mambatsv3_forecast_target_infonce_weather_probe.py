#!/usr/bin/env python3
"""Validation-only task-sign ACF InfoNCE gate on the weak Weather H192 coordinate."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/"artifacts/exploratory/mambatsv3_acf_infonce_weather_h192_screen_20261003"
OUTPUT=ROOT/"artifacts/exploratory/mambatsv3_forecast_target_infonce_weather_probe_20261008"
SEED=6121
ARMS=("raw_peak_sign","forecast_target_sign")

def load(path): return json.loads(path.read_text())

def replace(command,flag,value):
    result=list(command)
    result[result.index(flag)+1]=value
    return result

def add(command,*items): return list(command) if items[0] in command else [*command,*items]

def complete(path):
    return path.is_file() and load(path).get("status")=="completed" and load(path).get("trigger") is None

def guard(command,prefix,stage):
    subprocess.run([sys.executable,str(ROOT/"scripts/run_gpu_guarded.py"),"--temperature-limit-c","78","--fan-limit-percent","85","--interval-sec","5","--telemetry-csv",str(prefix.with_name(prefix.name+f"_{stage}_telemetry.csv")),"--event-log",str(prefix.with_name(prefix.name+f"_{stage}_event.json")),"--",*command],cwd=ROOT,check=True)

def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    plan={"status":"registered_before_training","validation_only":True,"test_metrics_produced":False,"target":"Weather","horizon":192,"few_shot_fraction":.1,"seed":SEED,"reference_protocol":str(REFERENCE),"arms":list(ARMS),"question":"Does using train-only future-residual ACF signs at fixed raw ACF peaks improve signed forecast InfoNCE on the weak Weather H192 mechanism coordinate?","fixed_contract":{"backbone":"channel-agnostic MambaTSv3 one scan","prediction_target":"last_value_residual","relation_view":"forecast_contrastive","source_relation_fraction":.1,"source_domains":["METR-LA","PEMS-BAY","Electricity"],"target_primary":"NLinear","exact_prediction_head_transfer":True},"only_difference":"forecast_target_sign replaces the raw-signal ACF value used for signed InfoNCE logits with the train-prefix H192 future-residual trajectory correlation at the identical selected lag; lag selection, pairs, confidence policy, sources, and all optimizer settings remain fixed","screen_rule":"expand only if task-sign validation MSE is at least .25 percent below raw-peak-sign"}
    (OUTPUT/"plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    reference=REFERENCE/"weather_h192_s5450_active"
    pre_base=load(reference.with_name(reference.name+"_pretrain_event.json"))["command"]
    ft_base=load(reference.with_name(reference.name+"_hybrid_event.json"))["command"]
    for arm in ARMS:
        prefix=OUTPUT/f"s{SEED}_{arm}"
        pre=list(pre_base); ft=list(ft_base)
        pre[6]=sys.executable; ft[6]=sys.executable
        pre=replace(pre,"--output",str(prefix.with_name(prefix.name+"_pretrain.pt")))
        pre=replace(pre,"--seed",str(SEED))
        ft=replace(ft,"--source-checkpoint",str(prefix.with_name(prefix.name+"_pretrain.pt")))
        ft=replace(ft,"--output",str(prefix.with_name(prefix.name+"_hybrid.pt")))
        ft=replace(ft,"--seed",str(SEED))
        if arm=="forecast_target_sign": pre=add(pre,"--forecast-target-acf")
        if "--evaluate-test" in pre+ft: raise ValueError("task-sign probe cannot read test")
        if not(prefix.with_name(prefix.name+"_pretrain.pt").is_file() and complete(prefix.with_name(prefix.name+"_pretrain_event.json"))): guard(pre,prefix,"pretrain")
        if not(prefix.with_name(prefix.name+"_hybrid.pt").is_file() and complete(prefix.with_name(prefix.name+"_hybrid_event.json"))): guard(ft,prefix,"hybrid")

if __name__=="__main__": main()
