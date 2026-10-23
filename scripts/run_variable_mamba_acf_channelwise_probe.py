#!/usr/bin/env python3
"""Validation-only channelwise ACF supervision gate for variable-token Mamba."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"artifacts/exploratory/variable_mamba_acf_channelwise_probe_20261008"
SEED=6108
ARMS=("global_relation","channelwise_relation")

def complete(path):
    return path.is_file() and json.loads(path.read_text()).get("status")=="completed" and json.loads(path.read_text()).get("trigger") is None

def guard(command,prefix,stage):
    subprocess.run([sys.executable,str(ROOT/"scripts/run_gpu_guarded.py"),"--temperature-limit-c","78","--fan-limit-percent","85","--interval-sec","5","--telemetry-csv",str(prefix.with_name(prefix.name+f"_{stage}_telemetry.csv")),"--event-log",str(prefix.with_name(prefix.name+f"_{stage}_event.json")),"--",*command],cwd=ROOT,check=True)

def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    plan={"status":"registered_before_training","validation_only":True,"test_metrics_produced":False,"target":"Traffic","horizon":96,"few_shot_fraction":.25,"seed":SEED,"arms":list(ARMS),"question":"Does preserving per-variable train-prefix ACF peaks improve variable-token Mamba representation transfer versus an averaged global ACF?","fixed_contract":{"backbone":"one variable-axis Mamba scan","d_model":32,"d_ff":64,"target_variable_mamba_frozen":True,"representation_relation_fraction":.05},"target_calibration_boundary":"only earliest 60% x 25% Traffic rows; validation and test excluded","only_difference":"global_relation supervises time representations using the channel-averaged ACF; channelwise_relation supervises the existing per-variable Mamba token states with each variable's own selected train-prefix peaks","screen_rule":"expand channelwise only if validation MSE is at least .25 percent below global_relation"}
    (OUTPUT/"plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    for arm in ARMS:
        prefix=OUTPUT/f"s{SEED}_{arm}"
        pre=[sys.executable,"scripts/pretrain_variable_mamba_acf_core.py","--source","Electricity:dataset/electricity/electricity.csv:csv","--calibration-source","Traffic:dataset/traffic/traffic.csv:csv","--calibration-fraction",".25","--output",str(prefix.with_name(prefix.name+"_pretrain.pt")),"--seq-len","96","--pred-len","96","--d-model","32","--d-ff","64","--mamba-d-state","8","--mamba-d-conv","2","--train-fraction",".6","--batch-size","4","--epochs","3","--max-windows","192","--relation-loss-fraction",".1","--calibration-relation-loss-fraction",".3","--representation-relation-fraction",".05","--calibration-representation-relation-fraction",".05","--seed",str(SEED)]
        if arm=="channelwise_relation": pre.append("--representation-channelwise-acf")
        ft=[sys.executable,"scripts/finetune_variable_mamba_acf_head.py","--dataset","dataset/traffic/traffic.csv","--source-checkpoint",str(prefix.with_name(prefix.name+"_pretrain.pt")),"--output",str(prefix.with_name(prefix.name+"_head.pt")),"--seq-len","96","--pred-len","96","--d-model","32","--d-ff","64","--mamba-d-state","8","--mamba-d-conv","2","--few-shot-fraction",".25","--train-fraction",".6","--val-fraction",".2","--batch-size","4","--epochs","5","--max-train-windows","64","--seed",str(SEED)]
        if "--evaluate-test" in pre+ft: raise ValueError("channelwise probe cannot read test")
        if not(prefix.with_name(prefix.name+"_pretrain.pt").is_file() and complete(prefix.with_name(prefix.name+"_pretrain_event.json"))): guard(pre,prefix,"pretrain")
        if not(prefix.with_name(prefix.name+"_head.pt").is_file() and complete(prefix.with_name(prefix.name+"_finetune_event.json"))): guard(ft,prefix,"finetune")

if __name__=="__main__":
    main()
