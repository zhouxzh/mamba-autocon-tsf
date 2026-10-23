#!/usr/bin/env python3
"""Fresh-seed 2x2 source-diversity x ACF-objective validation gate."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"artifacts/exploratory/variable_mamba_acf_source_interaction_20261008"
SEEDS=(6113,6114,6115)
ARMS={
    "replay_forecast":(["ElectricityA:dataset/electricity/electricity.csv:csv","ElectricityB:dataset/electricity/electricity.csv:csv"],"0"),
    "replay_acf":(["ElectricityA:dataset/electricity/electricity.csv:csv","ElectricityB:dataset/electricity/electricity.csv:csv"],".05"),
    "diverse_forecast":(["Electricity:dataset/electricity/electricity.csv:csv","Weather:dataset/weather/weather.csv:csv"],"0"),
    "diverse_acf":(["Electricity:dataset/electricity/electricity.csv:csv","Weather:dataset/weather/weather.csv:csv"],".05"),
}

def complete(path):
    return path.is_file() and json.loads(path.read_text()).get("status")=="completed" and json.loads(path.read_text()).get("trigger") is None

def guard(command,prefix,stage):
    subprocess.run([sys.executable,str(ROOT/"scripts/run_gpu_guarded.py"),"--temperature-limit-c","78","--fan-limit-percent","85","--interval-sec","5","--telemetry-csv",str(prefix.with_name(prefix.name+f"_{stage}_telemetry.csv")),"--event-log",str(prefix.with_name(prefix.name+f"_{stage}_event.json")),"--",*command],cwd=ROOT,check=True)

def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    plan={"status":"registered_before_training","validation_only":True,"test_metrics_produced":False,"target":"Traffic","horizon":96,"few_shot_fraction":.25,"seeds":list(SEEDS),"factorial_design":{"source_domain":["Electricity replay","Electricity + Weather"],"representation_acf":[0,.05]},"question":"Does the replicated source-diversity gain require train-only ACF representation supervision, or is it a general multi-source forecast-pretraining effect?","fixed_contract":{"backbone":"one variable-axis Mamba scan","d_model":32,"d_ff":64,"target_variable_mamba_frozen":True,"source_domain_updates":2},"target_calibration_boundary":"only earliest 60% x 25% Traffic rows; validation and test excluded","primary_contrast":"ACF interaction = (diverse_acf - diverse_forecast) - (replay_acf - replay_forecast)","success_rule":"diverse_acf beats diverse_forecast by at least .25 percent on mean paired MSE, wins at least two of three seeds, has paired ratio upper bound below 1.0, and its improvement exceeds replay_acf versus replay_forecast"}
    (OUTPUT/"plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    for seed in SEEDS:
        for arm,(sources,fraction) in ARMS.items():
            prefix=OUTPUT/f"s{seed}_{arm}"
            pre=[sys.executable,"scripts/pretrain_variable_mamba_acf_core.py"]
            for source in sources: pre.extend(("--source",source))
            pre.extend(("--calibration-source","Traffic:dataset/traffic/traffic.csv:csv","--calibration-fraction",".25","--output",str(prefix.with_name(prefix.name+"_pretrain.pt")),"--seq-len","96","--pred-len","96","--d-model","32","--d-ff","64","--mamba-d-state","8","--mamba-d-conv","2","--train-fraction",".6","--batch-size","4","--epochs","3","--max-windows","192","--relation-loss-fraction",".1","--calibration-relation-loss-fraction",".3","--representation-relation-fraction",fraction,"--calibration-representation-relation-fraction",fraction,"--seed",str(seed)))
            ft=[sys.executable,"scripts/finetune_variable_mamba_acf_head.py","--dataset","dataset/traffic/traffic.csv","--source-checkpoint",str(prefix.with_name(prefix.name+"_pretrain.pt")),"--output",str(prefix.with_name(prefix.name+"_head.pt")),"--seq-len","96","--pred-len","96","--d-model","32","--d-ff","64","--mamba-d-state","8","--mamba-d-conv","2","--few-shot-fraction",".25","--train-fraction",".6","--val-fraction",".2","--batch-size","4","--epochs","5","--max-train-windows","64","--seed",str(seed)]
            if "--evaluate-test" in pre+ft: raise ValueError("source interaction cannot read test")
            if not(prefix.with_name(prefix.name+"_pretrain.pt").is_file() and complete(prefix.with_name(prefix.name+"_pretrain_event.json"))): guard(pre,prefix,"pretrain")
            if not(prefix.with_name(prefix.name+"_head.pt").is_file() and complete(prefix.with_name(prefix.name+"_finetune_event.json"))): guard(ft,prefix,"finetune")

if __name__=="__main__": main()
