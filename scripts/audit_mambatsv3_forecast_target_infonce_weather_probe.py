#!/usr/bin/env python3
"""Read-only audit for task-sign InfoNCE Weather H192 gate."""
from __future__ import annotations

import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"artifacts/exploratory/mambatsv3_forecast_target_infonce_weather_probe_20261008"
SEED=6121

def load(path): return json.loads(path.read_text())

def main():
    raw=load(OUTPUT/f"s{SEED}_raw_peak_sign_hybrid.json")
    task=load(OUTPUT/f"s{SEED}_forecast_target_sign_hybrid.json")
    pretrain=load(OUTPUT/f"s{SEED}_forecast_target_sign_pretrain.json")
    if any(payload.get("test_metrics_produced") for payload in (raw,task,pretrain)): raise ValueError("test boundary")
    if not pretrain["config"]["forecast_target_acf"] or pretrain["config"]["relation_view"]!="forecast_contrastive": raise ValueError("task-sign InfoNCE contract")
    raw_mse=float(raw["best_validation_mse"])
    task_mse=float(task["best_validation_mse"])
    result={"validation_only":True,"test_metrics_produced":False,"raw_peak_sign_mse":raw_mse,"forecast_target_sign_mse":task_mse,"task_sign_vs_raw_percent":(task_mse/raw_mse-1)*100,"source_task_acf_metadata":{row["name"]:row["forecast_target_acf_metadata"] for row in pretrain["sources"]},"screen_passes_expansion_margin":bool(task_mse/raw_mse-1<=-.0025),"submission_ready":False}
    (OUTPUT/"audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,sort_keys=True))

if __name__=="__main__": main()
