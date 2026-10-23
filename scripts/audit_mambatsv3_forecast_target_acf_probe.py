#!/usr/bin/env python3
"""Read-only audit for the task-aligned ACF coefficient gate."""
from __future__ import annotations

import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"artifacts/exploratory/mambatsv3_forecast_target_acf_probe_20261008"
SEED=6120

def load(path): return json.loads(path.read_text())

def main():
    raw=load(OUTPUT/f"s{SEED}_raw_signal_acf_hybrid.json")
    target=load(OUTPUT/f"s{SEED}_forecast_target_acf_hybrid.json")
    pretrain=load(OUTPUT/f"s{SEED}_forecast_target_acf_pretrain.json")
    if any(payload.get("test_metrics_produced") for payload in (raw,target,pretrain)): raise ValueError("test boundary")
    if not pretrain["config"]["forecast_target_acf"]: raise ValueError("task-aligned ACF config missing")
    metadata={row["name"]:row["forecast_target_acf_metadata"] for row in pretrain["sources"]}
    raw_mse=float(raw["best_validation_mse"])
    target_mse=float(target["best_validation_mse"])
    result={"validation_only":True,"test_metrics_produced":False,"raw_signal_acf_mse":raw_mse,"forecast_target_acf_mse":target_mse,"forecast_target_vs_raw_percent":(target_mse/raw_mse-1)*100,"source_task_acf_metadata":metadata,"screen_passes_expansion_margin":bool(target_mse/raw_mse-1<=-.0025),"submission_ready":False}
    (OUTPUT/"audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,sort_keys=True))

if __name__=="__main__": main()
