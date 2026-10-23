#!/usr/bin/env python3
"""Read-only audit for source-diversity and ACF-objective interaction."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"artifacts/exploratory/variable_mamba_acf_source_interaction_20261008"
SEEDS=(6113,6114,6115)
ARMS=("replay_forecast","replay_acf","diverse_forecast","diverse_acf")

def load(path): return json.loads(path.read_text())

def upper_95(ratios):
    values=np.asarray(ratios,dtype=float)
    return float(values.mean()+4.303*values.std(ddof=1)/math.sqrt(len(values)))

def main():
    scores={arm:[] for arm in ARMS}
    frozen=[]
    for seed in SEEDS:
        for arm in ARMS:
            payload=load(OUTPUT/f"s{seed}_{arm}_head.json")
            if payload.get("test_metrics_produced"): raise ValueError("test boundary")
            scores[arm].append(float(payload["best_validation_mse"]))
            frozen.append(all(row["variable_mamba_frozen"] for row in payload["history"]))
    replay_forecast=np.asarray(scores["replay_forecast"])
    replay_acf=np.asarray(scores["replay_acf"])
    diverse_forecast=np.asarray(scores["diverse_forecast"])
    diverse_acf=np.asarray(scores["diverse_acf"])
    replay_ratio=replay_acf/replay_forecast
    diverse_ratio=diverse_acf/diverse_forecast
    replay_change=float((replay_ratio.mean()-1)*100)
    diverse_change=float((diverse_ratio.mean()-1)*100)
    result={"validation_only":True,"test_metrics_produced":False,"seeds":list(SEEDS),"mse":scores,"mean_mse":{arm:float(np.mean(values)) for arm,values in scores.items()},"replay_acf_vs_forecast_percent":replay_change,"diverse_acf_vs_forecast_percent":diverse_change,"acf_interaction_percent":diverse_change-replay_change,"diverse_acf_wins":int(np.sum(diverse_acf<diverse_forecast)),"diverse_acf_ratio_upper_95":upper_95(diverse_ratio),"success_rule_passes":bool(diverse_ratio.mean()<=.9975 and int(np.sum(diverse_acf<diverse_forecast))>=2 and upper_95(diverse_ratio)<1 and diverse_change<replay_change),"variable_mamba_frozen":all(frozen),"mamba_scan_count":1,"submission_ready":False}
    (OUTPUT/"audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,sort_keys=True))

if __name__=="__main__": main()
