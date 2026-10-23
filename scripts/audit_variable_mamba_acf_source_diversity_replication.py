#!/usr/bin/env python3
"""Read-only audit for fresh-seed source diversity replication."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"artifacts/exploratory/variable_mamba_acf_source_diversity_replication_20261008"
SEEDS=(6110,6111,6112)

def load(path): return json.loads(path.read_text())

def main():
    replay=[]
    diverse=[]
    frozen=[]
    for seed in SEEDS:
        control=load(OUTPUT/f"s{seed}_electricity_replay_head.json")
        candidate=load(OUTPUT/f"s{seed}_electricity_weather_head.json")
        if control.get("test_metrics_produced") or candidate.get("test_metrics_produced"): raise ValueError("test boundary")
        replay.append(float(control["best_validation_mse"]))
        diverse.append(float(candidate["best_validation_mse"]))
        frozen.append(all(row["variable_mamba_frozen"] for row in candidate["history"]))
    ratios=np.asarray(diverse)/np.asarray(replay)
    mean_ratio=float(ratios.mean())
    sd=float(ratios.std(ddof=1))
    upper=mean_ratio+4.303*sd/math.sqrt(len(ratios))
    result={"validation_only":True,"test_metrics_produced":False,"seeds":list(SEEDS),"electricity_replay_mse":replay,"electricity_weather_mse":diverse,"mean_replay_mse":float(np.mean(replay)),"mean_diverse_mse":float(np.mean(diverse)),"diverse_vs_replay_percent":(mean_ratio-1)*100,"paired_wins":int(np.sum(np.asarray(diverse)<np.asarray(replay))),"paired_ratio_upper_95":upper,"success_rule_passes":bool(mean_ratio<=.9975 and int(np.sum(np.asarray(diverse)<np.asarray(replay)))>=2 and upper<1.0),"variable_mamba_frozen":all(frozen),"mamba_scan_count":1,"submission_ready":False}
    (OUTPUT/"audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,sort_keys=True))

if __name__=="__main__": main()
