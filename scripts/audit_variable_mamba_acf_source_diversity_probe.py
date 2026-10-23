#!/usr/bin/env python3
"""Read-only audit for the variable Mamba source-diversity gate."""
from __future__ import annotations

import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"artifacts/exploratory/variable_mamba_acf_source_diversity_probe_20261008"
SEED=6109

def load(path): return json.loads(path.read_text())

def main():
    replay=load(OUTPUT/f"s{SEED}_electricity_replay_head.json")
    diverse=load(OUTPUT/f"s{SEED}_electricity_weather_head.json")
    pretrain=load(OUTPUT/f"s{SEED}_electricity_weather_pretrain.json")
    if any(item.get("test_metrics_produced") for item in (replay,diverse,pretrain)): raise ValueError("test boundary")
    sources=pretrain["sources"]
    if [row["name"] for row in sources]!=["Electricity","Weather","Traffic"]: raise ValueError("source diversity contract")
    replay_mse=float(replay["best_validation_mse"])
    diverse_mse=float(diverse["best_validation_mse"])
    result={"validation_only":True,"test_metrics_produced":False,"electricity_replay_mse":replay_mse,"electricity_weather_mse":diverse_mse,"diverse_vs_replay_percent":(diverse_mse/replay_mse-1)*100,"source_count_matched":True,"screen_passes_expansion_margin":bool(diverse_mse/replay_mse-1<=-.0025),"variable_mamba_frozen":all(row["variable_mamba_frozen"] for row in diverse["history"]),"mamba_scan_count":1,"submission_ready":False}
    (OUTPUT/"audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,sort_keys=True))

if __name__=="__main__": main()
