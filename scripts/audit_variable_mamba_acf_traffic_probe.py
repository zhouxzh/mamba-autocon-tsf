#!/usr/bin/env python3
"""Read-only audit for variable-token Mamba ACF Traffic gate."""
from __future__ import annotations

import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"artifacts/exploratory/variable_mamba_acf_traffic_probe_20261007"
SEED=6102

def load(path):
    return json.loads(path.read_text())

def main():
    forecast=load(OUTPUT/f"s{SEED}_forecast_head.json")
    candidate=load(OUTPUT/f"s{SEED}_representation_acf_head.json")
    pretrain=load(OUTPUT/f"s{SEED}_representation_acf_pretrain.json")
    if any(item.get("test_metrics_produced") for item in (forecast,candidate,pretrain)):
        raise ValueError("test boundary")
    if pretrain["config"]["representation_relation_fraction"]!=.05 or pretrain["config"]["calibration_representation_relation_fraction"]!=.05:
        raise ValueError("representation ACF contract")
    values={"forecast":float(forecast["best_validation_mse"]),"representation_acf":float(candidate["best_validation_mse"])}
    result={"validation_only":True,"test_metrics_produced":False,"mse":values,"representation_acf_vs_forecast_percent":(values["representation_acf"]/values["forecast"]-1)*100,"screen_passes_expansion_margin":bool(values["representation_acf"]/values["forecast"]-1<=-.0025),"variable_mamba_frozen":all(row["variable_mamba_frozen"] for row in candidate["history"]),"mamba_scan_count":1,"submission_ready":False}
    (OUTPUT/"audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,sort_keys=True))

if __name__=="__main__":
    main()
