#!/usr/bin/env python3
"""Read-only audit for variable-token Mamba representation ACF strength."""
from __future__ import annotations

import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"artifacts/exploratory/variable_mamba_acf_strength_probe_20261007"
SEED=6103

def load(path): return json.loads(path.read_text())

def main():
    low=load(OUTPUT/f"s{SEED}_repr_0p05_head.json")
    high=load(OUTPUT/f"s{SEED}_repr_0p10_head.json")
    pre=load(OUTPUT/f"s{SEED}_repr_0p10_pretrain.json")
    if any(item.get("test_metrics_produced") for item in (low,high,pre)):
        raise ValueError("test boundary")
    if pre["config"]["representation_relation_fraction"]!=.1:
        raise ValueError(".10 config missing")
    low_mse=float(low["best_validation_mse"]);high_mse=float(high["best_validation_mse"])
    result={"validation_only":True,"test_metrics_produced":False,"repr_0p05_mse":low_mse,"repr_0p10_mse":high_mse,"repr_0p10_vs_0p05_percent":(high_mse/low_mse-1)*100,"screen_passes_expansion_margin":bool(high_mse/low_mse-1<=-.0025),"submission_ready":False}
    (OUTPUT/"audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,sort_keys=True))

if __name__=="__main__": main()
