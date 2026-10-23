#!/usr/bin/env python3
"""Read-only audit for multi-positive ACF CPC variable Mamba gate."""
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"artifacts/exploratory/variable_mamba_acf_multipositive_probe_20261008"
SEED=6107
def load(path):return json.loads(path.read_text())
def main():
    selected=load(OUTPUT/f"s{SEED}_selected_cpc_head.json")
    high=load(OUTPUT/f"s{SEED}_high_acf_cpc_head.json")
    pre=load(OUTPUT/f"s{SEED}_high_acf_cpc_pretrain.json")
    if any(item.get("test_metrics_produced") for item in (selected,high,pre)):raise ValueError("test boundary")
    cfg=pre["config"]
    if not cfg["representation_peak_cpc"] or cfg["representation_peak_cpc_positive_mode"]!="high_acf" or float(cfg["representation_peak_cpc_threshold"])!=.3:raise ValueError("multi-positive CPC config")
    a=float(selected["best_validation_mse"]);b=float(high["best_validation_mse"])
    coverage=[row["representation_relation_active_fraction"] for row in pre["history"]]
    result={"validation_only":True,"test_metrics_produced":False,"selected_cpc_mse":a,"high_acf_cpc_mse":b,"high_acf_vs_selected_percent":(b/a-1)*100,"high_acf_active_coverage":coverage,"screen_passes_expansion_margin":bool(b/a-1<=-.0025),"submission_ready":False}
    (OUTPUT/"audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,sort_keys=True))
if __name__=="__main__":main()
