#!/usr/bin/env python3
"""Read-only audit for channelwise ACF variable-token Mamba gate."""
from __future__ import annotations

import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"artifacts/exploratory/variable_mamba_acf_channelwise_probe_20261008"
SEED=6108

def load(path): return json.loads(path.read_text())

def main():
    global_head=load(OUTPUT/f"s{SEED}_global_relation_head.json")
    channel_head=load(OUTPUT/f"s{SEED}_channelwise_relation_head.json")
    channel_pretrain=load(OUTPUT/f"s{SEED}_channelwise_relation_pretrain.json")
    if any(item.get("test_metrics_produced") for item in (global_head,channel_head,channel_pretrain)): raise ValueError("test boundary")
    cfg=channel_pretrain["config"]
    if not cfg["representation_channelwise_acf"] or cfg["representation_peak_cpc"]: raise ValueError("channelwise ACF objective contract")
    global_mse=float(global_head["best_validation_mse"])
    channel_mse=float(channel_head["best_validation_mse"])
    result={"validation_only":True,"test_metrics_produced":False,"global_relation_mse":global_mse,"channelwise_relation_mse":channel_mse,"channelwise_vs_global_percent":(channel_mse/global_mse-1)*100,"channelwise_active_coverage":[row["representation_relation_active_fraction"] for row in channel_pretrain["history"]],"screen_passes_expansion_margin":bool(channel_mse/global_mse-1<=-.0025),"variable_mamba_frozen":all(row["variable_mamba_frozen"] for row in channel_head["history"]),"mamba_scan_count":1,"submission_ready":False}
    (OUTPUT/"audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,sort_keys=True))

if __name__=="__main__": main()
