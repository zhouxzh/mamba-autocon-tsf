#!/usr/bin/env python3
"""Read-only audit for d32/24 channel-patch Electricity gate."""
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUTPUT=ROOT/"artifacts/exploratory/electricity_d32_channel_patch_probe_20261006";SEED=6043
def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def main():
 forecast=load(OUTPUT/f"s{SEED}_forecast_frozen_core.json");candidate=load(OUTPUT/f"s{SEED}_channel_patch_frozen_core.json");source=load(OUTPUT/f"s{SEED}_channel_patch_pretrain.json")
 if any(item.get("test_metrics_produced") for item in (forecast,candidate,source)):raise ValueError("test boundary")
 if float(source["config"].get("channel_patch_relation_fraction",0))!=.05:raise ValueError("channel-patch config mismatch")
 if not all(float(item.get("channel_patch_relation_active_fraction",0))>0 for item in source["history"]):raise ValueError("inactive channel-patch anchors")
 left,right=float(candidate["best_validation_mse"]),float(forecast["best_validation_mse"])
 result={"validation_only":True,"test_metrics_produced":False,"forecast_mse":right,"channel_patch_mse":left,"channel_patch_vs_forecast_percent":(left/right-1)*100,"channel_patch_better":left<right,"submission_ready":False}
 (OUTPUT/"audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps(result,sort_keys=True))
if __name__=="__main__":main()
