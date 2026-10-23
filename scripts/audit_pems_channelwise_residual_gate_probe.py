#!/usr/bin/env python3
"""Read-only audit for PEMS channelwise target gate probe."""
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUTPUT=ROOT/"artifacts/exploratory/pems_channelwise_residual_gate_probe_20261006";REFERENCE=ROOT/"artifacts/exploratory/mambats_channel_patch_replication_20261006/pems_bay";SEED=6026
def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def main():
 candidate=load(OUTPUT/f"s{SEED}_channelwise_gate.json");control=load(REFERENCE/f"s{SEED}_channel_patch_frozen_core.json")
 if candidate.get("test_metrics_produced") or control.get("test_metrics_produced"):raise ValueError("test boundary")
 if not candidate["config"].get("mamba_channelwise_residual_scale"):raise ValueError("gate config missing")
 if not all(row["residual_scale_shape"]==[1,1,candidate["target_channels"]] for row in candidate["history"]):raise ValueError("gate shape mismatch")
 left,right=float(candidate["best_validation_mse"]),float(control["best_validation_mse"])
 result={"validation_only":True,"test_metrics_produced":False,"channelwise_gate_mse":left,"channel_patch_mse":right,"channelwise_vs_channel_patch_percent":(left/right-1)*100,"channelwise_better":left<right,"final_gate_abs_max":candidate["history"][-1]["residual_scale_abs_max"],"submission_ready":False}
 (OUTPUT/"audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps(result,sort_keys=True))
if __name__=="__main__":main()
