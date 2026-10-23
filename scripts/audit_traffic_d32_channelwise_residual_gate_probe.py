#!/usr/bin/env python3
"""Read-only audit for Traffic active/null d32 channelwise target gate."""
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUTPUT=ROOT/"artifacts/exploratory/traffic_d32_channelwise_residual_gate_probe_20261006";REFERENCE=ROOT/"artifacts/exploratory/mambatsv3_traffic_mamba1_d32_stride24_transfer_probe_20261006";SEED=6006
def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def main():
 active=load(OUTPUT/f"s{SEED}_active_channelwise_gate.json");null=load(OUTPUT/f"s{SEED}_null_channelwise_gate.json");plain_active=load(REFERENCE/f"s{SEED}_active_frozen_core.json")
 if any(item.get("test_metrics_produced") for item in (active,null,plain_active)):raise ValueError("test boundary")
 for item in (active,null):
  if not item["config"].get("mamba_channelwise_residual_scale"):raise ValueError("gate config missing")
  if not all(row["residual_scale_shape"]==[1,1,item["target_channels"]] for row in item["history"]):raise ValueError("gate shape mismatch")
 a,n,p=float(active["best_validation_mse"]),float(null["best_validation_mse"]),float(plain_active["best_validation_mse"])
 result={"validation_only":True,"test_metrics_produced":False,"active_gate_mse":a,"null_gate_mse":n,"plain_active_mse":p,"active_vs_null_percent":(a/n-1)*100,"active_vs_plain_percent":(a/p-1)*100,"active_better_than_null":a<n,"submission_ready":False}
 (OUTPUT/"audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps(result,sort_keys=True))
if __name__=="__main__":main()
