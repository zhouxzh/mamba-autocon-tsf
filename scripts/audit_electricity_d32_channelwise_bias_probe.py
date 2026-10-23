#!/usr/bin/env python3
"""Read-only audit for d32 channelwise bias gate."""
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUTPUT=ROOT/"artifacts/exploratory/electricity_d32_channelwise_bias_probe_20261006";REFERENCE=ROOT/"artifacts/exploratory/electricity_d32_channelwise_residual_gate_replication_rerun_20261006";SEED=6047
def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def main():
 candidate=load(OUTPUT/f"s{SEED}_channelwise_bias.json");control=load(REFERENCE/f"s{SEED}_channelwise_gate.json")
 if candidate.get("test_metrics_produced") or control.get("test_metrics_produced"):raise ValueError("test boundary")
 if not candidate["config"].get("mamba_channelwise_residual_bias"):raise ValueError("bias config missing")
 if not all(row["residual_bias_shape"]==[1,96,candidate["target_channels"]] for row in candidate["history"]):raise ValueError("bias shape mismatch")
 left,right=float(candidate["best_validation_mse"]),float(control["best_validation_mse"])
 result={"validation_only":True,"test_metrics_produced":False,"channelwise_bias_mse":left,"channelwise_scale_mse":right,"bias_vs_scale_percent":(left/right-1)*100,"bias_better":left<right,"final_bias_abs_max":candidate["history"][-1]["residual_bias_abs_max"],"submission_ready":False}
 (OUTPUT/"audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps(result,sort_keys=True))
if __name__=="__main__":main()
