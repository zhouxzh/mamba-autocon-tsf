#!/usr/bin/env python3
"""Read-only audit for d32 channelwise residual-gate replication."""
from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUTPUT=ROOT/"artifacts/exploratory/electricity_d32_channelwise_residual_gate_replication_rerun_20261006";SEEDS=(6047,6048,6049)
def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def main():
 rows=[]
 for seed in SEEDS:
  gate=load(OUTPUT/f"s{seed}_channelwise_gate.json");forecast=load(OUTPUT/f"s{seed}_global_gate.json")
  if gate.get("test_metrics_produced") or forecast.get("test_metrics_produced"):raise ValueError("test boundary")
  if not gate["config"].get("mamba_channelwise_residual_scale"):raise ValueError("channelwise config missing")
  if not all(row["residual_scale_shape"]==[1,1,gate["target_channels"]] for row in gate["history"]):raise ValueError("gate shape mismatch")
  rows.append({"seed":seed,"forecast":float(forecast["best_validation_mse"]),"channelwise_gate":float(gate["best_validation_mse"]),"gate_abs_max":float(gate["history"][-1]["residual_scale_abs_max"])})
 forecast=np.asarray([row["forecast"] for row in rows]);gate=np.asarray([row["channelwise_gate"] for row in rows]);ratio=gate/forecast
 result={"validation_only":True,"test_metrics_produced":False,"rows":rows,"channelwise_gate_vs_forecast":{"gate_mean_mse":float(gate.mean()),"forecast_mean_mse":float(forecast.mean()),"percent":float((gate.mean()/forecast.mean()-1)*100),"wins":int((gate<forecast).sum()),"ratio_upper_1se":float(ratio.mean()+ratio.std(ddof=1)/math.sqrt(len(ratio))),"noninferior_at_one_se":bool(ratio.mean()+ratio.std(ddof=1)/math.sqrt(len(ratio))<=1)},"submission_ready":False}
 result["external_comparator_eligible"]=bool(result["channelwise_gate_vs_forecast"]["wins"]>=2 and result["channelwise_gate_vs_forecast"]["noninferior_at_one_se"])
 (OUTPUT/"audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps(result,sort_keys=True))
if __name__=="__main__":main()
