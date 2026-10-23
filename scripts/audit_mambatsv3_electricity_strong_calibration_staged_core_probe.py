#!/usr/bin/env python3
"""Read-only audit of Electricity staged-core strong-calibration adaptation."""
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/"artifacts/exploratory/mambatsv3_electricity_mamba1_strong_calibration_staged_core_probe_20261006";SEED=5948
def load(p):return json.loads(p.read_text(encoding="utf-8"))
def main():
 active,null=load(ART/f"s{SEED}_active_staged_core.json"),load(ART/f"s{SEED}_null_staged_core.json")
 if active.get("test_metrics_produced") or null.get("test_metrics_produced"):raise ValueError("test boundary")
 for arm,payload in (("active",active),("null",null)):
  cfg=payload["config"]
  if float(cfg["mamba_core_learning_rate"])!=.0001 or "--freeze-pretrained-core" in cfg:raise ValueError(f"staged contract: {arm}")
  history=payload["history"]
  if history[0]["mamba_frozen"] is not True or any(row["mamba_frozen"] for row in history[1:]):raise ValueError(f"warmup contract: {arm}")
  if not any(row["mamba_core_frozen"] is False for row in history[1:]):raise ValueError(f"core did not adapt: {arm}")
 result={"validation_only":True,"test_metrics_produced":False,"seed":SEED,"active_mse":float(active["best_validation_mse"]),"null_mse":float(null["best_validation_mse"]),"active_vs_null_percent":(float(active["best_validation_mse"])/float(null["best_validation_mse"])-1)*100,"active_better":bool(active["best_validation_mse"]<=null["best_validation_mse"]),"submission_ready":False}
 (ART/"audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps(result,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
