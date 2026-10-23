#!/usr/bin/env python3
"""Read-only audit for d32/24 Traffic SSL bridge probe."""
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/"artifacts/exploratory/mambatsv3_traffic_mamba1_d32_stride24_transfer_probe_20261006";SEED=6006
def load(p):return json.loads(p.read_text(encoding="utf-8"))
def main():
 a,n=load(ART/f"s{SEED}_active_frozen_core.json"),load(ART/f"s{SEED}_null_frozen_core.json")
 if a.get("test_metrics_produced") or n.get("test_metrics_produced"):raise ValueError("test boundary")
 for arm,x in (("active",a),("null",n)):
  c=x["config"]
  if (c.get("mamba_ts_mixer_type"),int(c["patch_len"]),int(c["patch_stride"]),int(c["d_model"]),int(c["d_ff"]),float(c["few_shot_fraction"]),bool(c["freeze_pretrained_core"])) != ("mamba1",24,24,32,64,.25,True):raise ValueError(f"contract: {arm}")
 result={"validation_only":True,"test_metrics_produced":False,"seed":SEED,"active_mse":float(a["best_validation_mse"]),"null_mse":float(n["best_validation_mse"]),"active_vs_null_percent":(float(a["best_validation_mse"])/float(n["best_validation_mse"])-1)*100,"active_better":bool(a["best_validation_mse"]<=n["best_validation_mse"]),"d24_source_only_active_reference":.817744,"submission_ready":False}
 result["d32_improves_d24_source_only"]=bool(result["active_mse"]<result["d24_source_only_active_reference"])
 (ART/"audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps(result,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
