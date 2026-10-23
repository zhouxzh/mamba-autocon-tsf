#!/usr/bin/env python3
"""Read-only audit for d32/24 Electricity SSL bridge probe."""
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/"artifacts/exploratory/mambatsv3_electricity_mamba1_d32_stride24_transfer_probe_20261006";SEED=5996
def load(p):return json.loads(p.read_text(encoding="utf-8"))
def main():
 active,null=load(ART/f"s{SEED}_active_frozen_core.json"),load(ART/f"s{SEED}_null_frozen_core.json")
 if active.get("test_metrics_produced") or null.get("test_metrics_produced"):raise ValueError("test boundary")
 for arm,payload in (("active",active),("null",null)):
  cfg=payload["config"]
  if (cfg.get("mamba_ts_mixer_type"),int(cfg["patch_len"]),int(cfg["patch_stride"]),int(cfg["d_model"]),int(cfg["d_ff"]),float(cfg["few_shot_fraction"]),bool(cfg["freeze_pretrained_core"])) != ("mamba1",24,24,32,64,.25,True):raise ValueError(f"contract: {arm}")
 result={"validation_only":True,"test_metrics_produced":False,"seed":SEED,"active_mse":float(active["best_validation_mse"]),"null_mse":float(null["best_validation_mse"]),"active_vs_null_percent":(float(active["best_validation_mse"])/float(null["best_validation_mse"])-1)*100,"active_better":bool(active["best_validation_mse"]<=null["best_validation_mse"]),"frozen_d24_active_reference":8.205704689025879,"official_patchtst_reference_mean":7.5365913,"submission_ready":False}
 result["d32_improves_frozen_d24_active"]=bool(result["active_mse"]<result["frozen_d24_active_reference"])
 (ART/"audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps(result,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
