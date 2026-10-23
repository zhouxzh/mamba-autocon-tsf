#!/usr/bin/env python3
"""Read-only audit for target forecast fusion of direct-relation representations."""
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/"artifacts/exploratory/mambatsv3_mamba1_repr_fusion_transfer_probe_20261006";SOURCE=ROOT/"artifacts/exploratory/mambatsv3_mamba1_representation_relation_probe_20261006";SEED=5990;TARGETS=("pems_bay","metr_la")
def load(p):return json.loads(p.read_text(encoding="utf-8"))
def main():
 result={"validation_only":True,"test_metrics_produced":False,"targets":{},"submission_ready":False}
 for target in TARGETS:
  values={}
  for arm in ("plain_fusion","representation_fusion"):
   prefix=ART/target/f"s{SEED}_{arm}";payload=load(prefix.with_name(prefix.name+"_frozen_core.json"));cfg=payload["config"]
   if payload.get("test_metrics_produced"):raise ValueError("test boundary")
   if (cfg.get("mamba_ts_mixer_type"),int(cfg["patch_len"]),int(cfg["patch_stride"]),float(cfg["few_shot_fraction"]),bool(cfg["freeze_pretrained_core"]),bool(cfg["mamba_ts_repr_forecast_fusion"])) != ("mamba1",24,8,.1,True,True):raise ValueError(f"fusion contract: {target}/{arm}")
   values[arm]=float(payload["best_validation_mse"])
  plain_no_fusion=load(SOURCE/target/f"s{SEED}_plain_frozen_core.json")["best_validation_mse"];repr_no_fusion=load(SOURCE/target/f"s{SEED}_representation_frozen_core.json")["best_validation_mse"]
  result["targets"][target]={"mse":{**values,"plain_no_fusion":float(plain_no_fusion),"representation_no_fusion":float(repr_no_fusion)},"representation_fusion_vs_plain_fusion_percent":(values["representation_fusion"]/values["plain_fusion"]-1)*100,"representation_fusion_vs_representation_no_fusion_percent":(values["representation_fusion"]/float(repr_no_fusion)-1)*100,"representation_fusion_beats_plain_fusion":values["representation_fusion"]<=values["plain_fusion"]}
 result["cross_domain_gate"]=all(v["representation_fusion_beats_plain_fusion"] for v in result["targets"].values());result["selection_outcome"]="positive_deployed_representation_signal" if result["cross_domain_gate"] else "stop_no_deployed_representation_gain"
 (ART/"audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps(result,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
