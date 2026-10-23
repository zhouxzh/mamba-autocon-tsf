#!/usr/bin/env python3
"""Read-only audit for direct representation-relation pretraining."""
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/"artifacts/exploratory/mambatsv3_mamba1_representation_relation_probe_20261006";SEED=5990;TARGETS=("pems_bay","metr_la")
def load(p):return json.loads(p.read_text(encoding="utf-8"))
def main():
 result={"validation_only":True,"test_metrics_produced":False,"targets":{},"submission_ready":False}
 for target in TARGETS:
  values={}
  for arm in ("null","plain","representation"):
   prefix=ART/target/f"s{SEED}_{arm}";pre,ft=load(prefix.with_name(prefix.name+"_pretrain.json")),load(prefix.with_name(prefix.name+"_frozen_core.json"))
   if pre.get("test_metrics_produced") or ft.get("test_metrics_produced"):raise ValueError("test boundary")
   cfg=ft["config"]
   if (cfg.get("mamba_ts_mixer_type"),int(cfg["patch_len"]),int(cfg["patch_stride"]),float(cfg["few_shot_fraction"]),bool(cfg["freeze_pretrained_core"])) != ("mamba1",24,8,.1,True):raise ValueError(f"transfer: {target}/{arm}")
   expected={"null":(0.,0.),"plain":(.1,0.),"representation":(.1,.05)}[arm]
   actual=(float(pre["config"].get("relation_loss_fraction",-1)),float(pre["config"].get("representation_relation_fraction",-1)))
   if actual!=expected:raise ValueError(f"pretrain: {target}/{arm}:{actual}")
   if arm!="null" and min(float(h.get("global_relation_active_fraction",0)) for h in pre["history"])<.8:raise ValueError(f"forecast coverage: {target}/{arm}")
   if arm=="representation" and min(float(h.get("representation_relation_active_fraction",0)) for h in pre["history"])<.8:raise ValueError(f"representation coverage: {target}")
   values[arm]=float(ft["best_validation_mse"])
  result["targets"][target]={"mse":values,"representation_vs_null_percent":(values["representation"]/values["null"]-1)*100,"representation_vs_plain_percent":(values["representation"]/values["plain"]-1)*100,"representation_beats_plain":values["representation"]<=values["plain"]}
 result["cross_domain_gate"]=all(v["representation_beats_plain"] for v in result["targets"].values());result["selection_outcome"]="positive_mechanism_signal" if result["cross_domain_gate"] else "stop_no_cross_domain_gain"
 (ART/"audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps(result,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
