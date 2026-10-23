#!/usr/bin/env python3
"""Audit current-source same-machine clean-interpreter validation replication."""
from __future__ import annotations
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/"artifacts/reproducibility/current_clean_interpreter_validation_20261006";TRACKED=("models/MambaTS.py","scripts/pretrain_mambatsv3_core_autocon.py","scripts/finetune_mambatsv3_dlinear_hybrid.py","scripts/run_gpu_guarded.py","dataset/PEMS-BAY/PEMS-BAY.csv","dataset/PEMS07/PEMS07.npz","dataset/PEMS08/PEMS08.npz","dataset/PEMS04/PEMS04.npz")
def load(p):return json.loads(p.read_text(encoding="utf-8"))
def digest(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
def main():
 manifest=load(ART/"manifest.json");env=load(ART/"environment_check.json");active_pre,null_pre=load(ART/"s6011_active_pretrain.json"),load(ART/"s6011_null_pretrain.json");active,null=load(ART/"s6011_active_frozen_core.json"),load(ART/"s6011_null_frozen_core.json")
 if manifest.get("independent_machine") or not manifest.get("same_machine") or not manifest.get("validation_only") or manifest.get("test_metrics_produced"):raise ValueError("scope mismatch")
 if not env.get("clean_environment_ready"):raise ValueError("clean environment failed")
 if {k:digest(ROOT/k) for k in TRACKED} != manifest["source_hashes"]:raise ValueError("current source/data drift")
 for name in ("active_pretrain","null_pretrain","active_finetune","null_finetune"):
  event=load(ART/f"s6011_{'active' if name.startswith('active') else 'null'}_{'pretrain' if 'pretrain' in name else 'finetune'}_event.json")
  if event.get("status")!="completed" or event.get("trigger") is not None:raise ValueError(f"guard failure: {name}")
 if any(x.get("test_metrics_produced") for x in (active_pre,null_pre,active,null)):raise ValueError("test boundary")
 if min(h["global_relation_active_fraction"] for h in active_pre["history"])<.8:raise ValueError("active relation coverage")
 value,reference=float(active["best_validation_mse"]),float(null["best_validation_mse"])
 result={"clean_interpreter":True,"same_machine":True,"independent_machine":False,"validation_only":True,"test_metrics_produced":False,"active_validation_mse":value,"null_validation_mse":reference,"active_vs_null_percent":(value/reference-1)*100,"active_wins":int(value<reference),"audit_valid":True,"scope":"current-source same-machine clean-interpreter validation replication; not an independent-machine result"}
 (ART/"audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps(result,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
