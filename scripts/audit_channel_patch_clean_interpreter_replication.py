#!/usr/bin/env python3
"""Read-only audit for same-machine clean-interpreter channel-patch run."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/"artifacts/reproducibility/channel_patch_clean_interpreter_validation_20261006";TRACKED=("models/MambaTS.py","layers/losses.py","scripts/pretrain_mambatsv3_core_autocon.py","scripts/finetune_mambatsv3_core_transfer.py","scripts/finetune_mambatsv3_dlinear_hybrid.py","scripts/run_gpu_guarded.py","dataset/PEMS-BAY/PEMS-BAY.csv","dataset/PEMS07/PEMS07.npz","dataset/PEMS08/PEMS08.npz","dataset/PEMS04/PEMS04.npz")
def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def digest(path):
 value=hashlib.sha256()
 with Path(path).open("rb") as handle:
  for block in iter(lambda:handle.read(1024*1024),b""):value.update(block)
 return value.hexdigest()
def main():
 manifest=load(ART/"manifest.json");env=load(ART/"environment_check.json")
 if manifest.get("independent_machine") or not manifest.get("same_machine") or not env.get("clean_environment_ready"):raise ValueError("scope/environment mismatch")
 if {name:digest(ROOT/name) for name in TRACKED}!=manifest["source_hashes"]:raise ValueError("current source/data drift")
 forecast_pre,candidate_pre=load(ART/"s6042_forecast_pretrain.json"),load(ART/"s6042_channel_patch_pretrain.json");forecast,candidate=load(ART/"s6042_forecast_frozen_core.json"),load(ART/"s6042_channel_patch_frozen_core.json")
 if any(item.get("test_metrics_produced") for item in (forecast_pre,candidate_pre,forecast,candidate)):raise ValueError("test boundary")
 for arm in ("forecast","channel_patch"):
  for stage in ("pretrain","finetune"):
   event=load(ART/f"s6042_{arm}_{stage}_event.json")
   if event.get("status")!="completed" or event.get("trigger") is not None:raise ValueError("guard failure")
 if not all(float(row.get("channel_patch_relation_active_fraction",0))>0 for row in candidate_pre["history"]):raise ValueError("inactive channel-patch anchors")
 left,right=float(candidate["best_validation_mse"]),float(forecast["best_validation_mse"])
 result={"clean_interpreter":True,"same_machine":True,"independent_machine":False,"validation_only":True,"test_metrics_produced":False,"channel_patch_validation_mse":left,"forecast_validation_mse":right,"channel_patch_vs_forecast_percent":(left/right-1)*100,"channel_patch_better":left<right,"audit_valid":True,"scope":"current-source same-machine clean-interpreter channel-patch validation; not independent-machine evidence"}
 (ART/"audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps(result,sort_keys=True))
if __name__=="__main__":main()
