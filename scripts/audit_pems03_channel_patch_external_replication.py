#!/usr/bin/env python3
"""Read-only audit for current-geometry PEMS03 external replication."""
from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUTPUT=ROOT/"artifacts/exploratory/pems03_channel_patch_external_replication_rerun_20261007";PATCH=ROOT/"artifacts/exploratory/official_patchtst_pems03_channel_patch_external_replication_rerun_20261007";SEEDS=(6055,6056,6057)
def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def values(command,flag):return [command[index+1] for index,item in enumerate(command[:-1]) if item==flag]
def summary(left,right):
 ratio=left/right;return {"left_mean_mse":float(left.mean()),"right_mean_mse":float(right.mean()),"percent":float((left.mean()/right.mean()-1)*100),"wins":int((left<right).sum()),"ratio_upper_1se":float(ratio.mean()+ratio.std(ddof=1)/math.sqrt(len(ratio))),"noninferior_at_one_se":bool(ratio.mean()+ratio.std(ddof=1)/math.sqrt(len(ratio))<=1)}
def main():
 plan=load(OUTPUT/"plan.json")
 if not plan.get("target_excluded_from_source_and_initialization") or any("pems03" in item.lower() for item in plan["initialization_checkpoints"]):raise ValueError("invalid external plan")
 rows=[]
 for seed in SEEDS:
  records={arm:load(OUTPUT/f"s{seed}_{arm}_frozen_core.json") for arm in ("null","forecast","channel_patch")};patch=load(PATCH/f"pems03_h192_s{seed}_official_patchtst.json");source=load(OUTPUT/f"s{seed}_channel_patch_pretrain.json")
  if any(item.get("test_metrics_produced") for item in (*records.values(),patch,source)):raise ValueError("test boundary")
  event=load(OUTPUT/f"s{seed}_channel_patch_pretrain_event.json")
  if any("pems03" in item.lower() for item in values(event["command"],"--source")+values(event["command"],"--initial-checkpoint")):raise ValueError("target leaked into source or initialization")
  if any(float(item["config"]["few_shot_fraction"])!=.1 for item in (*records.values(),patch)):raise ValueError("label mismatch")
  if not all(float(item.get("channel_patch_relation_active_fraction",0))>0 for item in source["history"]):raise ValueError("inactive patch anchors")
  rows.append({"seed":seed,**{arm:float(item["best_validation_mse"]) for arm,item in records.items()},"official_patchtst":float(patch["best_validation_mse"])})
 metrics={name:np.asarray([row[name] for row in rows]) for name in ("null","forecast","channel_patch","official_patchtst")}
 result={"validation_only":True,"test_metrics_produced":False,"target_excluded_from_source_and_initialization":True,"rows":rows,"channel_patch_vs_null":summary(metrics["channel_patch"],metrics["null"]),"channel_patch_vs_forecast":summary(metrics["channel_patch"],metrics["forecast"]),"channel_patch_vs_official_patchtst":summary(metrics["channel_patch"],metrics["official_patchtst"]),"submission_ready":False}
 result["external_eligible"]=bool(result["channel_patch_vs_forecast"]["noninferior_at_one_se"] and result["channel_patch_vs_official_patchtst"]["noninferior_at_one_se"])
 (OUTPUT/"audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps(result,sort_keys=True))
if __name__=="__main__":main()
