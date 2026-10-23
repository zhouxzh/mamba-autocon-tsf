#!/usr/bin/env python3
"""Read-only audit for current-geometry 25% channel-patch SSL replication."""
from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUTPUT=ROOT/"artifacts/exploratory/channel_patch_25pct_replication_20261006";SEEDS={"pems_bay":(6036,6037,6038),"metr_la":(6039,6040,6041)}
def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def summary(left,right):
 ratio=left/right;return {"left_mean_mse":float(left.mean()),"right_mean_mse":float(right.mean()),"percent":float((left.mean()/right.mean()-1)*100),"wins":int((left<right).sum()),"ratio_upper_1se":float(ratio.mean()+ratio.std(ddof=1)/math.sqrt(len(ratio))),"noninferior_at_one_se":bool(ratio.mean()+ratio.std(ddof=1)/math.sqrt(len(ratio))<=1)}
def main():
 result={"validation_only":True,"test_metrics_produced":False,"targets":{},"submission_ready":False}
 for target,seeds in SEEDS.items():
  rows=[]
  for seed in seeds:
   records={arm:load(OUTPUT/target/f"s{seed}_{arm}_frozen_core.json") for arm in ("null","forecast","channel_patch")};source=load(OUTPUT/target/f"s{seed}_channel_patch_pretrain.json")
   if any(item.get("test_metrics_produced") for item in (*records.values(),source)):raise ValueError("test boundary violated")
   if float(source["config"].get("channel_patch_relation_fraction",0))!=.05:raise ValueError("channel-patch contract mismatch")
   if not all(float(item.get("channel_patch_relation_active_fraction",0))>0 for item in source["history"]):raise ValueError("inactive patch anchors")
   if float(records["channel_patch"]["config"]["few_shot_fraction"])!=.25:raise ValueError("label fraction mismatch")
   rows.append({"seed":seed,**{arm:float(item["best_validation_mse"]) for arm,item in records.items()}})
  vals={arm:np.asarray([row[arm] for row in rows]) for arm in ("null","forecast","channel_patch")};result["targets"][target]={"rows":rows,"forecast_vs_null":summary(vals["forecast"],vals["null"]),"channel_patch_vs_null":summary(vals["channel_patch"],vals["null"]),"channel_patch_vs_forecast":summary(vals["channel_patch"],vals["forecast"])}
 result["external_baseline_eligible"]=bool(all(item["channel_patch_vs_forecast"]["noninferior_at_one_se"] and item["channel_patch_vs_forecast"]["wins"]>=2 for item in result["targets"].values()))
 (OUTPUT/"audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps(result,sort_keys=True))
if __name__=="__main__":main()
