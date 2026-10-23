#!/usr/bin/env python3
"""Read-only audit for fair 10-epoch channelwise gate screen."""
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUTPUT=ROOT/"artifacts/exploratory/electricity_d32_channelwise_gate_epoch10_screen_20261006";SEED=6047
def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def main():
 candidate=load(OUTPUT/f"s{SEED}_channelwise_gate_epoch10.json");patch=load(OUTPUT/f"electricity_h96_s{SEED}_official_patchtst_epoch10.json")
 if candidate.get("test_metrics_produced") or patch.get("test_metrics_produced"):raise ValueError("test boundary")
 if int(candidate["config"]["epochs"])!=10 or int(patch["config"]["epochs"])!=10:raise ValueError("epoch budget mismatch")
 left,right=float(candidate["best_validation_mse"]),float(patch["best_validation_mse"])
 result={"validation_only":True,"test_metrics_produced":False,"channelwise_gate_mse":left,"official_patchtst_mse":right,"candidate_vs_patchtst_percent":(left/right-1)*100,"candidate_better":left<right,"candidate_final_epoch_validation":candidate["history"][-1]["validation_mse"],"patch_final_epoch_validation":patch["history"][-1]["validation_mse"],"submission_ready":False}
 (OUTPUT/"audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps(result,sort_keys=True))
if __name__=="__main__":main()
