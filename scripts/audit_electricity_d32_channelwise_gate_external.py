#!/usr/bin/env python3
"""Read-only same-seed official PatchTST audit for d32 channelwise gates."""
from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];CANDIDATE=ROOT/"artifacts/exploratory/electricity_d32_channelwise_residual_gate_replication_rerun_20261006";PATCH=ROOT/"artifacts/exploratory/official_patchtst_electricity_d32_channelwise_gate_20261006";SEEDS=(6047,6048,6049)
def load(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def main():
 rows=[]
 for seed in SEEDS:
  candidate=load(CANDIDATE/f"s{seed}_channelwise_gate.json");patch=load(PATCH/f"electricity_h96_s{seed}_official_patchtst.json")
  if candidate.get("test_metrics_produced") or patch.get("test_metrics_produced"):raise ValueError("test boundary")
  rows.append({"seed":seed,"channelwise_gate":float(candidate["best_validation_mse"]),"official_patchtst":float(patch["best_validation_mse"])})
 candidate=np.asarray([row["channelwise_gate"] for row in rows]);patch=np.asarray([row["official_patchtst"] for row in rows]);ratio=candidate/patch
 result={"validation_only":True,"test_metrics_produced":False,"rows":rows,"channelwise_gate_vs_official_patchtst":{"candidate_mean_mse":float(candidate.mean()),"official_patchtst_mean_mse":float(patch.mean()),"percent":float((candidate.mean()/patch.mean()-1)*100),"wins":int((candidate<patch).sum()),"ratio_upper_1se":float(ratio.mean()+ratio.std(ddof=1)/math.sqrt(len(ratio))),"noninferior_at_one_se":bool(ratio.mean()+ratio.std(ddof=1)/math.sqrt(len(ratio))<=1)},"submission_ready":False}
 (CANDIDATE/"external_patchtst_audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps(result,sort_keys=True))
if __name__=="__main__":main()
