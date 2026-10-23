#!/usr/bin/env python3
"""Audit frozen METR-LA H192 10% active/PatchTST test release."""
import hashlib, json, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "artifacts/paper_final/mambatsv3_mamba1_metr10_frozen_test_release_20261006"
SEEDS = (5970, 5971, 5972)
COMMIT = "204c21efe0b39603ad6e2ca640ef5896646ab1a9"

def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()

def main():
    manifest = load(RELEASE / "manifest_before_test.json")
    if manifest.get("status") != "frozen_before_test" or tuple(manifest.get("seeds", ())) != SEEDS: raise ValueError("manifest lifecycle mismatch")
    candidates = {x["seed"]: x for x in manifest["candidate"]}; baselines = {x["seed"]: x for x in manifest["official_patchtst"]}
    rows = []
    for seed in SEEDS:
        active, patch = load(RELEASE / f"s{seed}_active_test.json"), load(RELEASE / f"s{seed}_official_patchtst_test.json")
        if not (active.get("test_metrics_produced") and patch.get("test_metrics_produced")) or active.get("training_performed") or patch.get("training_performed"): raise ValueError(f"reader lifecycle mismatch: {seed}")
        if not (active.get("selection_done_before_test") and patch.get("selection_done_before_test")): raise ValueError(f"selection mismatch: {seed}")
        if digest(active["checkpoint"]) != candidates[seed]["sha256"] or digest(patch["checkpoint"]) != baselines[seed]["sha256"] or patch.get("official_commit") != COMMIT: raise ValueError(f"frozen artifact mismatch: {seed}")
        for name in (f"s{seed}_active_event.json", f"s{seed}_official_patchtst_event.json"):
            event = load(RELEASE / name)
            if event.get("status") != "completed" or event.get("trigger") is not None or event.get("returncode") != 0: raise ValueError(f"guard failure: {name}")
        a, p = float(active["test_mse"]), float(patch["test_mse"])
        rows.append({"seed": seed, "active_test_mse": a, "official_patchtst_test_mse": p, "ratio": a / p, "active_better": a < p})
    a, p = np.asarray([r["active_test_mse"] for r in rows]), np.asarray([r["official_patchtst_test_mse"] for r in rows]); ratios = a / p
    result = {"audit_valid": True, "rows": rows, "active_mean_test_mse": float(a.mean()), "official_patchtst_mean_test_mse": float(p.mean()), "active_vs_patchtst_percent": float((a.mean()/p.mean()-1)*100), "active_wins": int((a<p).sum()), "ratio_upper_1se": float(ratios.mean()+ratios.std(ddof=1)/math.sqrt(len(ratios))), "noninferior_at_one_se": bool(ratios.mean()+ratios.std(ddof=1)/math.sqrt(len(ratios)) <= 1), "historical_public_test_access_caveat": bool(manifest.get("historical_public_test_access_caveat")), "submission_ready": False}
    (RELEASE / "audit.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8"); print(json.dumps(result, sort_keys=True))

if __name__ == "__main__": main()
