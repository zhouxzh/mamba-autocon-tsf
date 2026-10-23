#!/usr/bin/env python3
"""Audit the independent-host official PatchTST extension without pooling targets."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COMMIT = "204c21efe0b39603ad6e2ca640ef5896646ab1a9"
MODEL_HASH = "49d8bb865e1226d6338842f4b33c4bc6992cefb048dc2a4a5a8c410ca708586b"
BACKBONE_HASH = "df67173153787c2356bdfb6491159cd754332ef7382986efe879e1fbea8ebf26"
TARGETS = {
    "pems_bay_h192_10": ("PEMS-BAY", (6130, 6131, 6132), ROOT / "artifacts/reproducibility/mambatsv3_local_cluster_independent_machine_20261008"),
    "metr_la_h192_10": ("METR-LA", (6140, 6141, 6142), ROOT / "artifacts/reproducibility/mambatsv3_local_cluster_metr_independent_machine_20261008"),
    "weather_h192_10": ("Weather", (6150, 6151, 6152), ROOT / "artifacts/reproducibility/mambatsv3_local_cluster_weather_independent_machine_20261008"),
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def telemetry(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(row.get("guard_trigger") for row in rows):
        raise ValueError("invalid telemetry: {}".format(path))
    return {"max_temperature_c": max(float(row["temperature.gpu"]) for row in rows), "max_fan_percent": max(float(row["fan.speed"]) for row in rows)}


def summary(active: np.ndarray, baseline: np.ndarray) -> dict:
    ratios = active / baseline
    return {
        "active_mean_test_mse": float(active.mean()), "official_patchtst_mean_test_mse": float(baseline.mean()),
        "active_vs_official_patchtst_percent": float((active.mean() / baseline.mean() - 1) * 100),
        "active_wins": int((active < baseline).sum()),
        "ratio_upper_1se": float(ratios.mean() + ratios.std(ddof=1) / math.sqrt(len(ratios))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--host-evidence", type=Path, required=True)
    parser.add_argument("--verified-independent-machine-id", required=True)
    args = parser.parse_args()
    root, evidence = args.artifact_root.resolve(), load(args.host_evidence)
    manifest, gate = load(root / "manifest.json"), load(root / "validation_gate.json")
    remote = evidence.get("remote", {})
    if manifest.get("machine_id") != args.verified_independent_machine_id or evidence.get("machine_id") != args.verified_independent_machine_id:
        raise ValueError("reviewer machine identifier mismatch")
    if not evidence.get("distinct_from_local_hostname") or not remote.get("cuda_available"):
        raise ValueError("independent-host evidence incomplete")
    if not gate.get("passes") or gate.get("test_metrics_produced"):
        raise ValueError("validation gate mismatch")
    if manifest.get("official_commit") != COMMIT or manifest.get("official_model_sha256") != MODEL_HASH or manifest.get("official_backbone_sha256") != BACKBONE_HASH:
        raise ValueError("official source lock mismatch")
    targets, telemetry_rows = {}, {}
    for key, (target, seeds, active_root) in TARGETS.items():
        rows = []
        for seed in seeds:
            prefix = root / key / "s{}_official_patchtst".format(seed)
            validation, test = load(prefix.with_suffix(".json")), load(prefix.with_name(prefix.name + "_test.json"))
            active = load(active_root / "s{}_active_test.json".format(seed))
            if validation.get("test_metrics_produced") or validation.get("official_commit") != COMMIT:
                raise ValueError("validation lifecycle mismatch: {} {}".format(key, seed))
            if validation.get("official_model_sha256") != MODEL_HASH or validation.get("official_backbone_sha256") != BACKBONE_HASH:
                raise ValueError("official model lock mismatch: {} {}".format(key, seed))
            if not (test.get("test_metrics_produced") and test.get("selection_done_before_test") and not test.get("training_performed")):
                raise ValueError("test lifecycle mismatch: {} {}".format(key, seed))
            if test.get("official_commit") != COMMIT or test.get("official_model_sha256") != MODEL_HASH or test.get("official_backbone_sha256") != BACKBONE_HASH:
                raise ValueError("test source lock mismatch: {} {}".format(key, seed))
            if not (active.get("test_metrics_produced") and not active.get("training_performed") and active.get("selection_done_before_test")):
                raise ValueError("active test lifecycle mismatch: {} {}".format(key, seed))
            for stage in ("validation", "test"):
                event = load(prefix.with_name(prefix.name + "_{}_event.json".format(stage)))
                if event.get("status") != "completed" or event.get("trigger") is not None or event.get("returncode") != 0:
                    raise ValueError("thermal guard failure: {} {} {}".format(key, seed, stage))
                telemetry_rows["{}_s{}_{}".format(key, seed, stage)] = telemetry(prefix.with_name(prefix.name + "_{}_telemetry.csv".format(stage)))
            rows.append({"seed": seed, "active_test_mse": float(active["test_mse"]), "official_patchtst_test_mse": float(test["test_mse"]), "official_validation_mse": float(validation["best_validation_mse"])})
        active_values = np.asarray([row["active_test_mse"] for row in rows])
        baseline_values = np.asarray([row["official_patchtst_test_mse"] for row in rows])
        targets[key] = {"target": target, "rows": rows, "summary": summary(active_values, baseline_values)}
    result = {
        "audit_valid": True, "independent_machine": True,
        "verified_machine_id": args.verified_independent_machine_id,
        "host": {"hostname": remote.get("hostname"), "gpu": remote.get("gpu"), "torch": remote.get("torch"), "mamba_ssm": remote.get("mamba_ssm")},
        "official_repository": "yuqinie98/PatchTST", "official_commit": COMMIT,
        "comparison_scope": "post-release descriptive official-code extension; it does not change frozen active/null mechanism evidence",
        "pooled_effect_reported": False, "targets": targets, "telemetry": telemetry_rows,
        "submission_ready": False,
    }
    (root / "independent_patchtst_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# Independent-Host Official PatchTST Extension", "", "Each target is audited separately; the table is a post-release external-code comparison and is not pooled across domains.", "", "| Target | Active test MSE | Official PatchTST test MSE | Active vs PatchTST | Wins | One-SE upper |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for item in targets.values():
        row = item["summary"]
        lines.append("| {} | {:.6f} | {:.6f} | {:+.3f}% | {}/3 | {:.5f} |".format(item["target"], row["active_mean_test_mse"], row["official_patchtst_mean_test_mse"], row["active_vs_official_patchtst_percent"], row["active_wins"], row["ratio_upper_1se"]))
    lines += ["", "## Boundary", "", "These tests were added after the active/null protocol was frozen. They support reproducibility of an external comparator only and cannot be used to retune or reinterpret the ACF mechanism result."]
    (root / "independent_patchtst_readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
