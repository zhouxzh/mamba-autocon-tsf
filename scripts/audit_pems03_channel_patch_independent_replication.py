#!/usr/bin/env python3
"""Validate an independent-host PEMS03 Channel-Patch replication artifact."""
from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import numpy as np


SEEDS = (6080, 6081, 6082)


def load(path: Path): return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--verified-independent-machine-id", required=True,
                        help="reviewer-confirmed machine identifier; required to mark independence")
    parser.add_argument("--host-evidence", type=Path, default=None,
                        help="optional read-only remote host evidence collected before auditing")
    args = parser.parse_args(); root = args.artifact_root.resolve(); manifest = load(root / "manifest.json")
    if manifest.get("machine_id") != args.verified_independent_machine_id:
        raise ValueError("reviewer machine identifier does not match manifest")
    if not manifest.get("target_excluded_from_source_and_initialization") or manifest.get("test_metrics_produced"):
        raise ValueError("protocol boundary failure")
    hashes = manifest.get("source_hashes", {})
    if len(hashes) < 12 or any(len(value) != 64 for value in hashes.values()):
        raise ValueError("source hash manifest is incomplete")
    rows = []
    for seed in SEEDS:
        forecast_pre = load(root / f"s{seed}_forecast_pretrain.json"); patch_pre = load(root / f"s{seed}_channel_patch_pretrain.json")
        forecast = load(root / f"s{seed}_forecast_frozen_core.json"); patch = load(root / f"s{seed}_channel_patch_frozen_core.json")
        if any(item.get("test_metrics_produced") for item in (forecast_pre, patch_pre, forecast, patch)):
            raise ValueError("test boundary")
        if not all(item.get("channel_patch_relation_active_fraction", 0) > 0 for item in patch_pre["history"]):
            raise ValueError("inactive channel-patch anchors")
        for arm in ("forecast", "channel_patch"):
            for stage in ("pretrain", "finetune"):
                event = load(root / f"s{seed}_{arm}_{stage}_event.json")
                if event.get("status") != "completed" or event.get("trigger") is not None:
                    raise ValueError("thermal guard or command failure")
        rows.append({"seed": seed, "forecast": float(forecast["best_validation_mse"]), "channel_patch": float(patch["best_validation_mse"])})
    forecast = np.asarray([row["forecast"] for row in rows]); patch = np.asarray([row["channel_patch"] for row in rows]); ratio = patch / forecast
    host_verified = None
    if args.host_evidence is not None:
        evidence = load(args.host_evidence)
        remote = evidence.get("remote", {})
        if remote.get("hostname") != manifest.get("hostname") or not evidence.get("distinct_from_local_hostname"):
            raise ValueError("host-evidence independence mismatch")
        if not remote.get("cuda_available") or not remote.get("gpu"):
            raise ValueError("host-evidence accelerator mismatch")
        host_verified = {"hostname": remote["hostname"], "observed_machine_id": evidence["machine_id"], "gpu": remote["gpu"], "torch": remote["torch"], "mamba_ssm": remote["mamba_ssm"], "different_from_audit_host": remote["hostname"] != platform.node()}
    result = {"independent_machine": True, "verified_machine_id": args.verified_independent_machine_id, "host_evidence_verified": host_verified, "validation_only": True, "test_metrics_produced": False, "rows": rows, "channel_patch_vs_forecast": {"channel_patch_mean_mse": float(patch.mean()), "forecast_mean_mse": float(forecast.mean()), "percent": float((patch.mean() / forecast.mean() - 1) * 100), "wins": int((patch < forecast).sum()), "ratio_upper_1se": float(ratio.mean() + ratio.std(ddof=1) / math.sqrt(len(ratio))), "noninferior_at_one_se": bool(ratio.mean() + ratio.std(ddof=1) / math.sqrt(len(ratio)) <= 1)}, "audit_valid": True, "submission_ready": False}
    (root / "independent_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
