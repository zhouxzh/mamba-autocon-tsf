#!/usr/bin/env python3
"""Audit a high-dimensional held-out launch packet without launching jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("JSON object expected: {}".format(path))
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan_path = (ROOT / args.plan).resolve() if not args.plan.is_absolute() else args.plan.resolve()
    plan = load(plan_path)
    plan_hash = sha256(plan_path)
    errors: list[str] = []

    if plan.get("status") != "draft_test_release":
        errors.append("plan status is not draft_test_release")
    execution = plan.get("execution")
    if not isinstance(execution, dict) or execution.get("authorized") is not False:
        errors.append("plan must remain execution.authorized=false for this audit")
    counts = plan.get("registered_job_counts", {})
    if counts.get("total") != 86 or counts.get("training_total") != 86:
        errors.append("plan does not register exactly 86 jobs")
    if plan.get("global", {}).get("patch_len") != 24 or plan.get("global", {}).get("patch_stride") != 24:
        errors.append("plan is not the stride-24 candidate")
    required_files = {
        "candidate_validation": ROOT / "artifacts/exploratory/mambatsv3_stride24_full_baseline_h96_validation_20260929/readout/readout.md",
        "resource_attestation": ROOT / "artifacts/exploratory/mambatsv3_stride24_full_baseline_resource_20260929/readout/readout.md",
        "autocon_audit": ROOT / "artifacts/exploratory/mambatsv3_stride24_highdim_autocon_training_audit_20261001/readout/readout.md",
        "path_equivalence": ROOT / "artifacts/exploratory/mambatsv3_stride24_default_path_equivalence_20261001/audit.json",
        "launch_review": ROOT / "docs/research/stride24_highdim_heldout_launch_review_20261001.md",
    }
    for name, path in required_files.items():
        if not path.is_file():
            errors.append("missing {}: {}".format(name, path))

    equivalence = None
    if required_files["path_equivalence"].is_file():
        equivalence = load(required_files["path_equivalence"])
        if equivalence.get("passed") is not True:
            errors.append("default-path equivalence audit did not pass")
    if plan_hash not in required_files["launch_review"].read_text(encoding="utf-8"):
        errors.append("launch review does not bind the exact plan hash")

    payload = {
        "plan": str(plan_path),
        "plan_sha256": plan_hash,
        "authorized": bool(execution.get("authorized")) if isinstance(execution, dict) else None,
        "required_files": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in required_files.items() if path.is_file()
        },
        "default_path_equivalence_passed": None if equivalence is None else equivalence.get("passed"),
        "errors": errors,
        "status": "review_packet_valid" if not errors else "review_packet_invalid",
        "launch_started": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "errors": len(errors), "output": str(args.output)}))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
