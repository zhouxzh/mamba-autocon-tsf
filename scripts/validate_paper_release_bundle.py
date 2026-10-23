#!/usr/bin/env python3
"""Audit the machine-readable provenance of the frozen paper release bundles.

This command is intentionally read-only with respect to ``artifacts/paper_final``.
It produces a compact audit bundle elsewhere and never recomputes test metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


BUNDLES = (
    {
        "name": "PEMS H12",
        "release_dir": "pems_mambatsv3_d24_test_release_20260926",
        "phase_rel": "baselines",
        "summary_rows": 28,
        "test_rows": 160,
        "selection_rows": 12,
        "expected_noninferior_cells": 28,
    },
    {
        "name": "METR-LA/PEMS-BAY H12",
        "release_dir": "metro_pemsbay_mambatsv3_repr_fusion_h12_test_release_20260927",
        "phase_rel": "baselines",
        "summary_rows": 12,
        "test_rows": 70,
        "selection_rows": 6,
        "expected_noninferior_cells": 12,
    },
    {
        "name": "Weather H96/H336/H720",
        "release_dir": "mambatsv3_weather_test_release_20260922",
        "phase_rel": "weather_confirmatory_strong_baselines_test",
        "summary_rows": 12,
        "test_rows": 75,
        "selection_rows": None,
        "expected_noninferior_cells": 12,
    },
    {
        "name": "Electricity/Traffic H96 CPU-4",
        "release_dir": "mambatsv3_stride24_relation_auxiliary_highdim_cpu4_test_release_20261001",
        "phase_rel": "baselines",
        "summary_rows": 14,
        "test_rows": 80,
        "selection_rows": 6,
        # The frozen release deliberately retains the Traffic/iTransformer
        # failure rather than treating it as an artifact-validation failure.
        "expected_noninferior_cells": 13,
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve_plan(repo_root: Path, phase: dict[str, Any], auth: dict[str, Any]) -> Path:
    raw = phase.get("plan") or auth.get("plan_path")
    if not raw:
        raise ValueError("missing plan path in phase validation and authorization")
    candidate = Path(str(raw))
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve()


def audit_bundle(repo_root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    release = repo_root / "artifacts" / "paper_final" / spec["release_dir"]
    phase_dir = release / spec["phase_rel"]
    readout = release / "readout"
    phase_path = phase_dir / "phase_validation.json"
    auth_path = phase_dir / "provenance" / "launch_authorization.json"
    summary_path = readout / "paired_test_summary.csv"
    test_rows_path = readout / "test_rows.csv"
    readout_md = readout / "readout.md"

    required = [phase_path, auth_path, summary_path, test_rows_path, readout_md]
    missing = [str(path.relative_to(repo_root)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{spec['name']} missing required files: {missing}")

    phase = load_json(phase_path)
    auth = load_json(auth_path)
    plan_path = resolve_plan(repo_root, phase, auth)
    if not plan_path.is_file():
        raise FileNotFoundError(f"plan does not exist: {plan_path}")
    plan_hash = sha256_file(plan_path)

    summary_rows = read_csv(summary_path)
    test_rows = read_csv(test_rows_path)
    selection_path = readout / "selection_rows.csv"
    selection_rows = read_csv(selection_path) if selection_path.is_file() else None
    checks = {
        "phase_valid": phase.get("valid") is True,
        "phase_errors_empty": not phase.get("errors"),
        "manifest_jobs_match": phase.get("expected_jobs") == phase.get("manifest_unique_jobs"),
        "authorization_true": auth.get("authorized") is True,
        "authorization_plan_status": auth.get("plan_status") == "launch_authorized",
        "phase_plan_hash_matches": phase.get("plan_sha256") == plan_hash,
        "authorization_plan_hash_matches": auth.get("plan_sha256") == plan_hash,
        "summary_row_count": len(summary_rows) == spec["summary_rows"],
        "test_row_count": len(test_rows) == spec["test_rows"],
        "selection_row_count": (
            selection_rows is None
            if spec["selection_rows"] is None
            else selection_rows is not None and len(selection_rows) == spec["selection_rows"]
        ),
        "noninferior_cell_count": (
            sum(row.get("noninferior_ratio_1p02", "").lower() == "true" for row in summary_rows)
            == spec["expected_noninferior_cells"]
        ),
        "summary_numeric_complete": all(
            row.get("candidate_mse_mean") not in (None, "")
            and row.get("baseline_mse_mean") not in (None, "")
            and row.get("mean_mse_change_pct") not in (None, "")
            for row in summary_rows
        ),
    }
    passed = all(checks.values())
    file_hashes = {
        str(path.relative_to(repo_root)): sha256_file(path)
        for path in (phase_path, auth_path, plan_path, summary_path, test_rows_path, readout_md)
    }
    if selection_path.is_file():
        file_hashes[str(selection_path.relative_to(repo_root))] = sha256_file(selection_path)

    return {
        "name": spec["name"],
        "release_dir": str(release.relative_to(repo_root)),
        "phase": phase.get("phase"),
        "plan": str(plan_path.relative_to(repo_root)),
        "plan_sha256": plan_hash,
        "expected_jobs": phase.get("expected_jobs"),
        "manifest_unique_jobs": phase.get("manifest_unique_jobs"),
        "summary_rows": len(summary_rows),
        "test_rows": len(test_rows),
        "selection_rows": None if selection_rows is None else len(selection_rows),
        "noninferior_cells": sum(row.get("noninferior_ratio_1p02", "").lower() == "true" for row in summary_rows),
        "summary_cells": len(summary_rows),
        "candidate_wins_total": sum(int(row["candidate_wins"]) for row in summary_rows),
        "checks": checks,
        "passed": passed,
        "file_sha256": file_hashes,
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Frozen Paper Release Bundle Audit",
        "",
        "This is a read-only audit of existing held-out release artifacts. It does not rerun training or recompute test metrics.",
        "",
        "| Release | Summary cells | Test rows | Non-inferior cells | Candidate wins | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in audit["bundles"]:
        lines.append(
            f"| {item['name']} | {item['summary_cells']} | {item['test_rows']} | "
            f"{item['noninferior_cells']}/{item['summary_cells']} | {item['candidate_wins_total']} | "
            f"{'PASS' if item['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"Overall status: **{'PASS' if audit['passed'] else 'FAIL'}**.",
            "",
            "The scope is limited to these four frozen releases and their named comparators; it is not a universal claim over all datasets, horizons, or model families. The CPU-4 Electricity/Traffic bundle intentionally records one non-inferiority failure.",
            "",
            "Detailed hashes and individual checks are in `audit.json`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/exploratory/paper_release_bundle_audit_20260928"),
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    bundles = [audit_bundle(repo_root, spec) for spec in BUNDLES]
    audit = {"schema_version": 1, "passed": all(item["passed"] for item in bundles), "bundles": bundles}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (output_dir / "audit.md").write_text(render_markdown(audit), encoding="utf-8")
    print(json.dumps({"passed": audit["passed"], "output_dir": str(output_dir), "bundles": len(bundles)}))
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
