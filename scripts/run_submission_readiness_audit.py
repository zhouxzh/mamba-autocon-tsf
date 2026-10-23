#!/usr/bin/env python3
"""Run the local read-only submission-readiness checks as one report."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def run(command: list[str], root: Path) -> tuple[int, str]:
    completed = subprocess.run(command, cwd=root, text=True, capture_output=True)
    return completed.returncode, (completed.stdout + completed.stderr).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/exploratory/submission_readiness_audit_20261001"))
    parser.add_argument("--regression-log", type=Path, default=None, help="completed pytest output; this script never starts pytest")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    env = ["conda", "run", "--no-capture-output", "-n", "mamba", "python"]
    checks: dict[str, dict] = {}

    commands = {
        "release_bundle": env + ["scripts/validate_paper_release_bundle.py", "--output-dir", str(output / "release_bundle")],
        "dcrnn_numbers": env + ["scripts/audit_dcrnn_evidence_consistency.py", "--summary", "artifacts/exploratory/metro_pemsbay_dcrnn_mambatsv3_h12_readout_20260930/paired_summary.csv", "--resources", "artifacts/exploratory/metro_pemsbay_dcrnn_mambatsv3_h12_resource_attestation_20260930/resource_rows.csv", "--manuscript", "docs/research/manuscript_draft_20260928.md", "--output", str(output / "dcrnn_numbers.json")],
        "claim_evidence": env + ["scripts/audit_claim_evidence_integrity.py", "--manuscript", "docs/research/manuscript_draft_20260928.md", "--claims", "docs/research/claim_evidence_registry_20261001.md", "--evidence-registry", "docs/research/manuscript_draft_20260928.md", "--repo-root", ".", "--output", str(output / "claim_evidence.json")],
        "manuscript_bindings": env + ["scripts/audit_manuscript_claim_bindings.py", "--manuscript", "docs/research/manuscript_draft_20260928.md", "--output", str(output / "manuscript_bindings.json")],
        "citations": env + [str(Path.home() / ".agents/skills/citation-management/scripts/validate_citations.py"), "docs/research/references/mamba_autocon_references.bib", "--report", str(output / "citations.json")],
    }
    for name, command in commands.items():
        code, text = run(command, root)
        payload = {"returncode": code, "output_tail": text[-2000:]}
        target = output / f"{name}_result.json"
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        checks[name] = {"passed": code == 0, **payload}

    regression = {"passed": False, "note": "Run the full pytest suite separately; this audit does not launch a long test process."}
    if args.regression_log is not None:
        regression_text = args.regression_log.read_text(encoding="utf-8")
        match = re.search(r"(?m)^(\d+) passed, (\d+) warnings in ([0-9.]+)s$", regression_text)
        regression = {"passed": bool(match), "log": str(args.regression_log), "summary": match.group(0) if match else None}
    checks["regression_suite"] = regression
    report = {
        "schema_version": 1,
        "status": "review_required" if not all(item["passed"] for item in checks.values() if "passed" in item) else "checks_passed_human_approval_required",
        "checks": checks,
        "scope": "read-only local release, claim, citation and manuscript audits; no training or held-out test execution",
    }
    (output / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": len(checks)}))
    return 0 if report["status"] == "checks_passed_human_approval_required" else 1


if __name__ == "__main__":
    raise SystemExit(main())
