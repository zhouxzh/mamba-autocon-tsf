#!/usr/bin/env python3
"""Run the final local-cluster evidence audits in dependency order.

This is deliberately read-only with respect to model/data results: the called
auditors only regenerate derived audit records from existing artifacts.  It
never launches training, opens a new test reader, or selects a checkpoint.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "artifacts/paper_final/mambatsv3_local_cluster_acf_evidence_package_20261003"
RESULTS = ROOT / "docs/research/mambatsv3_acf_infonce_local_cluster_results_draft_20261003.md"
CLAIMS = ROOT / "docs/research/mambatsv3_acf_infonce_local_cluster_claim_registry_20261003.md"
EVIDENCE = ROOT / "docs/research/mambatsv3_acf_infonce_local_cluster_evidence_registry_20261003.md"


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    python = sys.executable
    # Claim/numeric audits must precede the evidence package; the backbone audit
    # then consumes the package audit, and submission readiness is last.
    run([python, "scripts/audit_claim_evidence_integrity.py", "--manuscript", str(RESULTS),
         "--claims", str(CLAIMS), "--evidence-registry", str(EVIDENCE), "--repo-root", str(ROOT),
         "--output", str(PACKAGE / "claim_binding_audit.json")])
    run([python, "scripts/audit_manuscript_claim_bindings.py", "--manuscript", str(RESULTS),
         "--output", str(PACKAGE / "numeric_claim_binding_audit.json")])
    run([python, "scripts/audit_local_cluster_acf_evidence_package.py"])
    run([python, "scripts/audit_final_backbone_selection.py"])
    run([python, "scripts/audit_local_cluster_submission_readiness.py"])
    run([python, "scripts/audit_final_local_cluster_source_drift.py"])
    run([python, "scripts/build_final_local_cluster_review_manifest.py"])
    package = load(PACKAGE / "package_audit.json")
    backbone = load(PACKAGE / "backbone_selection_audit_20261009.json")
    readiness = load(PACKAGE / "submission_readiness_20261008.json")
    source_drift = load(PACKAGE / "source_drift_audit_20261009.json")
    review_manifest = load(PACKAGE / "review_manifest_20261009.json")
    result = {
        "verification_valid": package.get("valid") is True and backbone.get("audit_valid") is True and source_drift.get("audit_valid") is True and review_manifest.get("valid") is True,
        "technical_evidence_ready": readiness.get("technical_evidence_ready") is True,
        "submission_ready": readiness.get("submission_ready") is True,
        "result_mutation": "derived_audits_only",
        "training_performed": False,
        "new_test_reader_opened": False,
        "selected_mixer": backbone.get("selected_backbone", {}).get("mixer"),
        "human_pending_claim_count": readiness.get("checks", {}).get("human_pending_claim_count"),
        "review_manifest_sha256": review_manifest.get("manifest_content_sha256"),
        "source_hashes_match": source_drift.get("source_hashes_match"),
    }
    (PACKAGE / "final_verification_20261009.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verification_valid"] and result["technical_evidence_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
