#!/usr/bin/env python3
"""Produce a fail-closed technical submission-readiness record for local-cluster evidence."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "artifacts/paper_final/mambatsv3_local_cluster_acf_evidence_package_20261003"
PROGRAM = ROOT / "artifacts/reproducibility/mambatsv3_local_cluster_independent_replication_program_20261008/audit.json"
OUTPUT = PACKAGE / "submission_readiness_20261008.json"
BACKBONE = PACKAGE / "backbone_selection_audit_20261009.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    package = load(PACKAGE / "package_audit.json")
    human = load(PACKAGE / "human_verification_status.json")
    program = load(PROGRAM)
    backbone = load(BACKBONE)
    checks = {
        "technical_evidence_package_valid": package.get("valid") is True,
        "six_local_confirmations_retained": package.get("checks", {}).get("six_heldout_confirmations") is True,
        "failed_boundaries_retained": package.get("checks", {}).get("failed_gates_closed") is True,
        "independent_three_domain_program_valid": (
            program.get("audit_valid") is True
            and program.get("replication_count") == 3
            and program.get("target_count") == 3
            and program.get("all_domains_improve") is True
            and program.get("all_domains_three_seed_wins") is True
            and program.get("all_domains_one_se_noninferior") is True
            and program.get("pooled_effect_reported") is False
        ),
        "claim_evidence_bindings_valid": package.get("checks", {}).get("claim_ids_and_files_bound") is True,
        "numeric_bindings_valid": package.get("checks", {}).get("numeric_prose_bound") is True,
        "backbone_selection_audit_valid": (
            backbone.get("audit_valid") is True
            and backbone.get("selected_backbone", {}).get("mixer") == "mamba1"
            and backbone.get("secondary_ablation", {}).get("status", "").startswith("validation-only")
        ),
        "human_review_still_required": human.get("human_verification_complete") is False,
        "human_pending_claim_count": len(human.get("pending_claim_ids", [])),
    }
    if checks["human_pending_claim_count"] != 9:
        raise ValueError("unexpected human-verification worksheet state")
    result = {
        "technical_evidence_ready": all(value is True for key, value in checks.items() if key not in {"human_review_still_required", "human_pending_claim_count"}),
        "submission_ready": False,
        "checks": checks,
        "remaining_human_gates": [
            "accountable author verification of all nine claim/evidence worksheet rows",
            "target venue selection and current policy review",
            "author list, CRediT, funding, conflicts, ethics, data/code availability, and AI-use declarations",
        ],
        "scope_boundary": "The evidence supports conditional low-label H192 local-cluster ACF-InfoNCE transfer on the named periodic sensor targets; it is not a universal forecasting claim.",
        "no_automatic_release": "This audit never upgrades submission_ready or verification_status because those require accountable human review.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
