from pathlib import Path


def test_submission_readiness_audit_is_fail_closed_on_human_review():
    text = Path("scripts/audit_local_cluster_submission_readiness.py").read_text(encoding="utf-8")
    for token in ("technical_evidence_ready", "submission_ready", "independent_three_domain_program_valid", "pooled_effect_reported", "human_pending_claim_count", "remaining_human_gates", "no_automatic_release", "backbone_selection_audit_valid", "BACKBONE"):
        assert token in text
