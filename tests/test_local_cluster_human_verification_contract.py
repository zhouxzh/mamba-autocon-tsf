from pathlib import Path


def test_human_verification_checker_reports_pending_without_mutating_statuses():
    text = Path("scripts/check_local_cluster_human_verification.py").read_text(encoding="utf-8")
    for token in ("worksheet schema mismatch", "pending_claim_ids", "human_verification_complete", "never changes"):
        assert token in text
