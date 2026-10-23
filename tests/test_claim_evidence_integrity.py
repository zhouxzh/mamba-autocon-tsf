from pathlib import Path

from scripts.audit_claim_evidence_integrity import audit


def test_claim_evidence_integrity_for_internal_manuscript():
    root = Path(__file__).resolve().parents[1]
    report = audit(
        root / "docs/research/manuscript_draft_20260928.md",
        root / "docs/research/claim_evidence_registry_20261001.md",
        root / "docs/research/manuscript_draft_20260928.md",
        root,
    )
    assert report["passed"], report
    assert report["used_claim_ids"]
    assert report["used_evidence_ids"]
