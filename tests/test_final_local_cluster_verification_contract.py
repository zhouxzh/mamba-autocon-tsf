from pathlib import Path


def test_final_verification_runs_only_ordered_derived_audits():
    text = Path("scripts/verify_final_local_cluster_evidence.py").read_text(encoding="utf-8")
    for token in (
        "audit_claim_evidence_integrity.py",
        "audit_manuscript_claim_bindings.py",
        "audit_local_cluster_acf_evidence_package.py",
        "audit_final_backbone_selection.py",
        "audit_local_cluster_submission_readiness.py",
        "audit_final_local_cluster_source_drift.py",
        "build_final_local_cluster_review_manifest.py",
        "review_manifest_sha256",
        "source_hashes_match",
        "derived_audits_only",
        "training_performed",
        "new_test_reader_opened",
    ):
        assert token in text
    assert "run_gpu_guarded.py" not in text
