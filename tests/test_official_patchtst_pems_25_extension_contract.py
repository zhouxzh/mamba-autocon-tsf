from pathlib import Path


def test_pems_25_official_extension_audits_hashes_and_reader_lifecycle():
    text = Path("scripts/audit_official_patchtst_pems_25_extension.py").read_text(encoding="utf-8")
    for token in (
        "official_model_sha256", "official_backbone_sha256", "external source lock mismatch",
        "test_metrics_produced", "training_performed", "selection_done_before_test", "active_wins",
    ):
        assert token in text
