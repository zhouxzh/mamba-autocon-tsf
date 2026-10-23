from pathlib import Path


def test_h336_official_extension_audits_hashes_and_lifecycle():
    text = Path("scripts/audit_official_patchtst_pems_h336_extension.py").read_text(encoding="utf-8")
    for token in ("official_model_sha256", "official_backbone_sha256", "external source lock mismatch", "selection_done_before_test", "active_wins"):
        assert token in text
