from pathlib import Path


def test_pems_25_official_itransformer_extension_binds_source_and_lifecycle():
    text = Path("scripts/audit_official_itransformer_pems_25_extension.py").read_text(encoding="utf-8")
    for token in ("official_attention_sha256", "official_encoder_sha256", "external source lock mismatch", "selection_done_before_test", "active_wins"):
        assert token in text
