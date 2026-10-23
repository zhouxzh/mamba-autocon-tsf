from pathlib import Path


def test_official_itransformer_adapter_uses_explicit_boundary_contract():
    text = Path("scripts/official_itransformer_adapter.py").read_text(encoding="utf-8")
    for token in ("thuml/iTransformer", "official-root", "official-commit", "train_end", "val_end", "few_end", "selection_done_before_test", "test_metrics_produced", "official model with local chronological adapter", "resolved outside --official-root", "official_model_sha256"):
        assert token in text
    assert "0.7" not in text


def test_official_itransformer_adapter_uses_audited_numeric_array_contract():
    text = Path("scripts/official_itransformer_adapter.py").read_text(encoding="utf-8")
    for token in ("load_numeric_array_file", "--data-format", "--data-key", "--data-feature-index", "REPO_ROOT"):
        assert token in text
