from pathlib import Path


def test_official_patchtst_adapter_owns_only_split_and_execution_boundaries():
    text = Path("scripts/official_patchtst_adapter.py").read_text(encoding="utf-8")
    for token in (
        "yuqinie98/PatchTST", "official-root", "official-commit",
        "train_end", "val_end", "few_end", "selection_done_before_test",
        "test_metrics_produced", "official model with local chronological adapter",
        "official_root.resolve().parents", "resolved outside --official-root",
    ):
        assert token in text
    assert "0.7" not in text


def test_official_patchtst_adapter_uses_the_audited_numeric_array_loader():
    text = Path("scripts/official_patchtst_adapter.py").read_text(encoding="utf-8")
    for token in ("load_numeric_array_file", "--data-format", "--data-key", "--data-feature-index", "REPO_ROOT"):
        assert token in text


def test_external_protocol_records_upstream_split_incompatibility():
    text = Path("docs/research/mambatsv3_acf_infonce_external_baseline_protocol_20261003.md").read_text(encoding="utf-8")
    for token in ("204c21efe0b39603ad6e2ca640ef5896646ab1a9", "70/10/20", "adapter", "single test read"):
        assert token in text


def test_patchtst_robustness_uses_the_same_array_contract_as_training():
    text = Path("scripts/evaluate_official_patchtst_robustness.py").read_text(encoding="utf-8")
    assert "load_numeric_array_file" in text
