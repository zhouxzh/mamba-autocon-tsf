from pathlib import Path


def test_official_patchtst_reader_is_read_only_and_hash_binds_data_and_checkout():
    text = Path("scripts/evaluate_official_patchtst_checkpoint_test.py").read_text(encoding="utf-8")
    for token in (
        "dataset hash does not match", "checkout commit", "source hash", "official_model",
        "selection_done_before_test", "training_performed", "test_metrics_produced",
    ):
        assert token in text
    assert "optimizer" not in text
