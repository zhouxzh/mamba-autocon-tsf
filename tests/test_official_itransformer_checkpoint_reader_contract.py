from pathlib import Path


def test_official_itransformer_reader_is_hash_bound_and_read_only():
    text = Path("scripts/evaluate_official_itransformer_checkpoint_test.py").read_text(encoding="utf-8")
    for token in ("dataset hash does not match", "source hash", "selection_done_before_test", "training_performed", "official_model_sha256"):
        assert token in text
    assert "optimizer" not in text
