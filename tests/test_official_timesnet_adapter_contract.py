from pathlib import Path


def test_official_timesnet_adapter_uses_explicit_boundary_contract():
    text = Path("scripts/official_timesnet_adapter.py").read_text(encoding="utf-8")
    for token in ("thuml/Time-Series-Library", "official-root", "official-commit", "train_end", "val_end", "few_end", "selection_done_before_test", "test_metrics_produced", "official model with local chronological adapter"):
        assert token in text
    assert "0.7" not in text
