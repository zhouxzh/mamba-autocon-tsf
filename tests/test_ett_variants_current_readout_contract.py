from pathlib import Path


def test_ett_variants_readout_enforces_complete_validation_only_matrix():
    text = Path("scripts/analyze_ett_variants_current_strong_baseline_extension.py").read_text(encoding="utf-8")
    for token in ("ETTh2", "ETTm2", "coverage/source mismatch", "metrics.npy", "skip_test"):
        assert token in text
