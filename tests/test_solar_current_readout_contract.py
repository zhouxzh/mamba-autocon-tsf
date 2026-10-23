from pathlib import Path


def test_solar_readout_enforces_complete_validation_only_matrix():
    text = Path("scripts/analyze_solar_current_strong_baseline_extension.py").read_text(encoding="utf-8")
    for token in ("DATASET = \"solar\"", "coverage/source mismatch", "metrics.npy", "data_interval_minutes"):
        assert token in text
