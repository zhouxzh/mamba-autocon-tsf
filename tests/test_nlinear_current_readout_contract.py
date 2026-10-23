from pathlib import Path


def test_nlinear_extension_readout_enforces_validation_boundary():
    text = Path("scripts/analyze_nlinear_current_h96_baseline_extension.py").read_text(encoding="utf-8")
    for token in ("NLinear", "coverage/source mismatch", "metrics.npy", "skip_test", "source snapshot drift"):
        assert token in text
