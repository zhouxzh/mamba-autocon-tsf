from pathlib import Path


def test_ramamba_lite_readout_enforces_null_control_and_validation_boundary():
    text = Path("scripts/analyze_ramamba_lite_h96_screen.py").read_text(encoding="utf-8")
    for token in (
        '"RaMambaLiteAutoCon_null"',
        '"RaMambaLiteAutoCon_active"',
        "expected_lambda == 0.0",
        "cross-phase source snapshot mismatch",
        "metrics.npy",
        "active_vs_null_percent",
    ):
        assert token in text
