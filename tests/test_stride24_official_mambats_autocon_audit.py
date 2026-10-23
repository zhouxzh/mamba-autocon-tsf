from pathlib import Path


def test_stride24_official_mambats_autocon_audit_is_complete():
    readout = Path("artifacts/exploratory/mambatsv3_stride24_official_mambats_autocon_training_audit_20260929/readout.md").read_text(encoding="utf-8")
    assert "12 candidate validation jobs" in readout
    assert readout.count("3 |") == 4
    assert "9.65e-10" in readout
