"""Regression checks for the official-inspired plain MambaTS comparison."""

from pathlib import Path


def test_stride24_plain_mambats_readouts_are_complete():
    accuracy = Path("artifacts/exploratory/mambatsv3_stride24_official_mambats_baseline_h96_readout_20260929/readout.md").read_text(encoding="utf-8")
    resource = Path("artifacts/exploratory/mambatsv3_stride24_official_mambats_baseline_resource_20260929/readout/readout.md").read_text(encoding="utf-8")
    assert "24/24 successful jobs" in accuracy
    assert "| ETTh1 | 0.701401 | 0.694153 | +1.047% | 0/3 |" in accuracy
    assert "| weather | 0.442215 | 0.482679 | -8.376% | 3/3 |" in accuracy
    assert "| electricity | 0.164678 | 0.182222 | -9.625% | 3/3 |" in accuracy
    assert "| traffic | 0.382755 | 0.396599 | -3.490% | 3/3 |" in accuracy
    assert "| ETTh1 | 0.2778 | 0.2177 | +27.6% |" in resource
    assert "| traffic | 0.8383 | 0.7329 | +14.4% |" in resource
