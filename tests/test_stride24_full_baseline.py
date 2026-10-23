"""Regression checks for the complete stride-24 baseline evidence."""

from pathlib import Path


def test_stride24_full_accuracy_and_resource_readouts_are_complete():
    accuracy = Path("artifacts/exploratory/mambatsv3_stride24_full_baseline_h96_validation_20260929/readout/readout.md").read_text(encoding="utf-8")
    resource = Path("artifacts/exploratory/mambatsv3_stride24_full_baseline_resource_20260929/readout/readout.md").read_text(encoding="utf-8")
    assert "42/42 successful jobs" in accuracy
    assert accuracy.count("3/3") == 12
    assert "TimesNet" in accuracy and "Mamba2" in accuracy
    assert "faster than PatchTST, Mamba1, TimesNet, Mamba2" in resource
    assert "faster than PatchTST, iTransformer, TimesNet, Mamba2" in resource
    assert "slower than DLinear" in resource
