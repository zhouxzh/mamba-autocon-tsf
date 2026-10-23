"""Regression checks for stride-24 cross-domain validation evidence."""

from pathlib import Path


def test_stride24_crossdomain_accuracy_and_resource_readouts_are_complete():
    accuracy = Path("artifacts/exploratory/mambatsv3_stride24_crossdomain_h96_readout_20260929/readout.md").read_text(encoding="utf-8")
    resource = Path("artifacts/exploratory/mambatsv3_stride24_crossdomain_resource_20260929/readout/readout.md").read_text(encoding="utf-8")
    assert "48/48 successful jobs" in accuracy
    assert accuracy.count("3/3") == 10
    assert "| ETTh1 | 96 | iTransformer | 0.699699 | 0.683140 | +2.425% | 0/3 |" in accuracy
    assert "| weather | 96 | TimesNet | 0.444975 | 0.463439 | -3.977% | 3/3 |" in accuracy
    assert "| ETTh1 | 0.2817 | 0.0957 | 0.0660 | 0.2676 | 0.1560 | 0.4186 | 2.8086 | 0.7066 |" in resource
    assert "| weather | 0.2755 | 0.0913 | 0.0644 | 0.2784 | 0.1553 | 0.4267 | 2.9053 | 0.7100 |" in resource
