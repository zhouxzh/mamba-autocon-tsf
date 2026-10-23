from pathlib import Path


def test_strict_transfer_audit_keeps_positive_and_negative_boundaries_together():
    text = Path("scripts/audit_strict_representation_transfer_evidence.py").read_text(encoding="utf-8")
    for token in (
        "pems_10", "pems_25", "metr_10", "weather_10",
        "negative_boundary_retained", "not globally pristine",
        "conditional_scope_visible", "single_scan_boundary_visible",
        "submission_ready", "independent-machine replication",
    ):
        assert token in text
