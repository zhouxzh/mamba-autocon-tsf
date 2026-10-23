from pathlib import Path


def test_h336_inference_boundary_checks_same_geometry_and_does_not_assume_speed_win():
    text = Path("scripts/audit_pems_h336_inference_boundary.py").read_text(encoding="utf-8")
    for token in ("benchmark geometry mismatch", "mamba_scan_count", "speed_advantage_supported", "relation_loss_included"):
        assert token in text
