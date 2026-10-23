from pathlib import Path


def test_weather_failed_gate_audit_requires_closed_test_partition():
    text = Path("scripts/audit_mambatsv3_acf_infonce_weather_25_gate.py").read_text(encoding="utf-8")
    for token in ("all-seed gate", "test_metrics_produced", "*_test.json", "guard failure"):
        assert token in text
