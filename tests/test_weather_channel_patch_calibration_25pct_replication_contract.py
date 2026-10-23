from pathlib import Path


def test_weather_calibrated_25pct_replication_is_registered_prefix_only():
    text = Path("scripts/run_weather_channel_patch_calibration_25pct_replication.py").read_text(encoding="utf-8")
    for token in ("SEEDS = (6076, 6077, 6078)", "calibration-fraction", "calibration-channel-patch-relation-fraction", "target_calibration_boundary", "promotion_rule"):
        assert token in text


def test_weather_calibrated_25pct_audit_checks_budget_prefix_and_active_patch():
    text = Path("scripts/audit_weather_channel_patch_calibration_25pct_replication.py").read_text(encoding="utf-8")
    for token in ("label budget mismatch", "train_prefix_fraction", "channel_patch_relation_active_fraction", "promotion_eligible"):
        assert token in text
