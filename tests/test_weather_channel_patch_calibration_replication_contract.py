from pathlib import Path


def test_weather_calibrated_replication_is_registered_prefix_only_and_guarded():
    text = Path("scripts/run_weather_channel_patch_calibration_replication.py").read_text(encoding="utf-8")
    for token in ("SEEDS = (6073, 6074, 6075)", "calibration-channel-patch-relation-fraction", "target_calibration_boundary", "promotion_rule", "run_gpu_guarded.py"):
        assert token in text


def test_weather_calibrated_replication_audit_requires_active_patch_and_all_seed_gate():
    text = Path("scripts/audit_weather_channel_patch_calibration_replication.py").read_text(encoding="utf-8")
    for token in ("channel_patch_relation_active_fraction", '["wins"] == 3', "promotion_eligible", "Weather calibration prefix"):
        assert token in text
