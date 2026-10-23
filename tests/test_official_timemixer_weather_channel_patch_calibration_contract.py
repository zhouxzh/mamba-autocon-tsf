from pathlib import Path


def test_weather_calibrated_timemixer_comparator_is_validation_only_and_same_seed():
    text = Path("scripts/run_official_timemixer_weather_channel_patch_calibration.py").read_text(encoding="utf-8")
    for token in ("validation_only", "test_metrics_produced", "same_seed_pairing", "run_gpu_guarded.py", "--evaluate-test"):
        assert token in text


def test_weather_calibrated_timemixer_audit_checks_commit_and_boundary():
    text = Path("scripts/audit_official_timemixer_weather_channel_patch_calibration.py").read_text(encoding="utf-8")
    for token in ("official_commit", "test boundary", "noninferior_at_one_se", "TimeMixer contract failure"):
        assert token in text
