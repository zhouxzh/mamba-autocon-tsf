from pathlib import Path


def test_weather_calibrated_robustness_is_frozen_checkpoint_validation_only():
    text = Path("scripts/run_weather_channel_patch_calibration_robustness.py").read_text(encoding="utf-8")
    for token in ("validation_only", "test_metrics_produced", "same_seed_pairing", "run_gpu_guarded.py", "evaluate_mambatsv3_lowdata_robustness.py"):
        assert token in text


def test_weather_calibrated_robustness_audit_checks_all_conditions_and_external_gate():
    text = Path("scripts/audit_weather_channel_patch_calibration_robustness.py").read_text(encoding="utf-8")
    for token in ("missing_channels_30", "noise_0p10", "condition mismatch", "all_external_noninferior"):
        assert token in text
