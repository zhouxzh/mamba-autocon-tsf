from pathlib import Path


def test_weather_calibrated_patchtst_comparator_is_same_seed_validation_only():
    text = Path("scripts/run_official_patchtst_weather_channel_patch_calibration.py").read_text(encoding="utf-8")
    for token in ("same_seed_pairing", "no_test_reader", "run_gpu_guarded.py", "--evaluate-test"):
        assert token in text


def test_weather_calibrated_patchtst_audit_has_promotion_gate():
    text = Path("scripts/audit_official_patchtst_weather_channel_patch_calibration.py").read_text(encoding="utf-8")
    for token in ("official_commit", "noninferior_at_one_se", "promotion_eligible", "test boundary"):
        assert token in text
