from pathlib import Path


def test_weather_strong_calibration_screen_has_fixed_prefix_and_relation_schedule():
    text = Path("scripts/run_weather_channel_patch_calibration_strength_probe.py").read_text(encoding="utf-8")
    for token in ("calibration-relation-loss-fraction", "target_calibration_boundary", "mamba_scan_count", "run_gpu_guarded.py", "screen_rule"):
        assert token in text


def test_weather_strong_calibration_audit_checks_prefix_and_active_patch_loss():
    text = Path("scripts/audit_weather_channel_patch_calibration_strength_probe.py").read_text(encoding="utf-8")
    for token in ("train_prefix_fraction", "relation_loss_fraction", "channel_patch_relation_active_fraction", "screen_passes_expansion_margin"):
        assert token in text
