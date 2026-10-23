from pathlib import Path


def test_weather_target_calibration_screen_is_prefix_only_and_one_scan():
    text = Path("scripts/run_weather_channel_patch_target_calibration_probe.py").read_text(encoding="utf-8")
    for token in ("calibration-source", "calibration-fraction", "target_calibration_boundary", "mamba_scan_count", "run_gpu_guarded.py"):
        assert token in text


def test_weather_target_calibration_audit_rechecks_prefix_and_patch_support():
    text = Path("scripts/audit_weather_channel_patch_target_calibration_probe.py").read_text(encoding="utf-8")
    for token in ("train_prefix_fraction", "channel_patch_relation_active_fraction", "Weather prefix boundary", "screen_passes_expansion_margin"):
        assert token in text
