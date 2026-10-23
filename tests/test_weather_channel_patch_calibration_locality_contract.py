from pathlib import Path


def test_weather_locality_screen_has_explicit_source_and_calibration_patch_fractions():
    text = Path("scripts/run_weather_channel_patch_calibration_locality_probe.py").read_text(encoding="utf-8")
    for token in ("calibration-channel-patch-relation-fraction", "target_calibration_boundary", "mamba_scan_count", "run_gpu_guarded.py", "screen_rule"):
        assert token in text


def test_weather_locality_audit_checks_role_specific_patch_fractions():
    text = Path("scripts/audit_weather_channel_patch_calibration_locality_probe.py").read_text(encoding="utf-8")
    for token in ("target-only locality contract", "source-and-target locality contract", "channel_patch_relation_active_fraction", "screen_passes_expansion_margin"):
        assert token in text
