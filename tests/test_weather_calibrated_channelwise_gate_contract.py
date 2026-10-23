from pathlib import Path


def test_weather_calibrated_channelwise_gate_is_one_scan_validation_only_screen():
    text = Path("scripts/run_weather_calibrated_channelwise_gate_probe.py").read_text(encoding="utf-8")
    for token in ("mamba-channelwise-residual-scale", "mamba_scan_count", "run_gpu_guarded.py", "screen_rule"):
        assert token in text


def test_weather_calibrated_channelwise_gate_audit_checks_shape_and_timemixer_gate():
    text = Path("scripts/audit_weather_calibrated_channelwise_gate_probe.py").read_text(encoding="utf-8")
    for token in ("residual_scale_shape", "official_timemixer_mse", "screen_passes_expansion_margin", "test boundary"):
        assert token in text
