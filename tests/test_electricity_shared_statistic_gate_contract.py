from pathlib import Path


def test_shared_statistic_gate_screen_is_one_scan_and_validation_only():
    text = Path("scripts/run_electricity_d32_shared_statistic_gate_probe.py").read_text(encoding="utf-8")
    for token in ("validation_only", "test_metrics_produced", "mamba_scan_count", "mamba-shared-channel-statistic-residual-gate", "run_gpu_guarded.py"):
        assert token in text


def test_shared_statistic_gate_audit_requires_active_shared_but_not_channel_indexed_gate():
    text = Path("scripts/audit_electricity_d32_shared_statistic_gate_probe.py").read_text(encoding="utf-8")
    for token in ("mamba_shared_channel_statistic_residual_gate", "mamba_channelwise_residual_scale", "statistic_gate_abs_max", "screen_passes_expansion_margin"):
        assert token in text
