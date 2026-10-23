from pathlib import Path


def test_task_aligned_acf_probe_is_paired_guarded_and_validation_only():
    text=Path("scripts/run_mambatsv3_forecast_target_acf_probe.py").read_text(encoding="utf-8")
    for token in ("raw_signal_acf", "forecast_target_acf", "--forecast-target-acf", "validation_only", "test_metrics_produced", "run_gpu_guarded.py", "last_value_residual"):
        assert token in text
