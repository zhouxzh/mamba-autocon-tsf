from pathlib import Path


def test_task_sign_infonce_probe_is_paired_guarded_and_validation_only():
    text=Path("scripts/run_mambatsv3_forecast_target_infonce_weather_probe.py").read_text(encoding="utf-8")
    for token in ("raw_peak_sign", "forecast_target_sign", "--forecast-target-acf", "forecast_contrastive", "validation_only", "test_metrics_produced", "run_gpu_guarded.py", "weather_h192_s5450_active"):
        assert token in text
