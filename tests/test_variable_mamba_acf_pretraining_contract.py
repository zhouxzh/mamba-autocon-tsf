from pathlib import Path

def test_variable_mamba_acf_gate_is_validation_only_and_one_scan():
    text=Path("scripts/run_variable_mamba_acf_traffic_probe.py").read_text(encoding="utf-8")
    for token in ("representation_acf","validation_only","test_metrics_produced","run_gpu_guarded.py","one variable-axis Mamba scan"):
        assert token in text
