from pathlib import Path
def test_multi_positive_cpc_gate_is_fixed_threshold_validation_only():
    text=Path("scripts/run_variable_mamba_acf_multipositive_probe.py").read_text(encoding="utf-8")
    for token in ("high_acf_cpc","representation-peak-cpc-positive-mode",".3","validation_only","run_gpu_guarded.py"):
        assert token in text
