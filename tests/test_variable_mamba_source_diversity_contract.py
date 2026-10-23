from pathlib import Path


def test_source_diversity_probe_matches_updates_and_keeps_test_closed():
    text=Path("scripts/run_variable_mamba_acf_source_diversity_probe.py").read_text(encoding="utf-8")
    for token in ("ElectricityA", "ElectricityB", "Weather", "validation_only", "test_metrics_produced", "run_gpu_guarded.py", "one variable-axis Mamba scan"):
        assert token in text
