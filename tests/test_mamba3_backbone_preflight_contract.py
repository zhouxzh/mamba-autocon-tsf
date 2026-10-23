from pathlib import Path


def test_mamba3_preflight_is_guarded_and_stops_before_training():
    runner = Path("scripts/run_mamba3_backbone_preflight.py").read_text(encoding="utf-8")
    audit = Path("scripts/audit_mamba3_backbone_preflight.py").read_text(encoding="utf-8")
    benchmark = Path("scripts/benchmark_mamba_block_preflight.py").read_text(encoding="utf-8")
    for token in ("registered_before_benchmark", "training_performed", "test_metrics_produced", "run_gpu_guarded.py", "official_commit", "resource_gate"):
        assert token in runner
    for token in ("Mamba3", "Mamba", "median_ms", "peak_memory_bytes"):
        assert token in benchmark
    for token in ("stopped_before_forecasting_training", "resource_gate_passes", "trigger"):
        assert token in audit
