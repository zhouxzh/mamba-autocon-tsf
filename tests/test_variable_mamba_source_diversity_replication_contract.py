from pathlib import Path


def test_source_diversity_replication_is_fresh_seed_and_test_closed():
    text=Path("scripts/run_variable_mamba_acf_source_diversity_replication.py").read_text(encoding="utf-8")
    for token in ("6110,6111,6112", "replication_seeds", "validation_only", "test_metrics_produced", "run_gpu_guarded.py", "one variable-axis Mamba scan"):
        assert token in text
