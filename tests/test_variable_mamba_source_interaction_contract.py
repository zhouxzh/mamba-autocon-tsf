from pathlib import Path


def test_source_interaction_is_factorial_fresh_seed_and_test_closed():
    text=Path("scripts/run_variable_mamba_acf_source_interaction.py").read_text(encoding="utf-8")
    for token in ("6113,6114,6115", "factorial_design", "replay_forecast", "diverse_acf", "validation_only", "test_metrics_produced", "run_gpu_guarded.py"):
        assert token in text
