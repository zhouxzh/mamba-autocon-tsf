from pathlib import Path


def test_fresh_seed_long_horizon_reader_checks_required_evidence():
    text = Path("scripts/analyze_mambatsv7_multihorizon_fresh_seed_confirmation.py").read_text(encoding="utf-8")
    for token in ("HORIZONS = (336, 720)", "SEEDS = (2024, 2025, 2026)", "cross-phase source snapshot mismatch", "metrics.npy", "acf_effective_memory_to_forecast_rms"):
        assert token in text
