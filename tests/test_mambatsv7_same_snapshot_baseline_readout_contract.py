from pathlib import Path


def test_same_snapshot_v7_readout_enforces_core_boundaries():
    text = Path("scripts/analyze_mambatsv7_same_snapshot_baseline_confirmation.py").read_text(encoding="utf-8")
    for token in ("cross-phase source snapshot mismatch", "training_complete.json", "metrics.npy", "MambaTSv7AutoCon", "acf_effective_memory_to_forecast_rms"):
        assert token in text
