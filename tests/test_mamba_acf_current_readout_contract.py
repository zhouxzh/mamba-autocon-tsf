from pathlib import Path


def test_current_mamba_acf_readout_enforces_sidecars_and_null_effect():
    text = Path("scripts/analyze_mamba_acf_current_strong_baseline_screen.py").read_text(encoding="utf-8")
    for token in ("acf_prior.npz", "MambaACFAutoCon_null", "active_vs_null_percent", "cross-phase source snapshot mismatch", "metrics.npy"):
        assert token in text
