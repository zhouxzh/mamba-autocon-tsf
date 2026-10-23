from pathlib import Path


def test_v7_readout_script_exists_and_declares_strict_contract():
    path = Path("scripts/analyze_mambatsv7_acf_linear_base_screen.py")
    text = path.read_text(encoding="utf-8")
    for token in ("source_tree_sha256_before", "source_tree_sha256_after", "training_complete.json", "metrics.npy", "MambaTSv7AutoCon", "acf_effective_memory_rms"):
        assert token in text

