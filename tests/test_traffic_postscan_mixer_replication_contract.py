from pathlib import Path


def test_postscan_replication_uses_fresh_seeds_and_mixer_transfer_gate():
    text=Path("scripts/run_traffic_d32_postscan_mixer_replication.py").read_text(encoding="utf-8")
    for token in ("6099", "6100", "6101", "no_result_combination", "--mamba-ts-shared-postscan-mixer", "validation_only", "run_gpu_guarded.py"):
        assert token in text
