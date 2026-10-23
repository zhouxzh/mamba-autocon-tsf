from pathlib import Path


def test_postscan_mixer_gate_is_validation_only_and_single_scan():
    text = Path("scripts/run_traffic_d32_postscan_mixer_probe.py").read_text(encoding="utf-8")
    for token in ("postscan_channel_patch", "validation_only", "test_metrics_produced", "--mamba-ts-shared-postscan-mixer", "mamba_scan_count", "run_gpu_guarded.py"):
        assert token in text
