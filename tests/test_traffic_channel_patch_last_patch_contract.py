from pathlib import Path


def test_final_patch_channel_similarity_screen_preserves_single_scan_contract():
    text = Path("scripts/run_traffic_d32_channel_patch_last_patch_probe.py").read_text(encoding="utf-8")
    for token in (
        "trajectory", "last_patch", "validation_only", "test_metrics_produced",
        "--channel-patch-similarity", "mamba_scan_count", "run_gpu_guarded.py",
    ):
        assert token in text
