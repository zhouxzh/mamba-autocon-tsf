from pathlib import Path


def test_shared_channel_patch_projector_is_train_only_and_opt_in():
    text = Path("scripts/pretrain_mambatsv3_core_autocon.py").read_text(encoding="utf-8")
    for token in (
        "--channel-patch-projector", "choices=[\"identity\", \"mlp\"]",
        "channel_patch_projector(encoded)", "pretrain_channel_patch_projector",
        "--channel-patch-channel-weighting", "channel_patch_channel_weights",
    ):
        assert token in text
    assert 'default="identity"' in text


def test_projector_probe_preserves_strict_validation_contract():
    text = Path("scripts/run_pems03_channel_patch_projector_probe.py").read_text(encoding="utf-8")
    for token in (
        "validation_only", "test_metrics_produced", "target_excluded_from_source_and_initialization",
        "mamba_scan_count", "run_gpu_guarded.py", "--channel-patch-projector",
    ):
        assert token in text
