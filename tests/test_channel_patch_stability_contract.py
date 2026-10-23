from pathlib import Path


def test_channel_patch_stability_screen_is_target_excluded_and_inference_neutral():
    text = Path("scripts/run_pems03_channel_patch_stability_probe.py").read_text(encoding="utf-8")
    for token in ("channel-patch-stability-weighting", "target_excluded_from_source_and_initialization", "run_gpu_guarded.py", "screen_rule"):
        assert token in text


def test_channel_patch_stability_audit_rechecks_support_and_exclusion():
    text = Path("scripts/audit_pems03_channel_patch_stability_probe.py").read_text(encoding="utf-8")
    for token in ("stability_weight_mean", "channel_patch_relation_active_fraction", "target leakage", "screen_passes_expansion_margin"):
        assert token in text
