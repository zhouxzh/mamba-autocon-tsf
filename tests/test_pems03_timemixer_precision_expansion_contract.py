from pathlib import Path


def test_pems03_timemixer_precision_expansion_is_registered_and_guarded():
    text = Path("scripts/run_pems03_channel_patch_timemixer_precision_expansion.py").read_text(encoding="utf-8")
    for token in ("existing_seed_set", "aggregation_rule", "target excluded", "run_gpu_guarded.py", "validation_only"):
        assert token in text


def test_pems03_timemixer_precision_audit_uses_only_registered_five_seed_set():
    text = Path("scripts/audit_pems03_channel_patch_timemixer_precision_expansion.py").read_text(encoding="utf-8")
    for token in ("SEEDS = (6060, 6061, 6062, 6065, 6066)", "test boundary", "noninferior_at_one_se", "aggregation_rule", "_timemixer_event.json"):
        assert token in text
