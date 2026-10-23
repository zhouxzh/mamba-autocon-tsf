from pathlib import Path


def test_batch2_precision_expansion_registers_fresh_and_prior_seed_sets():
    text = Path("scripts/run_pems03_channel_patch_batch2_timemixer_expansion.py").read_text(encoding="utf-8")
    for token in ("fresh_seeds", "existing_seeds", "aggregation_rule", "candidate_batch_size", "timemixer_batch_size", "run_gpu_guarded.py"):
        assert token in text


def test_batch2_precision_audit_requires_ten_seed_geometry_and_fresh_contract():
    text = Path("scripts/audit_pems03_channel_patch_batch2_timemixer_expansion.py").read_text(encoding="utf-8")
    for token in ("PRIOR_SEEDS", "FRESH_SEEDS", "channel_patch_batch2_batch2", "fresh TimeMixer contract failure", "same_batch_geometry", "noninferior_at_one_se"):
        assert token in text
