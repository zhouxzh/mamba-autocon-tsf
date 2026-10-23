from pathlib import Path


def test_final_batch2_expansion_has_explicit_five_seed_cap_and_stop_rule():
    text = Path("scripts/run_pems03_channel_patch_batch2_timemixer_final_expansion.py").read_text(encoding="utf-8")
    for token in ("SEEDS = (6089, 6090, 6091, 6092, 6093)", "stop_rule", "final TimeMixer precision expansion", "same_seed_pairing", "--evaluate-test"):
        assert token in text


def test_final_batch2_audit_combines_only_fifteen_registered_seeds():
    text = Path("scripts/audit_pems03_channel_patch_batch2_timemixer_final_expansion.py").read_text(encoding="utf-8")
    for token in ("PRIOR_FIRST", "PRIOR_SECOND", "channel_patch_batch2.json", "channel_patch_batch2_batch2.json", "TIMEMIXER_BASE", "TIMEMIXER_FIRST_EXPANSION", "prior_candidate_path", "prior_timemixer_path", "FRESH_SEEDS", "final expansion registration failure", "final_precision_expansion", "noninferior_at_one_se"):
        assert token in text
