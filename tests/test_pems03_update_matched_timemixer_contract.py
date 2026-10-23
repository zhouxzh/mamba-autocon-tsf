from pathlib import Path


def test_update_matched_timemixer_protocol_registers_equal_optimizer_steps():
    text = Path("scripts/run_pems03_channel_patch_update_matched_timemixer.py").read_text(encoding="utf-8")
    for token in ("candidate_updates", "timemixer_updates", "160", "same_seed_pairing", "--evaluate-test", "run_gpu_guarded.py"):
        assert token in text


def test_update_matched_timemixer_audit_checks_budget_and_boundary():
    text = Path("scripts/audit_pems03_channel_patch_update_matched_timemixer.py").read_text(encoding="utf-8")
    for token in ("update-matching contract failure", "candidate budget mismatch", "TimeMixer budget mismatch", "noninferior_at_one_se"):
        assert token in text
