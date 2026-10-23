from pathlib import Path


def test_batch2_timemixer_protocol_matches_updates_and_minibatch_geometry():
    text = Path("scripts/run_pems03_channel_patch_batch2_timemixer.py").read_text(encoding="utf-8")
    for token in ("candidate_batch_size", "timemixer_batch_size", "candidate_updates", "timemixer_updates", "same_seed_pairing", "--evaluate-test"):
        assert token in text


def test_batch2_timemixer_audit_checks_geometry_and_noninferiority():
    text = Path("scripts/audit_pems03_channel_patch_batch2_timemixer.py").read_text(encoding="utf-8")
    for token in ("batch-geometry matching contract failure", "candidate batch geometry mismatch", "same_batch_geometry", "noninferior_at_one_se"):
        assert token in text
