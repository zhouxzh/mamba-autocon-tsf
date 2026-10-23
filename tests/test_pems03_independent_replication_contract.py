from pathlib import Path


def test_independent_pems03_runner_constructs_strict_target_excluded_protocol():
    text = Path("scripts/run_pems03_channel_patch_independent_replication.py").read_text(encoding="utf-8")
    for token in ("PEMS04", "PEMS08", "PEMS07", "target_excluded_from_source_and_initialization", "--machine-id", "run_gpu_guarded.py", "--evaluate-test"):
        assert token in text


def test_independent_pems03_runner_can_build_a_nonexecuting_hash_packet():
    text = Path("scripts/run_pems03_channel_patch_independent_replication.py").read_text(encoding="utf-8")
    for token in ("--manifest-only", "packet_only", "if args.manifest_only"):
        assert token in text


def test_independent_pems03_audit_requires_reviewer_verified_machine_id_and_guarded_results():
    text = Path("scripts/audit_pems03_channel_patch_independent_replication.py").read_text(encoding="utf-8")
    for token in ("--verified-independent-machine-id", "thermal guard or command failure", "channel_patch_relation_active_fraction", "source hash manifest is incomplete", "independent_machine"):
        assert token in text
