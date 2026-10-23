from pathlib import Path


def test_local_cluster_packet_requires_explicit_remote_verification_and_validation_gate():
    runner = Path("scripts/run_mambatsv3_local_cluster_independent_replication.py").read_text(encoding="utf-8")
    audit = Path("scripts/audit_mambatsv3_local_cluster_independent_replication.py").read_text(encoding="utf-8")
    for token in ("6130, 6131, 6132", "contrastive_cluster", "channel_coverage", "--open-test-after-validation", "validation_gate_before_test", "unverified_operator_claim", "run_gpu_guarded.py"):
        assert token in runner
    for token in ("--verified-independent-machine-id", "--host-evidence", "different_from_audit_host", "validation_gate.json", "independent_machine"):
        assert token in audit
