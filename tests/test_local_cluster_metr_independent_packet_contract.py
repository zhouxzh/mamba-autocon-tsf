from pathlib import Path


def test_metr_packet_keeps_target_out_and_requires_validation_before_test():
    runner = Path("scripts/run_mambatsv3_local_cluster_metr_independent_replication.py").read_text(encoding="utf-8")
    audit = Path("scripts/audit_mambatsv3_local_cluster_metr_independent_replication.py").read_text(encoding="utf-8")
    for token in ("6140, 6141, 6142", "METR-LA", "PEMS-BAY", "contrastive_cluster", "channel_coverage", "--open-test-after-validation", "validation_gate_before_test", "run_gpu_guarded.py"):
        assert token in runner
    for token in ("--verified-independent-machine-id", "--host-evidence", "unexpected active source pool", "validation_gate.json", "independent_machine"):
        assert token in audit
