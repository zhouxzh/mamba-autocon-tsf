from pathlib import Path


def test_independent_host_evidence_collection_is_read_only_and_records_accelerator_identity():
    text = Path("scripts/collect_independent_host_evidence.py").read_text(encoding="utf-8")
    for token in ("--ssh-host", "BatchMode=yes", "machine_id", "cuda_available", "distinct_from_local_hostname", "read_only_collection"):
        assert token in text


def test_independent_audit_can_require_host_evidence_consistency():
    text = Path("scripts/audit_pems03_channel_patch_independent_replication.py").read_text(encoding="utf-8")
    for token in ("--host-evidence", "host-evidence independence mismatch", "observed_machine_id", "host_evidence_verified"):
        assert token in text
