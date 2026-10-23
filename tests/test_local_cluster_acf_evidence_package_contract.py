from pathlib import Path


def test_local_cluster_package_separates_confirmations_extensions_and_boundaries():
    builder = Path("scripts/build_local_cluster_acf_evidence_package.py").read_text(encoding="utf-8")
    auditor = Path("scripts/audit_local_cluster_acf_evidence_package.py").read_text(encoding="utf-8")
    for token in ("CONFIRMATIONS", "FAILED_GATES", "ROBUSTNESS", "OFFICIAL", "OFFICIAL_ITRANSFORMER", "LAUNCHER_REHEARSAL", "submission_ready"):
        assert token in builder
    for token in ("six_heldout_confirmations", "failed_gates_closed", "latency_boundary_retained", "same_machine_replication_retained", "clean_install_replication_retained", "launcher_rehearsal_retained", "independent_machine_replications_valid", "independent_replication_program_valid", "independent_official_patchtst_extension_valid", "independent_machine_official_patchtst_extension", "weather_h192_10", "iTransformer_extension_retained", "claim_ids_and_files_bound", "human_evidence_still_unverified", "human_review_gate_visible", "human_gates"):
        assert token in auditor
