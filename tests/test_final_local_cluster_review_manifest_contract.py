from pathlib import Path


def test_review_manifest_covers_handoff_inputs_without_training():
    text = Path("scripts/build_final_local_cluster_review_manifest.py").read_text(encoding="utf-8")
    for token in (
        "review_manifest_20261009.json",
        "independent_patchtst_audit.json",
        "independent_replication_program_20261008/audit.json",
        "backbone_selection_audit_20261009.json",
        "local_cluster_manuscript_outline_20261009.md",
        "result_mutation",
        "training_performed",
        "new_test_reader_opened",
    ):
        assert token in text
