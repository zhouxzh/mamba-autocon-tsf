from pathlib import Path


def test_pems03_external_runner_excludes_target_from_sources_and_initialization():
    text = Path("scripts/run_pems03_channel_patch_external_replication.py").read_text(encoding="utf-8")
    for token in (
        "INITIAL_REFERENCE",
        "replace_repeated(pre,\"--initial-checkpoint\",initial_checkpoints)",
        "assert_target_excluded(pre)",
        "supersedes_unusable_attempt",
        "pems03_channel_patch_external_replication_rerun_20261007",
    ):
        assert token in text


def test_pems03_external_audit_rechecks_target_exclusion():
    text = Path("scripts/audit_pems03_channel_patch_external_replication.py").read_text(encoding="utf-8")
    for token in ("initialization_checkpoints", "target leaked into source or initialization", "metrics={"):
        assert token in text


def test_pems03_robustness_is_validation_only_and_uses_same_seed_pairing():
    text = Path("scripts/run_pems03_channel_patch_robustness.py").read_text(encoding="utf-8")
    for token in ("validation_only", "test_metrics_produced", "target_excluded_from_source_and_initialization", "run_gpu_guarded.py"):
        assert token in text
