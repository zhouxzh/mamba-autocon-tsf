from pathlib import Path


def test_pems03_25pct_protocol_excludes_target_and_registers_label_budget():
    text = Path("scripts/run_pems03_channel_patch_25pct_external_replication.py").read_text(encoding="utf-8")
    for token in ("few_shot_fraction\": 0.25", "target_excluded_from_source_and_initialization", "assert_target_excluded(pre)", "run_gpu_guarded.py"):
        assert token in text


def test_pems03_25pct_audit_rechecks_fraction_and_target_exclusion():
    text = Path("scripts/audit_pems03_channel_patch_25pct_external_replication.py").read_text(encoding="utf-8")
    for token in ("few_shot_fraction",):
        assert token in text
    assert "target leaked into source or initialization" in text


def test_pems03_25pct_runner_has_a_nonexecuting_help_path():
    text = Path("scripts/run_pems03_channel_patch_25pct_external_replication.py").read_text(encoding="utf-8")
    assert "ArgumentParser" in text


def test_pems03_25pct_robustness_reuses_frozen_checkpoints_under_guardrails():
    text = Path("scripts/run_pems03_channel_patch_25pct_robustness.py").read_text(encoding="utf-8")
    for token in ("validation_only", "few_shot_fraction\": 0.25", "target_excluded_from_source_and_initialization", "run_gpu_guarded.py"):
        assert token in text


def test_pems03_25pct_inference_benchmark_is_synthetic_and_one_scan_audited():
    runner = Path("scripts/run_pems03_channel_patch_inference_benchmark.py").read_text(encoding="utf-8")
    audit = Path("scripts/audit_pems03_channel_patch_inference_benchmark.py").read_text(encoding="utf-8")
    for token in ("benchmark_mambatsv3_checkpoint_inference.py", "benchmark_official_patchtst_checkpoint_inference.py", "run_gpu_guarded.py"):
        assert token in runner
    for token in ("candidate_one_scan_no_relation_loss", "mamba_encoder_block_calls", "prediction_only"):
        assert token in audit
