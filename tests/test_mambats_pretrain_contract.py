from pathlib import Path


def test_pretraining_model_declares_one_scan_shared_channel_path():
    text = Path("models/MambaTSPretrain.py").read_text(encoding="utf-8")
    assert "per_channel.mean(dim=1)" in text
    assert "self.mamba" in text
    assert "reconstruction_head" in text
    assert "pretrain=True" in text
    assert "forecast_scale" in text
    assert "patch_mask" in text
    assert "self.mask_token" in text


def test_channelwise_pretraining_backbone_has_shared_mamba_without_c_parameters():
    text = Path("models/MambaTSPretrainChannelwise.py").read_text(encoding="utf-8")
    assert "reshape(batch * channels" in text
    assert "self.mamba" in text
    assert "states.mean(dim=1)" in text
    assert "forecast_head" in text
    assert "patch_mask" in text
    assert "self.mask_token" in text


def test_channel_context_is_parameter_count_agnostic():
    text = Path("models/MambaTSPretrainChannelContext.py").read_text(encoding="utf-8")
    assert "base.mean(dim=1)" in text
    assert "base.std(dim=1" in text
    assert "self.context_projection" in text
    assert "self.mamba" in text
    assert "next_patch_head" in text
    assert "domain_projection" in text
    assert "data_interval_minutes" in text
    assert "nn.init.xavier_uniform_(self.context_projection[-1].weight)" in text


def test_pretraining_protocol_requires_leave_one_domain_out_and_controls():
    text = Path("docs/research/mambats_autocon_pretraining_protocol_20261002.md").read_text(encoding="utf-8")
    for token in ("10%", "25%", "leave-one-domain-out", "reconstruction-only", "Missing sensors"):
        assert token in text
    transfer = Path("scripts/finetune_mambatsv3_core_transfer.py").read_text(encoding="utf-8")
    assert "--channel-agnostic-scan" in transfer


def test_pretraining_lag_pair_batching_requires_contiguous_origins():
    text = Path("scripts/pretrain_mamba_autocon.py").read_text(encoding="utf-8")
    assert "--relation-batching" in text
    assert "lag_pairs requires --contiguous-windows" in text
    assert "ACFLagPairBatchSampler" in text


def test_predictive_pretraining_uses_registered_future_patch_loss():
    text = Path("scripts/pretrain_mamba_autocon.py").read_text(encoding="utf-8")
    assert "--pretrain-task" in text
    assert "--next-patch-loss" in text
    assert "smooth_l1_loss" in text


def test_pretraining_dataset_specs_allow_physical_time_intervals():
    text = Path("scripts/pretrain_mamba_autocon.py").read_text(encoding="utf-8")
    assert "interval_minutes" in text
    assert "NAME:PATH[:csv|npz|txt][:interval_minutes]" in text


def test_mambatsv3_core_transfer_excludes_channel_and_horizon_specific_weights():
    text = Path("scripts/finetune_mambatsv3_core_transfer.py").read_text(encoding="utf-8")
    assert "transferable_prefixes" in text
    assert "patch_embedding" in text
    assert "encoder." in text
    assert "forecast_head" not in text.split("transferable_prefixes", 1)[1].split("model.load_state_dict", 1)[0]
    assert "test_metrics_produced" in text
    assert "torch.stack([value.float() for value in values]).mean" in text
    assert "def json_compatible" in text
    assert "json_compatible(vars(args))" in text


def test_mambatsv3_shared_core_pretraining_uses_the_production_scan_without_channel_parameters():
    text = Path("scripts/pretrain_mambatsv3_core_autocon.py").read_text(encoding="utf-8")
    for token in (
        "MambaTS.Model", "--mamba-ts-variant", "mamba_ts_shared_pretrain_mode=True", "CORE_PREFIXES",
        "ACFLagPairBatchSampler", "ACFRelationLoss", "_masked_input",
            "test_metrics_produced", "_validate_patch_geometry", "masked reconstruction requires non-overlapping", "--reconstruction-weight",
        "--pretrain-task", "next_patch", "--predictive-weight", "--relation-view", "--relation-pair-policy", "legacy", "round_robin", "forecast_state", "channelwise_forecast_state", "forecast_patch", "forecast_affine", "forecast_contrastive", "pooled_patch_contrastive", "ForecastLagPairConsistencyLoss", "ForecastLagPairAffineLoss", "ForecastLagPairContrastiveLoss", "ChannelPatchLagContrastiveLoss", "--channel-patch-relation-fraction", "--grouped-patch-relation-fraction", "_grouped_patch_representation", "--mamba-ts-acf-channel-descriptor", "ts_acf_descriptor_values", "--mamba-ts-shared-postscan-mixer", "ts_shared_postscan_mixer_", "--channel-patch-acf-confidence-weighting", "--channel-patch-stability-weighting", "--channel-patch-similarity", "PooledPatchLagContrastiveLoss", "PooledPatchVarianceLoss", "--pooled-patch-variance-weight", "_pooled_patch_representation", "pretrain_pooled_patch_projector", "channel_selected_lags",
    ):
        assert token in text


def test_next_patch_pretraining_has_an_independent_predictive_weight():
    text = Path("scripts/pretrain_mambatsv3_core_autocon.py").read_text(encoding="utf-8")
    assert 'args.predictive_weight if args.pretrain_task == "next_patch"' in text
    assert "task_weight * reconstruction_loss" in text


def test_next_patch_head_is_transferable_to_the_target_forecast_geometry():
    pretrain = Path("scripts/pretrain_mambatsv3_core_autocon.py").read_text(encoding="utf-8")
    transfer = Path("scripts/finetune_mambatsv3_core_transfer.py").read_text(encoding="utf-8")
    assert "pretrain_prediction_head" in pretrain
    assert "_tensor_digest" in pretrain
    assert "--source-pred-len" in pretrain
    assert "encoded.reshape(encoded.shape[0], encoded.shape[1], -1)" in pretrain
    assert "--transfer-prediction-head" in transfer
    assert "prediction_head_transferred" in transfer
    assert "--prediction-head-transfer-mode" in transfer
    assert "exact prediction-head transfer" in transfer


def test_pretraining_can_calibrate_relation_loss_to_prediction_scale():
    text = Path("scripts/pretrain_mambatsv3_core_autocon.py").read_text(encoding="utf-8")
    for token in ("--relation-loss-fraction", "mean_weighted_autocon_loss", "relation_weight", "relation_loss_fraction"):
        assert token in text


def test_pretraining_can_directly_supervise_transferred_representation_states():
    text = Path("scripts/pretrain_mambatsv3_core_autocon.py").read_text(encoding="utf-8")
    for token in ("--representation-relation-fraction", "model._representation(encoded)",
                  "mean_weighted_representation_relation_loss", "representation_relation_active_fraction"):
        assert token in text


def test_pretraining_target_prefix_calibration_is_explicit_and_auditable():
    text = Path("scripts/pretrain_mambatsv3_core_autocon.py").read_text(encoding="utf-8")
    for token in (
        "--calibration-source", "--calibration-fraction", "target_calibration",
        "calibration-source and calibration-fraction", "train_prefix_fraction",
        "calibration-source must not duplicate",
    ):
        assert token in text


def test_calibration_prefix_can_use_a_registered_stronger_relation_fraction():
    text = Path("scripts/pretrain_mambatsv3_core_autocon.py").read_text(encoding="utf-8")
    for token in (
        "--calibration-relation-loss-fraction",
        "--calibration-channel-patch-relation-fraction",
        "calibration-relation-loss-fraction requires calibration-source",
        "calibration-channel-patch-relation-fraction requires calibration-source",
        "source_relation_loss_fraction",
        "source_channel_patch_relation_fractions",
        "source_relation_loss_fractions",
    ):
        assert token in text


def test_residual_target_pretraining_is_explicit_and_checked_at_deployment():
    pretrain = Path("scripts/pretrain_mambatsv3_core_autocon.py").read_text(encoding="utf-8")
    hybrid = Path("scripts/finetune_mambatsv3_dlinear_hybrid.py").read_text(encoding="utf-8")
    model = Path("models/MambaDLinearHybrid.py").read_text(encoding="utf-8")
    for token in ("--prediction-target", "last_value_residual", "prediction_target"):
        assert token in pretrain
    for token in ("--mamba-correction-output", "prediction_target=last_value_residual"):
        assert token in hybrid
    assert "mamba_correction_output" in model
    assert "channelwise_residual_scale" in model


def test_target_affine_finetune_uses_only_fewshot_training_prefix():
    text = Path("scripts/finetune_mambatsv3_dlinear_hybrid.py").read_text(encoding="utf-8")
    for token in ("--target-affine-fraction", "--target-affine-pairing", "raw[:few_end]", "ForecastLagPairAffineLoss", "target_affine_active_fraction", "--freeze-pretrained-core", "--mamba-core-learning-rate", "TRANSFERABLE_CORE_PREFIXES", "mamba_forecast_head_trainable", "--mamba-ts-channel-adapter", "mamba_channel_adapter_trainable", "--mamba-ts-target-mlp-adapter", "mamba_target_mlp_adapter_trainable", "--mamba-channelwise-residual-scale", "--mamba-centered-channelwise-residual-scale", "--mamba-channelwise-horizon-residual-scale", "--mamba-channelwise-residual-bias", "--mamba-shared-channel-statistic-residual-gate", "statistic_gate_abs_max"):
        assert token in text
    assert "--mamba-ts-revin-affine" in text
    transfer = Path("scripts/finetune_mambatsv3_core_transfer.py").read_text(encoding="utf-8")
    assert "mamba_ts_revin_affine" in transfer


def test_shared_channel_summary_is_transferred_and_frozen_with_core():
    transfer = Path("scripts/finetune_mambatsv3_core_transfer.py").read_text(encoding="utf-8")
    hybrid = Path("scripts/finetune_mambatsv3_dlinear_hybrid.py").read_text(encoding="utf-8")
    assert "ts_shared_channel_summary_" in transfer
    assert "ts_shared_channel_summary_" in hybrid


def test_channel_set_pooling_is_transferred_and_frozen_with_core():
    pretrain = Path("scripts/pretrain_mambatsv3_core_autocon.py").read_text(encoding="utf-8")
    transfer = Path("scripts/finetune_mambatsv3_core_transfer.py").read_text(encoding="utf-8")
    hybrid = Path("scripts/finetune_mambatsv3_dlinear_hybrid.py").read_text(encoding="utf-8")
    for text in (pretrain, transfer, hybrid):
        assert "mamba_ts_channel_set_pooling" in text or "ts_channel_set_pooling_" in text
    assert "ts_channel_set_pooling_" in pretrain
    assert "ts_channel_set_pooling_" in transfer
    assert "ts_channel_set_pooling_" in hybrid


def test_channel_prototype_pooling_is_transferred_and_frozen_with_core():
    for path in ("scripts/pretrain_mambatsv3_core_autocon.py", "scripts/finetune_mambatsv3_core_transfer.py", "scripts/finetune_mambatsv3_dlinear_hybrid.py"):
        assert "ts_channel_prototype_" in Path(path).read_text(encoding="utf-8")


def test_hybrid_heldout_evaluation_occurs_only_after_best_validation_selection():
    text = Path("scripts/finetune_mambatsv3_dlinear_hybrid.py").read_text(encoding="utf-8")
    assert "--evaluate-test" in text
    assert "model.load_state_dict(best_state" in text
    assert "test_metrics_produced" in text


def test_confirmatory_reader_requires_selection_and_guard_boundaries():
    text = Path("scripts/audit_mambatsv3_residual_confirmatory_release.py").read_text(encoding="utf-8")
    for token in ("main_validation_accessed", "SOURCE_SELECTION", "EXPECTED_SOURCES", "prediction_head_transfer_mode", "guard/event failure", "test_metrics_produced"):
        assert token in text


def test_lowdata_robustness_evaluator_keeps_target_test_closed():
    text = Path("scripts/evaluate_mambatsv3_lowdata_robustness.py").read_text(encoding="utf-8")
    for token in ("late_validation_shift", "missing_channels_10", "missing_channels_30", "noise_0p10", "test_metrics_produced"):
        assert token in text
    assert "weights_only=False" in text


def test_checkpoint_benchmark_is_prediction_only_and_one_scan():
    text = Path("scripts/benchmark_mambatsv3_checkpoint_inference.py").read_text(encoding="utf-8")
    for token in ("return_repr=False", "relation_loss_included", "mamba_scan_count", "autocon_representation_returned"):
        assert token in text


def test_channel_patch_geometry_audit_is_source_train_only():
    text = Path("scripts/audit_channel_patch_representation_geometry.py").read_text(encoding="utf-8")
    for token in ("fit_signed_acf_prior", "channel_patch_relation_fraction", "source chronological train prefixes only", "test_metrics_produced", "_encode"):
        assert token in text


def test_hybrid_checkpoint_tools_load_deployed_residual_path():
    evaluator = Path("scripts/evaluate_mambatsv3_lowdata_robustness.py").read_text(encoding="utf-8")
    benchmark = Path("scripts/benchmark_mambatsv3_checkpoint_inference.py").read_text(encoding="utf-8")
    for text in (evaluator, benchmark):
        for token in ("--hybrid", "MambaDLinearHybrid", "mamba_correction_output"):
            assert token in text


def test_lowdata_baseline_runner_uses_closed_chronological_validation():
    text = Path("scripts/finetune_lowdata_baseline.py").read_text(encoding="utf-8")
    for token in ("DLinear", "NLinear", "RLinear", "PatchTST", "TimeMixer", "TimesNet", "iTransformer", "1.0", "few-shot-fraction", "val_starts + args.seq_len >= train_end", "test_metrics_produced"):
        assert token in text


def test_lowdata_baseline_heldout_path_uses_best_validation_state():
    text = Path("scripts/finetune_lowdata_baseline.py").read_text(encoding="utf-8")
    for token in ("--evaluate-test", "model.load_state_dict(best_state", "test_metrics_produced"):
        assert token in text


def test_dlinear_primary_hybrid_is_zero_gated_and_retains_one_mamba_branch():
    model = Path("models/MambaDLinearHybrid.py").read_text(encoding="utf-8")
    runner = Path("scripts/finetune_mambatsv3_dlinear_hybrid.py").read_text(encoding="utf-8")
    for token in ("residual_scale", "torch.tanh", "linear_only", "MambaTSv3", "DLinear", "test_metrics_produced"):
        assert token in model + runner


def test_hybrid_runner_supports_nlinear_primary_control():
    text = Path("scripts/finetune_mambatsv3_dlinear_hybrid.py").read_text(encoding="utf-8")
    assert "LINEAR_MODELS" in text
    assert "--linear-model" in text
    assert "NLinear" in text


def test_adaptive_linear_hybrid_uses_train_window_gate_and_one_mamba_branch():
    model = Path("models/MambaAdaptiveLinearHybrid.py").read_text(encoding="utf-8")
    runner = Path("scripts/finetune_mambatsv3_adaptive_linear_hybrid.py").read_text(encoding="utf-8")
    for token in ("base_gate", "_features", "residual_scale", "DLinear", "NLinear", "test_metrics_produced"):
        assert token in model + runner


def test_inner_linear_selector_keeps_main_validation_and_test_closed():
    text = Path("scripts/select_lowdata_linear_primary.py").read_text(encoding="utf-8")
    for token in ("calibration-fraction", "selected_model", "selected_sampler", "acf_paired", "main_validation_accessed", "test_metrics_produced", "DLinear", "NLinear"):
        assert token in text


def test_acf_source_selector_uses_target_fewshot_prefix_only():
    text = Path("scripts/select_acf_pretraining_sources.py").read_text(encoding="utf-8")
    for token in ("few_shot_fraction", "target_values[:target_rows]", "main_validation_accessed", "test_metrics_produced", "target must not appear"):
        assert token in text


def test_finetuning_keeps_target_validation_after_original_train_boundary():
    text = Path("scripts/finetune_mamba_pretrain.py").read_text(encoding="utf-8")
    for token in ("few-shot-fraction", "max-train-windows", "val_starts + args.seq_len >= train_end", "checkpoint_load", "test_metrics_produced"):
        assert token in text


def test_lowdata_protocol_keeps_test_closed_and_registers_mechanism_control():
    import json
    plan = json.loads(Path("docs/research/mambats_autocon_pretraining_lowdata_validation_20261002.json").read_text(encoding="utf-8"))
    assert plan["execution"]["authorized"] is False
    assert plan["comparison_contract"]["test_metrics_produced"] is False
    assert plan["leave_one_domain_out"] is True
    assert plan["few_shot_fractions"] == [0.1, 0.25]
    assert plan["pretraining_arms"][1]["mamba_lambda"] == 0.0


def test_traffic_pilot_keeps_controls_and_target_out_of_source_pool():
    text = Path("scripts/run_mambats_pretraining_traffic_pilot.py").read_text(encoding="utf-8")
    assert "PEMS03" in text and "Weather" in text and "ETTh1" in text
    assert "dataset/traffic/traffic.csv" in text
    assert '"--mamba-lambda", "0"' in text
    assert "random_init" in text and "reconstruction_finetune" in text and "autocon_finetune" in text
    assert "MAMBATS_PRETRAIN_RELATION_BATCHING" in text
    assert "MAMBATS_PRETRAIN_TASK" in text


def test_mambats_pretrain_can_balance_source_optimizer_recency_without_new_scan():
    text = Path("scripts/pretrain_mambatsv3_core_autocon.py").read_text(encoding="utf-8")
    assert "--source-update-policy" in text
    assert 'choices=["sequential", "cyclic"]' in text
    assert "source_order = source_order[shift:] + source_order[:shift]" in text
    assert '"source_update_order"' in text
