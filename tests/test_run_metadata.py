import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from run import (
    _bind_acf_channel_lag_bank_metadata,
    compact_setting,
    experiment_setting,
    load_acf_prior_artifact,
    mamba_architecture_tag,
    _mamba_acf_prior_required,
    _mamba_runtime_contract_payload,
    _effective_mamba_headdim,
    apply_mamba_public_alias_defaults,
    _preflight_mamba_acf_sidecar,
    clear_acf_prior_metadata,
    load_eval_acf_prior_if_required,
    preserve_acf_prior_metadata,
    prepare_eval_acf_prior_args,
    persist_run_config,
    persist_acf_prior_artifact,
    restore_acf_resolution_metadata,
    validate_persisted_acf_runtime_contract,
    validate_mamba_acf_arguments,
)


class RunMetadataTests(unittest.TestCase):
    def test_lite_public_alias_expands_fixed_architecture_defaults(self):
        args = SimpleNamespace(
            model="MambaACFLiteAutoCon",
            AutoCon=False,
            mamba_scales=[1, 2],
            mamba_causal_embedding=False,
            mamba_low_rank_head=False,
            mamba_temporal_rank=16,
            mamba_variable_mixer=True,
            mamba_patch_mixer=True,
            mamba_local_mixer=True,
            mamba_use_acf_memory=True,
            mamba_acf_relation_loss=False,
        )
        apply_mamba_public_alias_defaults(args)
        self.assertTrue(args.AutoCon)
        self.assertEqual(args.mamba_scales, [1])
        self.assertTrue(args.mamba_causal_embedding)
        self.assertTrue(args.mamba_low_rank_head)
        self.assertEqual(args.mamba_temporal_rank, 8)
        self.assertFalse(args.mamba_variable_mixer)
        self.assertFalse(args.mamba_patch_mixer)
        self.assertFalse(args.mamba_local_mixer)
        self.assertFalse(args.mamba_use_acf_memory)
        self.assertTrue(args.mamba_acf_relation_loss)

    def test_stepwise_reliability_has_explicit_false_default_and_round_trips(self):
        base = {
            "model": "MambaACF",
            "mamba_use_acf_memory": True,
            "mamba_dynamic_acf": False,
        }
        default_payload = _mamba_runtime_contract_payload(base)
        self.assertFalse(default_payload["mamba_acf_step_reliability"])
        enabled_payload = _mamba_runtime_contract_payload(
            {**base, "mamba_acf_step_reliability": "true"}
        )
        self.assertTrue(enabled_payload["mamba_acf_step_reliability"])
        self.assertNotEqual(default_payload, enabled_payload)

    def test_data_dependent_channel_mixer_runtime_contract_and_tag(self):
        base = {
            "model": "MambaACF",
            "mamba_data_dependent_channel_mixer": False,
            "mamba_data_dependent_channel_mixer_rank": 99,
        }
        dormant = _mamba_runtime_contract_payload(base)
        self.assertIsNone(dormant["mamba_data_dependent_channel_mixer_rank"])

        enabled = dict(
            base,
            mamba_data_dependent_channel_mixer=True,
            mamba_data_dependent_channel_mixer_rank=3,
        )
        active = _mamba_runtime_contract_payload(enabled)
        self.assertEqual(active["mamba_data_dependent_channel_mixer_rank"], 3)
        self.assertNotEqual(
            mamba_architecture_tag(SimpleNamespace(**base)),
            mamba_architecture_tag(SimpleNamespace(**enabled)),
        )
        self.assertNotEqual(
            mamba_architecture_tag(SimpleNamespace(**enabled)),
            mamba_architecture_tag(
                SimpleNamespace(
                    **dict(enabled, mamba_data_dependent_channel_mixer_rank=4)
                )
            ),
        )

    def test_data_dependent_channel_mixer_argument_validation(self):
        validate_mamba_acf_arguments(
            SimpleNamespace(
                model="MambaACF",
                mamba_data_dependent_channel_mixer=True,
                mamba_data_dependent_channel_mixer_rank=2,
            )
        )

    def test_variable_gdd_mixer_runtime_contract_and_validation(self):
        base = {
            "model": "MambaACF",
            "mamba_variable_mixer": True,
            "mamba_variable_gdd_mixer": False,
            "mamba_variable_gdd_mixer_rank": 99,
        }
        dormant = _mamba_runtime_contract_payload(base)
        self.assertFalse(dormant["mamba_variable_gdd_mixer"])
        self.assertIsNone(dormant["mamba_variable_gdd_mixer_rank"])
        self.assertIsNone(dormant["mamba_variable_gdd_init_scale"])
        enabled = {
            **base,
            "mamba_variable_gdd_mixer": True,
            "mamba_variable_gdd_mixer_rank": 3,
            "mamba_variable_gdd_init_scale": 0.01,
        }
        active = _mamba_runtime_contract_payload(enabled)
        self.assertTrue(active["mamba_variable_gdd_mixer"])
        self.assertEqual(active["mamba_variable_gdd_mixer_rank"], 3)
        self.assertAlmostEqual(active["mamba_variable_gdd_init_scale"], 0.01)
        self.assertNotEqual(
            mamba_architecture_tag(SimpleNamespace(**base)),
            mamba_architecture_tag(SimpleNamespace(**enabled)),
        )
        validate_mamba_acf_arguments(SimpleNamespace(**enabled))
        with self.assertRaisesRegex(ValueError, "requires --mamba_variable_mixer"):
            validate_mamba_acf_arguments(SimpleNamespace(
                model="MambaACF", mamba_variable_gdd_mixer=True,
            ))
        with self.assertRaisesRegex(ValueError, "supported only by MambaACF"):
            validate_mamba_acf_arguments(SimpleNamespace(
                model="Mamba2", mamba_variable_gdd_mixer=True,
            ))
        with self.assertRaisesRegex(ValueError, "positive integer"):
            validate_mamba_acf_arguments(SimpleNamespace(
                model="MambaACF", mamba_variable_mixer=True,
                mamba_variable_gdd_mixer=True,
                mamba_variable_gdd_mixer_rank=0,
            ))
        with self.assertRaisesRegex(ValueError, "in \[0, 1\]"):
            validate_mamba_acf_arguments(SimpleNamespace(
                model="MambaACF", mamba_variable_mixer=True,
                mamba_variable_gdd_mixer=True,
                mamba_variable_gdd_mixer_rank=3,
                mamba_variable_gdd_init_scale=1.1,
            ))

    def test_variable_repr_fusion_runtime_contract_and_tag(self):
        base = {
            "model": "MambaACFAutoCon",
            "AutoCon": True,
            "mamba_variable_mixer": True,
            "mamba_variable_repr_fusion": False,
        }
        disabled = _mamba_runtime_contract_payload(base)
        enabled = _mamba_runtime_contract_payload(
            {**base, "mamba_variable_repr_fusion": True}
        )
        self.assertFalse(disabled["mamba_variable_repr_fusion"])
        self.assertTrue(enabled["mamba_variable_repr_fusion"])
        self.assertNotEqual(
            mamba_architecture_tag(SimpleNamespace(**base)),
            mamba_architecture_tag(
                SimpleNamespace(**{**base, "mamba_variable_repr_fusion": True})
            ),
        )
        # The field is canonicalized inactive when the variable branch or
        # AutoCon objective is absent, preventing false replay mismatches.
        self.assertFalse(
            _mamba_runtime_contract_payload(
                {**base, "mamba_variable_mixer": False,
                 "mamba_variable_repr_fusion": True}
            )["mamba_variable_repr_fusion"]
        )
        with self.assertRaisesRegex(ValueError, "positive integer"):
            validate_mamba_acf_arguments(
                SimpleNamespace(
                    model="MambaACF",
                    mamba_data_dependent_channel_mixer=True,
                    mamba_data_dependent_channel_mixer_rank=0,
                )
            )
        with self.assertRaisesRegex(ValueError, "shared-mode.*channel_independent"):
            validate_mamba_acf_arguments(
                SimpleNamespace(
                    model="MambaACF",
                    mamba_data_dependent_channel_mixer=True,
                    mamba_data_dependent_channel_mixer_rank=2,
                    mamba_channel_independent=True,
                )
            )
        with self.assertRaisesRegex(ValueError, "supported only by MambaACF"):
            validate_mamba_acf_arguments(
                SimpleNamespace(
                    model="Mamba2",
                    mamba_data_dependent_channel_mixer=True,
                    mamba_data_dependent_channel_mixer_rank=2,
                )
            )

    def test_mamba_runtime_contract_uses_effective_inactive_defaults(self):
        persisted = {
            "model": "MambaACF",
            "mamba_channel_gate": False,
            "mamba_acf_channel_lag_bank": False,
            "mamba_local_mixer": True,
            "d_model": 32,
            "mamba_local_width": None,
        }
        requested = SimpleNamespace(**{
            **persisted,
            "mamba_channel_gate_rank": 99,
            "mamba_acf_channel_lag_top_k": 7,
            "mamba_local_width": 32,
        })
        left = _mamba_runtime_contract_payload(requested)
        right = _mamba_runtime_contract_payload(persisted)
        self.assertEqual(left["mamba_channel_gate_rank"], None)
        self.assertEqual(left["mamba_acf_channel_lag_top_k"], None)
        self.assertEqual(left["mamba_local_width"], 32)
        self.assertEqual(left, right)

    def test_mamba_runtime_contract_rejects_active_branch_changes(self):
        base = {
            "model": "MambaACF",
            "mamba_use_acf_memory": True,
            "mamba_dynamic_acf": False,
            "mamba_acf_temperature": 4.0,
            "mamba_local_mixer": True,
            "d_model": 32,
            "mamba_local_width": 32,
        }
        persisted = _mamba_runtime_contract_payload(base)
        changed = dict(base, mamba_acf_temperature=2.0)
        self.assertNotEqual(
            _mamba_runtime_contract_payload(changed)["mamba_acf_temperature"],
            persisted["mamba_acf_temperature"],
        )

        dynamic = dict(base, mamba_dynamic_acf=True, mamba_acf_temperature=4.0)
        dynamic_changed = dict(dynamic, mamba_acf_temperature=2.0)
        self.assertNotEqual(
            _mamba_runtime_contract_payload(dynamic)["mamba_acf_temperature"],
            _mamba_runtime_contract_payload(dynamic_changed)[
                "mamba_acf_temperature"
            ],
        )

    def test_mamba_runtime_contract_binds_training_permutation_switches(self):
        base = {
            "model": "MambaACF",
            "mamba_variable_mixer": True,
            "mamba_patch_mixer": True,
            "mamba_variable_permutation": False,
            "mamba_patch_permutation": False,
        }
        baseline = _mamba_runtime_contract_payload(base)
        variable_changed = _mamba_runtime_contract_payload(
            {**base, "mamba_variable_permutation": True}
        )
        patch_changed = _mamba_runtime_contract_payload(
            {**base, "mamba_patch_permutation": True}
        )
        self.assertNotEqual(
            baseline["mamba_variable_permutation"],
            variable_changed["mamba_variable_permutation"],
        )
        self.assertNotEqual(
            baseline["mamba_patch_permutation"],
            patch_changed["mamba_patch_permutation"],
        )

    def test_mamba_runtime_contract_compares_effective_headdim(self):
        # Expanded width 24 resolves both requests to the same valid divisor.
        self.assertEqual(_effective_mamba_headdim(12, 2, 20), 12)
        self.assertEqual(_effective_mamba_headdim(12, 2, 16), 12)
        left = _mamba_runtime_contract_payload(
            {"model": "MambaACF", "d_model": 12, "mamba_expand": 2,
             "mamba_headdim": 20}
        )
        right = _mamba_runtime_contract_payload(
            {"model": "MambaACF", "d_model": 12, "mamba_expand": 2,
             "mamba_headdim": 16}
        )
        self.assertEqual(left["mamba_headdim"], 12)
        self.assertEqual(left, right)

    def test_empty_acf_memory_lag_bank_does_not_require_sidecar(self):
        self.assertFalse(
            _mamba_acf_prior_required(
                {
                    "model": "MambaACF",
                    "mamba_use_acf_memory": True,
                    "mamba_dynamic_acf": False,
                    "mamba_acf_lags": [],
                }
            )
        )

    def test_similarity_channel_order_requires_train_sidecar(self):
        self.assertTrue(
            _mamba_acf_prior_required(
                {
                    "model": "MambaACF",
                    "mamba_variable_mixer": True,
                    "mamba_acf_channel_order": "similarity",
                }
            )
        )

    def test_dormant_repr_fusion_is_not_an_inference_contract(self):
        plain = {
            "model": "MambaACF",
            "AutoCon": False,
            "mamba_acf_repr_fusion": False,
        }
        requested = dict(plain, mamba_acf_repr_fusion=True)
        left = _mamba_runtime_contract_payload(plain)
        right = _mamba_runtime_contract_payload(requested)
        self.assertEqual(left, right)
        self.assertFalse(_mamba_acf_prior_required(plain))
        self.assertFalse(_mamba_acf_prior_required(requested))

        autocon = dict(plain, AutoCon=True)
        self.assertNotEqual(
            _mamba_runtime_contract_payload(autocon),
            _mamba_runtime_contract_payload(
                dict(autocon, mamba_acf_repr_fusion=True)
            ),
        )
        self.assertTrue(
            _mamba_acf_prior_required(
                dict(autocon, mamba_acf_repr_fusion=True)
            )
        )

    def test_dynamic_acf_memory_does_not_require_train_sidecar(self):
        self.assertFalse(
            _mamba_acf_prior_required(
                {
                    "model": "MambaACF",
                    "mamba_use_acf_memory": True,
                    "mamba_dynamic_acf": True,
                    "mamba_acf_channel_order": "input",
                }
            )
        )
        self.assertTrue(
            _mamba_acf_prior_required(
                {
                    "model": "MambaACF",
                    "mamba_use_acf_memory": True,
                    "mamba_dynamic_acf": True,
                    "mamba_acf_relation_loss": True,
                }
            )
        )

    def test_mamba_acf_sidecar_preflight_is_fail_closed(self):
        args = SimpleNamespace(
            model="MambaACF",
            mamba_use_acf_memory=True,
            mamba_dynamic_acf=False,
        )
        self.assertTrue(_mamba_acf_prior_required(args))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "requires acf_prior.npz"):
                _preflight_mamba_acf_sidecar(directory, args, {"model": "MambaACF"})

    def test_mamba_acf_sidecar_preflight_requires_hash(self):
        args = SimpleNamespace(
            model="MambaACF",
            mamba_use_acf_memory=True,
            mamba_dynamic_acf=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "acf_prior.npz"
            path.write_bytes(b"placeholder")
            previous = {
                "model": "MambaACF",
                "mamba_use_acf_memory": True,
                "mamba_acf_prior_artifact": "acf_prior.npz",
            }
            with self.assertRaisesRegex(RuntimeError, "64-hex"):
                _preflight_mamba_acf_sidecar(directory, args, previous)

    def test_compact_setting_preserves_short_names_and_hashes_long_names(self):
        self.assertEqual(compact_setting("short"), "short")
        original = "setting_" + "x" * 400
        first = compact_setting(original)
        second = compact_setting(original)
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 180)
        self.assertIn("_h", first)
        self.assertNotEqual(first, compact_setting(original + "y"))

    def test_experiment_setting_bounds_long_plan_identifiers(self):
        args = SimpleNamespace(
            task_name="long_term_forecast",
            model_id="plan_" + "x" * 300,
            model="MambaACFAutoCon",
            data="traffic",
            features="M",
            seq_len=96,
            label_len=48,
            pred_len=96,
            d_model=32,
            n_heads=4,
            e_layers=1,
            d_layers=1,
            d_ff=64,
            factor=1,
            embed="timeF",
            distil=True,
            des="relation_screen",
            mamba_d_state=16,
            mamba_d_conv=4,
            mamba_expand=2,
            mamba_headdim=16,
            AutoCon=True,
            AutoCon_lambda=0.03,
            AutoCon_acf_source="residual",
        )
        setting = experiment_setting(args, 0, 2021)
        self.assertLessEqual(len(setting), 180)
        self.assertTrue(setting.startswith("long_term_forecast_plan_"))

    def test_mamba_token_stride_runtime_validation_and_architecture_tag(self):
        valid = SimpleNamespace(
            model="MambaACF1", seq_len=96, mamba_token_stride=8,
            mamba_scales=[1, 2],
        )
        validate_mamba_acf_arguments(valid)

        no_overlap = SimpleNamespace(
            model="MambaACF1AutoCon", seq_len=96, pred_len=96,
            mamba_token_stride=1, mamba_scales=[1, 2],
            AutoCon=True, mamba_acf_relation_loss=True,
            mamba_acf_relation_lag_mask=True,
            mamba_acf_relation_min_lag=96,
        )
        validate_mamba_acf_arguments(no_overlap)

        # Local relation labels use the original time grid.  A strided
        # representation is only safe for the window-level global objective.
        with self.assertRaisesRegex(ValueError, "token_stride=8.*relation_scope=local"):
            validate_mamba_acf_arguments(SimpleNamespace(
                model="MambaACFAutoCon", seq_len=96, pred_len=96,
                mamba_token_stride=8, mamba_scales=[1],
                AutoCon=True, mamba_acf_relation_loss=True,
                mamba_acf_relation_scope="local",
            ))
        validate_mamba_acf_arguments(SimpleNamespace(
            model="MambaACFAutoCon", seq_len=96, pred_len=96,
            mamba_token_stride=8, mamba_scales=[1],
            AutoCon=True, mamba_acf_relation_loss=True,
            mamba_acf_relation_scope="global",
        ))

        legacy = SimpleNamespace(
            model="MambaACF1", seq_len=96, mamba_token_stride=1,
            mamba_scales=[1, 2],
        )
        self.assertNotEqual(mamba_architecture_tag(valid), mamba_architecture_tag(legacy))

        with self.assertRaisesRegex(ValueError, "supported only by MambaACF"):
            validate_mamba_acf_arguments(SimpleNamespace(
                model="Mamba2", seq_len=96, mamba_token_stride=8,
                mamba_scales=[1, 2],
            ))
        with self.assertRaisesRegex(ValueError, "base tokens"):
            validate_mamba_acf_arguments(SimpleNamespace(
                model="MambaACF1", seq_len=96, mamba_token_stride=96,
                mamba_scales=[1, 2],
            ))
        with self.assertRaisesRegex(ValueError, "positive integers"):
            validate_mamba_acf_arguments(SimpleNamespace(
                model="MambaACF1", seq_len=96, mamba_token_stride=1,
                mamba_scales=[1, 0],
            ))
        # The minimum relation lag is an independent overlap control; it does
        # not require the selected-positive mask to be enabled.
        validate_mamba_acf_arguments(SimpleNamespace(
            model="MambaACF1AutoCon", seq_len=96,
            mamba_token_stride=1, mamba_scales=[1, 2],
            AutoCon=True, mamba_acf_relation_loss=True,
            mamba_acf_relation_min_lag=96,
            mamba_acf_relation_lag_mask=False,
        ))
        validate_mamba_acf_arguments(SimpleNamespace(
            model="MambaACF1AutoCon", seq_len=96,
            mamba_token_stride=1, mamba_scales=[1, 2],
            AutoCon=True, mamba_acf_relation_loss=True,
            mamba_acf_relation_target_mode="soft_multi_peak",
            mamba_acf_relation_soft_min_abs=0.25,
            mamba_acf_relation_selected_lag_boost=0.5,
        ))
        with self.assertRaisesRegex(ValueError, "relation_min_lag cannot exceed"):
            validate_mamba_acf_arguments(SimpleNamespace(
                model="MambaACF1AutoCon", seq_len=96,
                mamba_token_stride=1, mamba_scales=[1, 2],
                AutoCon=True,
                mamba_acf_relation_loss=True,
                mamba_acf_relation_min_lag=721,
                mamba_acf_max_lag=720,
            ))
        with self.assertRaisesRegex(ValueError, "target_mode"):
            validate_mamba_acf_arguments(SimpleNamespace(
                model="MambaACF1AutoCon", seq_len=96,
                mamba_token_stride=1, mamba_scales=[1, 2],
                AutoCon=True, mamba_acf_relation_loss=True,
                mamba_acf_relation_target_mode="unknown",
            ))
        with self.assertRaisesRegex(ValueError, "effective temporal tokens"):
            validate_mamba_acf_arguments(SimpleNamespace(
                model="MambaACF1", seq_len=96, pred_len=96,
                mamba_token_stride=8, mamba_scales=[1],
                mamba_low_rank_head=True, mamba_temporal_rank=12,
            ))

    def test_acf_scale_conditioning_requires_multiple_mamba_scales(self):
        with self.assertRaisesRegex(ValueError, "at least two temporal scales"):
            validate_mamba_acf_arguments(SimpleNamespace(
                model="MambaACF", seq_len=96, pred_len=96,
                mamba_scales=[1], mamba_acf_scale_conditioning=True,
                AutoCon_acf_source="raw",
            ))
        with self.assertRaisesRegex(ValueError, "only by MambaACF"):
            validate_mamba_acf_arguments(SimpleNamespace(
                model="Mamba2", seq_len=96, mamba_scales=[1, 2],
                mamba_acf_scale_conditioning=True,
            ))
        with self.assertRaisesRegex(ValueError, "AutoCon_acf_source=raw"):
            validate_mamba_acf_arguments(SimpleNamespace(
                model="MambaACF", seq_len=96, pred_len=96,
                mamba_scales=[1, 2], mamba_acf_scale_conditioning=True,
                AutoCon_acf_source="residual",
            ))
        with self.assertRaisesRegex(ValueError, "16 or more window pairs"):
            validate_mamba_acf_arguments(SimpleNamespace(
                model="MambaACF", seq_len=96, pred_len=96,
                mamba_scales=[1, 2], mamba_acf_scale_conditioning=True,
                AutoCon_acf_source="raw", mamba_acf_lags=[168, 336],
            ))

    def test_mamba_acf_guards_resolve_lowercase_source_alias(self):
        base = dict(
            model="MambaACF", seq_len=96, pred_len=96,
            mamba_scales=[1, 2], mamba_acf_scale_conditioning=True,
            autocon_acf_source="raw", AutoCon_acf_source="residual",
            mamba_acf_lags=[24],
        )
        validate_mamba_acf_arguments(SimpleNamespace(**base))

    def test_strict_representation_causality_runtime_guard(self):
        base = dict(
            model="MambaACFAutoCon", seq_len=96, pred_len=96,
            AutoCon=True, mamba_acf_relation_loss=True,
            mamba_causal_embedding=True, mamba_bidirectional=False,
            mamba_token_stride=1, mamba_scales=[1],
            mamba_channel_gate=False, mamba_acf_scale_conditioning=False,
            mamba_local_mixer=False, mamba_acf_repr_fusion=False,
            autocon_acf_source="raw",
            mamba_require_strict_representation_causality=True,
        )
        with self.assertRaisesRegex(ValueError, "window_full_normalization_forbidden"):
            validate_mamba_acf_arguments(SimpleNamespace(**base))
        with self.assertRaisesRegex(ValueError, "mamba_scales_must_be_single_scale_one"):
            validate_mamba_acf_arguments(SimpleNamespace(
                **{**base, "mamba_scales": [1, 2]}
            ))
        with self.assertRaisesRegex(ValueError, "mamba_causal_embedding_required"):
            validate_mamba_acf_arguments(SimpleNamespace(
                **{**base, "mamba_causal_embedding": False}
            ))
        with self.assertRaisesRegex(ValueError, "supported only by MambaACF"):
            validate_mamba_acf_arguments(SimpleNamespace(
                **{
                    **base,
                    "model": "Mamba2",
                    "AutoCon": False,
                    "mamba_acf_relation_loss": False,
                }
            ))

    def test_channel_lag_bank_runtime_guard_and_architecture_identity(self):
        base = dict(
            model="MambaACF", seq_len=96, pred_len=96,
            mamba_acf_channel_lag_bank=True,
            mamba_acf_channel_lag_top_k=2,
            mamba_acf_lags=[24, 168, 336],
            mamba_use_acf_memory=True, mamba_dynamic_acf=False,
        )
        validate_mamba_acf_arguments(SimpleNamespace(**base))
        self.assertNotEqual(
            mamba_architecture_tag(SimpleNamespace(**base)),
            mamba_architecture_tag(SimpleNamespace(
                **{**base, "mamba_acf_channel_lag_bank": False}
            )),
        )
        with self.assertRaisesRegex(ValueError, "requires mamba_use_acf_memory=true"):
            validate_mamba_acf_arguments(SimpleNamespace(
                **{**base, "mamba_use_acf_memory": False}
            ))
        with self.assertRaisesRegex(ValueError, "requires static train-only ACF memory"):
            validate_mamba_acf_arguments(SimpleNamespace(
                **{**base, "mamba_dynamic_acf": True}
            ))
        with self.assertRaisesRegex(ValueError, "cannot exceed mamba_acf_lags count"):
            validate_mamba_acf_arguments(SimpleNamespace(
                **{**base, "mamba_acf_channel_lag_top_k": 4}
            ))
        with self.assertRaisesRegex(ValueError, "positive integer"):
            validate_mamba_acf_arguments(SimpleNamespace(
                **{**base, "mamba_acf_channel_lag_top_k": 1.5}
            ))
        with self.assertRaisesRegex(ValueError, "supported only by MambaACF"):
            validate_mamba_acf_arguments(SimpleNamespace(
                **{**base, "model": "Mamba2"}
            ))

    def test_relation_runtime_guard_rejects_odd_batch_and_nonfinite_options(self):
        base = dict(
            model="MambaACFAutoCon", seq_len=96, pred_len=96,
            AutoCon=True, mamba_acf_relation_loss=True,
            mamba_acf_relation_batching="lag_pairs", batch_size=32,
        )
        validate_mamba_acf_arguments(SimpleNamespace(**base))
        with self.assertRaisesRegex(ValueError, "even"):
            validate_mamba_acf_arguments(SimpleNamespace(
                **{**base, "batch_size": 31}
            ))
        with self.assertRaisesRegex(ValueError, "AutoCon_temperature"):
            validate_mamba_acf_arguments(SimpleNamespace(
                **{**base, "AutoCon_temperature": float("nan")}
            ))

    def test_mamba_shared_head_shape_contract_is_explicit(self):
        common = dict(
            model="Mamba2", seq_len=96, pred_len=96,
            enc_in=7, c_out=7, features="M", mamba_scales=[1, 2],
        )
        validate_mamba_acf_arguments(SimpleNamespace(**common))
        with self.assertRaisesRegex(ValueError, "requires c_out=enc_in"):
            validate_mamba_acf_arguments(SimpleNamespace(
                **{**common, "c_out": 1}
            ))
        # MS is the documented legacy exception: the trainer slices the last
        # returned channel after the shared head.
        validate_mamba_acf_arguments(SimpleNamespace(
            **{**common, "features": "MS", "c_out": 1}
        ))
        with self.assertRaisesRegex(ValueError, "features=MS requires c_out=1"):
            validate_mamba_acf_arguments(SimpleNamespace(
                **{**common, "features": "MS", "c_out": 2}
            ))

    def test_runtime_rejects_malformed_acf_fit_controls(self):
        base = dict(
            model="MambaACF", seq_len=96, pred_len=96,
            mamba_scales=[1, 2],
        )
        invalid = (
            ("mamba_acf_max_lag", 1.5),
            ("mamba_acf_top_k", True),
            ("mamba_acf_min_lag", float("nan")),
            ("mamba_acf_min_abs", float("nan")),
            ("mamba_acf_confidence_z", float("inf")),
            ("mamba_acf_chunk", 2.5),
            ("mamba_acf_relation_lag_tolerance", 1.5),
            ("mamba_acf_relation_min_lag", np.bool_(True)),
            ("mamba_scales", [1, 2.5]),
        )
        for name, value in invalid:
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_mamba_acf_arguments(SimpleNamespace(
                    **{**base, name: value}
                ))

    def test_channel_lag_bank_metadata_matches_float32_model_tie_break(self):
        # The first and third scores are equal only after float32 arithmetic:
        # stable ranking must select the lower canonical lag in both the model
        # compiler and the serialized metadata path.
        compact = np.asarray(
            [[0.61787856, 0.01, 0.20595953]], dtype=np.float32
        )
        support = np.asarray([[1, 0, 0]], dtype=np.uint8)
        model = SimpleNamespace(
            acf_memory_static_channel_support=torch.as_tensor(
                support, dtype=torch.bool
            )
        )
        args = SimpleNamespace(
            mamba_acf_channel_lag_bank=True,
            mamba_acf_channel_lag_top_k=1,
            seq_len=8,
            pred_len=3,
        )

        _bind_acf_channel_lag_bank_metadata(
            args, compact, np.asarray([1, 2, 3], dtype=np.int64), model
        )

        self.assertEqual(args.mamba_acf_channel_lag_bank_resolved_lags, [1, 2, 3])
        self.assertEqual(args.mamba_acf_channel_lag_bank_legal_lags, [1, 2, 3])
        self.assertEqual(args.mamba_acf_channel_lag_bank_support_shape, [1, 3])
        self.assertEqual(args.mamba_acf_channel_lag_bank_selected_slots, 1)
        self.assertEqual(
            args.mamba_acf_channel_lag_bank_support_sha256,
            hashlib.sha256(support.tobytes(order="C")).hexdigest(),
        )

    def test_reliability_interpolation_requires_static_raw_memory(self):
        base = dict(
            model="MambaACF", seq_len=96, pred_len=96,
            mamba_acf_reliability_interpolation=True,
            mamba_use_acf_memory=True, mamba_dynamic_acf=False,
            mamba_acf_step_reliability=False, AutoCon_acf_source="raw",
        )
        validate_mamba_acf_arguments(SimpleNamespace(**base))
        with self.assertRaisesRegex(ValueError, "mamba_use_acf_memory=true"):
            validate_mamba_acf_arguments(SimpleNamespace(
                **{**base, "mamba_use_acf_memory": False}
            ))
        with self.assertRaisesRegex(ValueError, "static ACF memory"):
            validate_mamba_acf_arguments(SimpleNamespace(
                **{**base, "mamba_dynamic_acf": True}
            ))
        with self.assertRaisesRegex(ValueError, "AutoCon_acf_source=raw"):
            validate_mamba_acf_arguments(SimpleNamespace(
                **{**base, "AutoCon_acf_source": "residual"}
            ))
        with self.assertRaisesRegex(ValueError, "step_reliability"):
            validate_mamba_acf_arguments(SimpleNamespace(
                **{**base, "mamba_acf_step_reliability": True}
            ))
        with self.assertRaisesRegex(ValueError, "only by MambaACF"):
            validate_mamba_acf_arguments(SimpleNamespace(
                **{**base, "model": "Mamba2"}
            ))

    def test_replica_seed_is_distinguished_from_base_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            args = SimpleNamespace(seed=2021, plan_job_id=None)
            persist_run_config(str(output), args, 2023, "replica")
            payload = json.loads((output / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["base_seed"], 2021)
            self.assertEqual(payload["seed"], 2023)
            self.assertEqual(payload["run_seed"], 2023)

    def test_persist_config_omits_sidecar_construction_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                seed=1,
                plan_job_id=None,
                mamba_acf_prior_artifact_path="/machine/specific/acf_prior.npz",
                mamba_acf_prior_expected_sha256="abc",
            )
            persist_run_config(directory, args, 1, "portable")
            payload = json.loads((Path(directory) / "config.json").read_text())
            self.assertNotIn("mamba_acf_prior_artifact_path", payload)
            self.assertNotIn("mamba_acf_prior_expected_sha256", payload)

    def test_restore_acf_metadata_rejects_sidecar_lag_mismatch(self):
        args = SimpleNamespace(mamba_acf_prior_lags=[3, 7])
        previous = {
            "mamba_acf_effective_lags": [3, 9],
            "mamba_acf_train_peak_lags": [3, 9, 12],
        }
        with self.assertRaisesRegex(RuntimeError, "sidecar lags"):
            restore_acf_resolution_metadata(args, previous)

    def test_restore_acf_metadata_preserves_peak_provenance(self):
        args = SimpleNamespace(mamba_acf_prior_lags=[3, 7])
        previous = {
            "mamba_acf_effective_lags": [3, 7],
            "mamba_acf_train_peak_lags": [3, 7, 12],
            "mamba_acf_effective_lag_source": "train_peaks_available",
        }
        restore_acf_resolution_metadata(args, previous)
        self.assertEqual(args.mamba_acf_train_peak_lags, [3, 7, 12])
        self.assertEqual(args.mamba_acf_effective_lag_source, "train_peaks_available")

    def test_preserve_acf_metadata_keeps_sidecar_for_acf_off_replay(self):
        args = SimpleNamespace(
            mamba_acf_prior_artifact="cli-placeholder",
            mamba_acf_prior_sha256="cli-hash",
            mamba_acf_effective_lags=[1],
            unrelated_runtime_flag=True,
        )
        previous = {
            "mamba_acf_prior_artifact": "acf_prior.npz",
            "mamba_acf_prior_sha256": "a" * 64,
            "mamba_acf_prior_lags": [24, 168],
            "mamba_acf_effective_lags": [24, 168],
            "mamba_acf_train_peak_lags": [24, 168, 336],
        }
        preserve_acf_prior_metadata(args, previous)
        self.assertEqual(args.mamba_acf_prior_artifact, "acf_prior.npz")
        self.assertEqual(args.mamba_acf_prior_sha256, "a" * 64)
        self.assertEqual(args.mamba_acf_prior_lags, [24, 168])
        self.assertEqual(args.mamba_acf_effective_lags, [24, 168])
        self.assertEqual(args.mamba_acf_train_peak_lags, [24, 168, 336])
        self.assertTrue(args.unrelated_runtime_flag)

    def test_clear_acf_metadata_removes_derived_sidecar_and_resolution_fields(self):
        args = SimpleNamespace(
            mamba_acf_prior_artifact="acf_prior.npz",
            mamba_acf_prior_lags=[24],
            mamba_acf_effective_lags=[24],
            mamba_acf_train_peak_lags=[24, 168],
            unrelated_runtime_flag=True,
        )
        clear_acf_prior_metadata(args)
        self.assertFalse(hasattr(args, "mamba_acf_prior_artifact"))
        self.assertFalse(hasattr(args, "mamba_acf_prior_lags"))
        self.assertFalse(hasattr(args, "mamba_acf_effective_lags"))
        self.assertFalse(hasattr(args, "mamba_acf_train_peak_lags"))
        self.assertTrue(args.unrelated_runtime_flag)

    def test_acf_off_eval_clears_inherited_sidecar_and_never_loads_it(self):
        args = SimpleNamespace(
            mamba_acf_prior_artifact_path="/stale/acf_prior.npz",
            mamba_acf_prior_expected_sha256="a" * 64,
            mamba_acf_prior_artifact="acf_prior.npz",
            mamba_acf_prior_lags=[24],
        )
        prepare_eval_acf_prior_args(args, prior_required=False)
        self.assertFalse(hasattr(args, "mamba_acf_prior_artifact_path"))
        self.assertFalse(hasattr(args, "mamba_acf_prior_expected_sha256"))
        self.assertFalse(hasattr(args, "mamba_acf_prior_artifact"))
        self.assertFalse(hasattr(args, "mamba_acf_prior_lags"))
        with patch("run.load_acf_prior_artifact") as loader:
            self.assertFalse(
                load_eval_acf_prior_if_required(
                    "/unused", args, object(), prior_required=False
                )
            )
        loader.assert_not_called()

    def test_required_eval_restores_and_loads_preflight_sidecar(self):
        args = SimpleNamespace()
        prepare_eval_acf_prior_args(
            args,
            prior_required=True,
            preflight_artifact_path="/validated/acf_prior.npz",
            expected_sha="b" * 64,
        )
        self.assertEqual(
            args.mamba_acf_prior_artifact_path, "/validated/acf_prior.npz"
        )
        self.assertEqual(args.mamba_acf_prior_expected_sha256, "b" * 64)
        experiment = object()
        with patch("run.load_acf_prior_artifact", return_value=True) as loader:
            self.assertTrue(
                load_eval_acf_prior_if_required(
                    "/metadata",
                    args,
                    experiment,
                    expected_sha="b" * 64,
                    prior_required=True,
                )
            )
        loader.assert_called_once_with(
            "/metadata", args, experiment, expected_sha="b" * 64
        )

    def test_required_eval_drops_unvalidated_inherited_sidecar_path(self):
        args = SimpleNamespace(
            mamba_acf_prior_artifact_path="/stale/acf_prior.npz",
            mamba_acf_prior_expected_sha256="c" * 64,
        )
        prepare_eval_acf_prior_args(args, prior_required=True)
        self.assertFalse(hasattr(args, "mamba_acf_prior_artifact_path"))
        self.assertFalse(hasattr(args, "mamba_acf_prior_expected_sha256"))

    def test_required_eval_does_not_load_sidecar_twice_after_experiment_attach(self):
        class FakeModel:
            def load_acf_prior(self, _path):
                raise AssertionError("an already attached sidecar must not reload")

            def export_acf_prior_state(self):
                return {
                    "acf_values": np.asarray([[0.2, -0.3]], dtype=np.float32),
                    "lags": np.asarray([3, 9], dtype=np.int64),
                }

        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "acf_prior.npz"
            artifact.write_bytes(b"already-mounted-sidecar")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            args = SimpleNamespace()
            experiment = SimpleNamespace(
                _acf_prior_attached=True,
                model=FakeModel(),
            )
            with patch("run.load_acf_prior_artifact") as loader:
                self.assertTrue(
                    load_eval_acf_prior_if_required(
                        directory,
                        args,
                        experiment,
                        expected_sha=digest,
                        prior_required=True,
                    )
                )
            loader.assert_not_called()
            self.assertEqual(args.mamba_acf_prior_artifact, "acf_prior.npz")
            self.assertEqual(args.mamba_acf_prior_sha256, digest)
            self.assertEqual(args.mamba_acf_prior_shape, [1, 2])
            self.assertEqual(args.mamba_acf_prior_lags, [3, 9])

    def test_attached_sidecar_metadata_binding_rejects_hash_drift(self):
        class FakeModel:
            def load_acf_prior(self, _path):
                raise AssertionError("the attached model should not reload")

            def export_acf_prior_state(self):
                return {
                    "acf_values": np.asarray([[0.2]], dtype=np.float32),
                    "lags": np.asarray([3], dtype=np.int64),
                }

        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "acf_prior.npz").write_bytes(b"sidecar")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                load_eval_acf_prior_if_required(
                    directory,
                    SimpleNamespace(),
                    SimpleNamespace(_acf_prior_attached=True, model=FakeModel()),
                    expected_sha="e" * 64,
                    prior_required=True,
                )

    def test_restore_acf_metadata_accepts_sidecar_lags_in_canonical_order(self):
        args = SimpleNamespace(mamba_acf_prior_lags=[72, 96, 120])
        previous = {
            # The availability selector records ranking order; the sidecar
            # stores columns in canonical ascending lag order.
            "mamba_acf_effective_lags": [96, 120, 72],
            "mamba_acf_train_peak_lags": [120, 72, 96],
            "mamba_acf_relation_effective_lags": [120, 96],
            "mamba_acf_relation_selected_lags": [120, 96],
        }
        restore_acf_resolution_metadata(args, previous)
        self.assertEqual(args.mamba_acf_effective_lags, [72, 96, 120])
        self.assertEqual(args.mamba_acf_train_peak_lags, [72, 96, 120])
        self.assertEqual(args.mamba_acf_relation_effective_lags, [96, 120])
        self.assertEqual(args.mamba_acf_relation_selected_lags, [96, 120])

    def test_restore_acf_metadata_rejects_duplicate_previous_lags(self):
        args = SimpleNamespace(mamba_acf_prior_lags=[72, 96])
        previous = {"mamba_acf_effective_lags": [72, 72]}
        with self.assertRaisesRegex(RuntimeError, "unique positive integers"):
            restore_acf_resolution_metadata(args, previous)

    def test_persisted_channel_lag_bank_must_match_before_model_construction(self):
        previous = {
            "model": "MambaACF",
            "mamba_acf_channel_lag_bank": True,
            "mamba_acf_channel_lag_top_k": 3,
        }
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            validate_persisted_acf_runtime_contract(
                SimpleNamespace(
                    model="MambaACF",
                    mamba_acf_channel_lag_bank=False,
                    mamba_acf_channel_lag_top_k=3,
                ),
                previous,
            )
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            validate_persisted_acf_runtime_contract(
                SimpleNamespace(
                    model="MambaACF",
                    mamba_acf_channel_lag_bank=True,
                    mamba_acf_channel_lag_top_k=2,
                ),
                previous,
            )

        # The reverse direction is equally unsafe: an old dense-bank run must
        # not be evaluated with a newly enabled sparse mask.
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            validate_persisted_acf_runtime_contract(
                SimpleNamespace(
                    model="MambaACF",
                    mamba_acf_channel_lag_bank=True,
                    mamba_acf_channel_lag_top_k=2,
                ),
                {"model": "MambaACF", "mamba_acf_channel_lag_bank": False},
            )

    def test_serialized_channel_lag_bank_booleans_restore_consistently(self):
        for persisted, requested in ((0, False), ("false", False), (1, True), ("true", True)):
            with self.subTest(persisted=persisted):
                validate_persisted_acf_runtime_contract(
                    SimpleNamespace(
                        model="MambaACF",
                        mamba_acf_channel_lag_bank=requested,
                    ),
                    {
                        "model": "MambaACF",
                        "mamba_acf_channel_lag_bank": persisted,
                    },
                )

        args = SimpleNamespace(
            mamba_acf_prior_lags=[24, 72],
            mamba_acf_channel_lag_bank=False,
        )
        restore_acf_resolution_metadata(
            args,
            {
                "mamba_acf_effective_lags": [24, 72],
                "mamba_acf_channel_lag_bank": "false",
            },
        )

    def test_persisted_mamba_model_identity_rejects_wrong_cli_model(self):
        with self.assertRaisesRegex(RuntimeError, "does not match the persisted Mamba model"):
            validate_persisted_acf_runtime_contract(
                SimpleNamespace(model="PatchTST"),
                {"model": "Mamba2"},
            )
        with self.assertRaisesRegex(RuntimeError, "does not match the persisted Mamba model"):
            validate_persisted_acf_runtime_contract(
                SimpleNamespace(model="Mamba1"),
                {"model": "Mamba2"},
            )

    def test_restore_acf_metadata_rejects_channel_lag_bank_mismatch(self):
        args = SimpleNamespace(
            mamba_acf_prior_lags=[24, 72],
            mamba_acf_channel_lag_bank=False,
        )
        previous = {
            "mamba_acf_effective_lags": [24, 72],
            "mamba_acf_channel_lag_bank": True,
        }
        with self.assertRaisesRegex(RuntimeError, "lag-bank"):
            restore_acf_resolution_metadata(args, previous)

    def test_acf_prior_sidecar_is_saved_and_hash_bound(self):
        class FakeModel:
            def export_acf_prior_state(self):
                import numpy as np
                return {
                    "acf_values": np.asarray([[0.2, -0.3]], dtype=np.float32),
                    "lags": np.asarray([3, 9], dtype=np.int64),
                }

            def save_acf_prior(self, path):
                import numpy as np
                np.savez_compressed(
                    path,
                    schema_version=np.asarray(1),
                    acf_values=np.asarray([[0.2, -0.3]], dtype=np.float32),
                    lags=np.asarray([3, 9], dtype=np.int64),
                )
                return True

            def load_acf_prior(self, path):
                return True

        class FakeExperiment:
            model = FakeModel()

        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace()
            self.assertTrue(persist_acf_prior_artifact(directory, args, FakeExperiment()))
            self.assertEqual(args.mamba_acf_prior_artifact, "acf_prior.npz")
            self.assertTrue(
                load_acf_prior_artifact(
                    directory, args, FakeExperiment(), args.mamba_acf_prior_sha256
                )
            )
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                load_acf_prior_artifact(directory, args, FakeExperiment(), "0" * 64)


if __name__ == "__main__":
    unittest.main()
