"""CUDA contract tests for the proposed shared MambaACF forecaster."""

from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import torch
import numpy as np

from layers.losses import ACFRelationLoss
from models.MambaACF import Model, _TemporalProjection
from models.MambaACF1 import Model as MambaACF1Model


class TemporalProjectionTests(unittest.TestCase):
    """CPU-only checks for the optional rank-constrained forecast head."""

    def test_low_rank_projection_shape_and_parameter_reduction(self):
        input_length, output_length, rank = 96, 720, 16
        full = _TemporalProjection(input_length, output_length)
        low_rank = _TemporalProjection(input_length, output_length, rank)
        x = torch.randn(4, input_length, requires_grad=True)
        y = low_rank(x)
        self.assertEqual(tuple(y.shape), (4, output_length))
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(y).all())
        full_parameters = sum(parameter.numel() for parameter in full.parameters())
        low_rank_parameters = sum(parameter.numel() for parameter in low_rank.parameters())
        self.assertLess(low_rank_parameters, full_parameters)
        self.assertEqual(low_rank.rank, rank)
        self.assertTrue(low_rank.is_low_rank)

    def test_rank_must_be_strictly_below_both_lengths(self):
        with self.assertRaises(ValueError):
            _TemporalProjection(24, 96, rank=24)
        with self.assertRaises(ValueError):
            _TemporalProjection(24, 96, rank=0)


def _args(channels=3, c_out=None, *, channel_independent=False, pred_len=24):
    return SimpleNamespace(
        seq_len=96,
        label_len=48,
        pred_len=pred_len,
        enc_in=channels,
        c_out=channels if c_out is None else c_out,
        d_model=16,
        d_ff=32,
        e_layers=1,
        mamba_d_state=8,
        mamba_d_conv=2,
        mamba_expand=2,
        mamba_headdim=8,
        AutoCon=True,
        AutoCon_wnorm="LastVal",
        AutoCon_acf_source="raw",
        embed="timeF",
        freq="h",
        wo_freq="",
        dropout=0.0,
        mamba_scales=[1],
        mamba_token_stride=1,
        mamba_bidirectional=True,
        mamba_channel_independent=channel_independent,
        mamba_channel_gate=False,
        mamba_channel_gate_rank=8,
        mamba_data_dependent_channel_mixer=False,
        mamba_data_dependent_channel_mixer_rank=8,
        mamba_acf_scale_conditioning=False,
        mamba_acf_channel_gate=False,
        mamba_causal_embedding=False,
        mamba_variable_mixer=False,
        mamba_variable_gdd_mixer=False,
        mamba_variable_gdd_mixer_rank=8,
        mamba_variable_gdd_init_scale=0.0,
        mamba_variable_mixer_depth=1,
        mamba_variable_group_size=1,
        mamba_variable_permutation=False,
        mamba_acf_channel_order="input",
        mamba_acf_channel_conditioning=False,
        mamba_acf_repr_fusion=False,
        mamba_variable_repr_fusion=False,
        mamba_patch_mixer=False,
        mamba_patch_mixer_depth=1,
        mamba_patch_len=16,
        mamba_patch_stride=8,
        mamba_patch_order="time_major",
        mamba_patch_permutation=False,
        mamba_local_mixer=False,
        mamba_local_patch_len=16,
        mamba_local_patch_stride=8,
        mamba_local_width=16,
        mamba_local_rank=8,
        mamba_local_depth=1,
        mamba_low_rank_head=False,
        mamba_temporal_rank=16,
        mamba_acf_lags=[24, 168, 336],
        mamba_use_acf_memory=True,
        mamba_acf_temperature=4.0,
        mamba_acf_step_reliability=False,
        mamba_acf_reliability_interpolation=False,
        mamba_acf_horizon_calibration=False,
        mamba_acf_horizon_calibration_rank=2,
        mamba_acf_memory_gate_mode="hidden",
        mamba_acf_anchor_copy=False,
        mamba_dynamic_acf=False,
    )


class MambaACFStructuralTests(unittest.TestCase):
    """CPU-only checks for channel fairness metadata and parameter scaling."""

    def test_variable_repr_fusion_is_narrowly_gated_and_profiled(self):
        args = _args(channels=4, pred_len=12)
        args.mamba_variable_mixer = True
        args.mamba_variable_repr_fusion = True
        model = Model(args)
        profile = model.resource_profile()
        self.assertTrue(profile["variable_repr_fusion_requested"])
        self.assertTrue(profile["variable_repr_fusion_enabled"])
        self.assertTrue(profile["variable_repr_fusion_path_active"])
        self.assertEqual(
            profile["variable_repr_fusion_input_shape"], "B,C,D"
        )
        self.assertIn(
            "window_pooled_variable_token_representation_context",
            profile["representation_grid_causality_limits"],
        )
        self.assertGreater(profile["variable_repr_fusion_parameters"], 0)
        self.assertGreater(
            profile["variable_repr_fusion_macs_per_batch_item"], 0
        )

        with self.assertRaisesRegex(ValueError, "requires AutoCon"):
            plain = _args(channels=4, pred_len=12)
            plain.AutoCon = False
            plain.mamba_variable_mixer = True
            plain.mamba_variable_repr_fusion = True
            Model(plain)
        with self.assertRaisesRegex(ValueError, "requires mamba_variable_mixer"):
            missing_branch = _args(channels=4, pred_len=12)
            missing_branch.mamba_variable_repr_fusion = True
            Model(missing_branch)

    def test_variable_repr_fusion_routes_relation_gradient_to_variable_tokens(self):
        args = _args(channels=3, pred_len=12)
        args.mamba_variable_mixer = True
        args.mamba_variable_repr_fusion = True
        model = Model(args)
        model.variable_repr_scale.data.fill_(0.5)
        values = torch.randn(2, args.seq_len, args.enc_in)
        hidden = torch.randn(2, args.seq_len, args.d_model, requires_grad=True)
        # Avoid depending on a CPU Mamba extension in this graph-contract
        # test; the real branch is exercised by the CUDA smoke suite.
        with patch.object(model, "_run_variable_branch", side_effect=lambda tokens: tokens):
            tokens = model._variable_token_features(values)
        fused = model._fuse_variable_representation(hidden, tokens)
        self.assertEqual(tuple(fused.shape), tuple(hidden.shape))
        fused.square().mean().backward()
        self.assertIsNotNone(model.variable_embedding.weight.grad)
        self.assertGreater(float(model.variable_embedding.weight.grad.abs().sum()), 0.0)
        self.assertIsNotNone(model.variable_repr_projection[0].weight.grad)

    def test_variable_gdd_mixer_is_channel_preserving_and_profiled(self):
        args = _args(channels=5, pred_len=12)
        args.mamba_variable_mixer = True
        args.mamba_variable_gdd_mixer = True
        args.mamba_variable_gdd_mixer_rank = 99
        model = Model(args)
        profile = model.resource_profile()
        self.assertTrue(profile["variable_gdd_mixer_enabled"])
        self.assertTrue(profile["variable_gdd_mixer_path_active"])
        # The effective bottleneck is capped below the channel count.
        self.assertEqual(profile["variable_gdd_mixer_rank"], 4)
        self.assertEqual(profile["variable_gdd_mixer_input_shape"], "B,C,1,D")
        self.assertEqual(profile["variable_gdd_mixer_output_shape"], "B,C,D")
        self.assertTrue(profile["variable_gdd_mixer_channel_preserving"])
        self.assertFalse(profile["variable_gdd_mixer_dense_channel_matrix"])
        self.assertGreater(profile["variable_gdd_mixer_parameters"], 0)
        self.assertGreater(profile["variable_gdd_mixer_macs_per_batch_item"], 0)
        self.assertIn("variable_token_gdd", profile["channel_preserving_forecast_branches"])

    def test_variable_gdd_initial_scale_is_explicit_and_profiled(self):
        args = _args(channels=5, pred_len=12)
        args.mamba_variable_mixer = True
        args.mamba_variable_gdd_mixer = True
        args.mamba_variable_gdd_init_scale = 0.01
        model = Model(args)
        self.assertAlmostEqual(
            float(model.variable_gdd_mixer.residual_scale.detach()), 0.01, places=7
        )
        self.assertAlmostEqual(
            model.resource_profile()["variable_gdd_mixer_init_scale"], 0.01, places=7
        )

    def test_variable_gdd_mixer_state_is_shared_with_autocon_representation(self):
        args = _args(channels=4, pred_len=12)
        args.mamba_variable_mixer = True
        args.mamba_variable_gdd_mixer = True
        args.mamba_variable_repr_fusion = True
        model = Model(args)
        model.train()
        # Make the optional ReZero path visible for this gradient contract.
        with torch.no_grad():
            model.variable_gdd_mixer.residual_scale.fill_(0.5)
        values = torch.randn(2, args.seq_len, args.enc_in)
        hidden = torch.randn(2, args.seq_len, args.d_model, requires_grad=True)
        with patch.object(model, "_run_variable_branch", side_effect=lambda tokens: tokens):
            tokens = model._variable_token_features(values)
        fused = model._fuse_variable_representation(hidden, tokens)
        fused.square().mean().backward()
        self.assertIsNotNone(model.variable_gdd_mixer.residual_scale.grad)
        self.assertGreater(
            float(model.variable_gdd_mixer.residual_scale.grad.abs().sum()), 0.0
        )
        self.assertIsNotNone(model.variable_gdd_mixer.summary_to_latent.weight.grad)
        self.assertTrue(profile := model.resource_profile())
        self.assertTrue(profile["variable_gdd_mixer_autocon_shared_state"])

    def test_variable_gdd_mixer_requires_variable_branch(self):
        args = _args(channels=4, pred_len=12)
        args.mamba_variable_gdd_mixer = True
        with self.assertRaisesRegex(ValueError, "requires mamba_variable_mixer"):
            Model(args)

    def test_mamba1_variable_gdd_mixer_uses_same_contract(self):
        args = _args(channels=4, pred_len=12)
        args.mamba_variable_mixer = True
        args.mamba_variable_gdd_mixer = True
        model = MambaACF1Model(args)
        profile = model.resource_profile()
        self.assertTrue(profile["variable_gdd_mixer_path_active"])
        self.assertEqual(profile["variable_gdd_mixer_output_shape"], "B,C,D")

    def test_variable_repr_fusion_does_not_perturb_shared_initialization(self):
        """A paired off/on screen must start from identical common weights."""

        original_rng_state = torch.random.get_rng_state()
        try:
            torch.manual_seed(20260903)
            off_args = _args(channels=3, pred_len=12)
            off_args.mamba_variable_mixer = True
            off = Model(off_args)

            torch.manual_seed(20260903)
            on_args = _args(channels=3, pred_len=12)
            on_args.mamba_variable_mixer = True
            on_args.mamba_variable_repr_fusion = True
            on = Model(on_args)

            off_state = off.state_dict()
            on_state = on.state_dict()
            shared_keys = sorted(set(off_state).intersection(on_state))
            self.assertTrue(shared_keys)
            for key in shared_keys:
                self.assertTrue(
                    torch.equal(off_state[key], on_state[key]),
                    msg="shared off/on initialization differs at {}".format(key),
                )
        finally:
            torch.random.set_rng_state(original_rng_state)

    def test_lag_configuration_rejects_truncation_and_duplicates(self):
        for invalid_lags in ([1.5, 3], [True, 3], [1, 1]):
            args = _args(channels=1)
            args.mamba_acf_lags = invalid_lags
            with self.assertRaises(ValueError):
                Model(args)

        model = Model(_args(channels=1))
        with self.assertRaises(ValueError):
            model.set_acf_prior(torch.ones(1, 8), lags=[1.5, 3])
        with self.assertRaises(ValueError):
            model.set_compact_acf_prior(torch.ones(1, 2), lags=[1, 1])

    def test_constructor_rejects_nonfinite_acf_temperature(self):
        for temperature in (float("nan"), float("inf"), 0.0, -1.0):
            args = _args(channels=1)
            args.mamba_acf_temperature = temperature
            with self.assertRaisesRegex(ValueError, "temperature"):
                Model(args)

    def test_data_dependent_channel_mixer_profile_shape_and_rezero_contract(self):
        args = _args(channels=4, pred_len=12)
        args.mamba_data_dependent_channel_mixer = True
        args.mamba_data_dependent_channel_mixer_rank = 99
        model = Model(args)

        profile = model.resource_profile()
        self.assertTrue(profile["data_dependent_channel_mixer_enabled"])
        self.assertTrue(profile["data_dependent_channel_mixer_path_active"])
        self.assertEqual(profile["data_dependent_channel_mixer_requested_rank"], 99)
        # C=4 caps the actual low-rank bottleneck at C-1.
        self.assertEqual(profile["data_dependent_channel_mixer_rank"], 3)
        self.assertEqual(
            profile["data_dependent_channel_mixer_input_shape"], "B,T,D"
        )
        self.assertEqual(
            profile["data_dependent_channel_mixer_summary_shape"], "B,D"
        )
        self.assertEqual(
            profile["data_dependent_channel_mixer_residual_shape"], "B,T,D"
        )
        self.assertTrue(profile["data_dependent_channel_mixer_autocon_shared_state"])
        self.assertEqual(
            profile["data_dependent_channel_mixer_state_embedding_parameters"], 0
        )
        self.assertGreater(
            profile["data_dependent_channel_mixer_summary_projection_parameters"], 0
        )
        self.assertEqual(
            profile["data_dependent_channel_mixer_state_embedding_macs_per_batch_item"],
            0,
        )
        self.assertEqual(
            profile["data_dependent_channel_mixer_tokenizer"],
            "existing_scale_gate_hidden_summary",
        )
        self.assertEqual(
            profile["data_dependent_channel_mixer_summary_width"],
            args.d_model,
        )
        self.assertEqual(
            profile["data_dependent_channel_mixer_state_length"],
            args.seq_len,
        )
        self.assertEqual(
            profile["data_dependent_channel_mixer_pooling_scalar_ops_per_batch_item"],
            0,
        )
        self.assertGreater(
            profile["data_dependent_channel_mixer_adapter_linear_macs_per_batch_item"],
            0,
        )
        self.assertEqual(
            profile["data_dependent_channel_mixer_macs_per_batch_item"],
            profile["data_dependent_channel_mixer_state_embedding_macs_per_batch_item"]
            + profile["data_dependent_channel_mixer_adapter_linear_macs_per_batch_item"]
            + profile["data_dependent_channel_mixer_affine_macs_per_batch_item"],
        )

        window = torch.randn(2, args.seq_len, args.enc_in)
        residual = model._data_dependent_channel_residual(window)
        self.assertEqual(tuple(residual.shape), (2, 1, args.d_model))
        # The branch is enabled structurally but closed by ReZero at startup.
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))

        disabled = Model(_args(channels=4, pred_len=12)).resource_profile()
        self.assertEqual(
            disabled["data_dependent_channel_mixer_macs_per_batch_item"], 0
        )
        self.assertEqual(
            disabled["data_dependent_channel_mixer_activation_elements_per_batch_item"],
            0,
        )

    def test_data_dependent_channel_mixer_stride_scales_state_cost_without_repooling(self):
        args = _args(channels=7, pred_len=24)
        args.mamba_data_dependent_channel_mixer = True
        args.mamba_data_dependent_channel_mixer_rank = 4
        args.mamba_token_stride = 8
        args.seq_len = 97
        args.mamba_scales = [1, 2]
        model = Model(args)
        profile = model.resource_profile()

        self.assertEqual(profile["data_dependent_channel_mixer_state_length"], 13)
        self.assertEqual(
            profile["data_dependent_channel_mixer_affine_macs_per_batch_item"],
            13 * args.d_model,
        )
        self.assertEqual(
            profile["data_dependent_channel_mixer_pooling_scalar_ops_per_batch_item"],
            0,
        )

        dense_args = _args(channels=7, pred_len=24)
        dense_args.mamba_data_dependent_channel_mixer = True
        dense_args.mamba_data_dependent_channel_mixer_rank = 4
        dense = Model(dense_args).resource_profile()
        self.assertLess(
            profile["data_dependent_channel_mixer_affine_macs_per_batch_item"],
            dense["data_dependent_channel_mixer_affine_macs_per_batch_item"],
        )

    def test_data_dependent_channel_mixer_rejects_independent_mode(self):
        args = _args(channels=4, channel_independent=True)
        args.mamba_data_dependent_channel_mixer = True
        with self.assertRaisesRegex(ValueError, "shared-mode.*channel_independent"):
            Model(args)

    def test_dead_optional_parameters_are_excluded_from_plain_shared_capacity(self):
        args = _args(channels=5)
        args.AutoCon = False
        args.mamba_use_acf_memory = False
        model = Model(args)

        self.assertFalse(any(parameter.requires_grad for parameter in model.acf_gate.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.channel_pool.parameters()))
        trainable_names = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        self.assertNotIn("acf_gate.weight", trainable_names)
        self.assertNotIn("acf_gate.bias", trainable_names)
        self.assertNotIn("channel_pool.weight", trainable_names)
        self.assertNotIn("channel_pool.bias", trainable_names)

    def test_acf_gate_is_trainable_only_when_memory_branch_is_enabled(self):
        enabled = Model(_args(channels=3))
        self.assertTrue(all(parameter.requires_grad for parameter in enabled.acf_gate.parameters()))

        args = _args(channels=3)
        args.mamba_use_acf_memory = False
        model = Model(args)
        self.assertFalse(any(parameter.requires_grad for parameter in model.acf_gate.parameters()))

    def test_acf_scale_conditioning_uses_train_prior_and_is_zero_initialized(self):
        args = _args(channels=4, pred_len=12)
        args.mamba_scales = [1, 2]
        args.mamba_acf_scale_conditioning = True
        model = Model(args)
        self.assertIsNotNone(model.acf_scale_conditioner)
        self.assertEqual(model.acf_scale_conditioning_scale.item(), 0.0)
        window = torch.randn(2, args.seq_len, args.enc_in)
        before = model._acf_scale_summary(window)
        self.assertTrue(torch.equal(before, torch.zeros(2, 6)))
        prior = torch.zeros(4, 400)
        prior[:, 24] = torch.tensor([0.2, 0.4, 0.6, 0.8])
        model.set_acf_prior(prior, lags=[24, 168, 336])
        summary = model._acf_scale_summary(window)
        self.assertEqual(tuple(summary.shape), (2, 6))
        self.assertGreater(float(summary[:, 1].mean()), 0.0)
        bias = model._acf_scale_bias(window)
        self.assertEqual(tuple(bias.shape), (2, 2))
        # ReZero makes the enabled branch identical to the ordinary scale
        # logits until optimization explicitly opens it.
        self.assertTrue(torch.allclose(
            bias, torch.zeros_like(bias), atol=0.0, rtol=0.0
        ))

    def test_acf_scale_conditioning_summary_depends_on_current_window(self):
        args = _args(channels=2, pred_len=12)
        args.mamba_scales = [1, 2]
        args.mamba_acf_scale_conditioning = True
        model = Model(args)
        prior = torch.zeros(2, 400)
        prior[:, 24] = 0.8
        model.set_acf_prior(prior, lags=[24, 168, 336])

        flat = torch.zeros(2, args.seq_len, args.enc_in)
        positions = torch.arange(args.seq_len, dtype=torch.float32)
        periodic = torch.sin(2.0 * torch.pi * positions / 24.0)
        periodic = periodic.view(1, -1, 1).expand(2, -1, args.enc_in).clone()
        flat_summary = model._acf_scale_summary(flat)
        periodic_summary = model._acf_scale_summary(periodic)
        self.assertTrue(torch.equal(flat_summary, torch.zeros_like(flat_summary)))
        self.assertFalse(torch.allclose(flat_summary, periodic_summary))
        self.assertGreater(float(periodic_summary[:, 1].mean()), 0.5)

    def test_acf_scale_conditioning_reports_path_and_parameters(self):
        args = _args(channels=3, pred_len=12)
        args.mamba_scales = [1, 2]
        args.mamba_acf_scale_conditioning = True
        model = Model(args)
        profile = model.resource_profile()
        self.assertTrue(profile["acf_scale_conditioning_requested"])
        self.assertFalse(profile["acf_scale_conditioning_path_active"])
        self.assertGreater(profile["acf_scale_conditioning_parameters"], 0)
        model.set_acf_prior(torch.ones(3, 400), lags=[24, 168, 336])
        self.assertTrue(model.resource_profile()["acf_scale_conditioning_path_active"])

    def test_acf_scale_conditioning_preserves_shared_initialization(self):
        """Scale-on must differ only through an initially closed new path."""

        torch.manual_seed(907)
        off_args = _args(channels=3, pred_len=12)
        off_args.mamba_scales = [1, 2]
        off = Model(off_args)

        torch.manual_seed(907)
        on_args = _args(channels=3, pred_len=12)
        on_args.mamba_scales = [1, 2]
        on_args.mamba_acf_scale_conditioning = True
        on = Model(on_args)

        off_state = off.state_dict()
        on_state = on.state_dict()
        self.assertTrue(any(name.startswith("acf_scale_conditioner.") for name in on_state))
        scale_only = {
            name for name in on_state
            if name.startswith("acf_scale_conditioner.")
            or name == "acf_scale_conditioning_scale"
        }
        self.assertEqual(set(off_state), set(on_state) - scale_only)
        for name, value in off_state.items():
            self.assertTrue(torch.equal(value, on_state[name]), name)

    def test_variable_token_memory_gate_requires_static_variable_memory(self):
        args = _args(channels=3)
        args.mamba_acf_memory_gate_mode = "variable_token"
        with self.assertRaisesRegex(ValueError, "requires mamba_variable_mixer"):
            Model(args)

        args.mamba_variable_mixer = True
        args.mamba_dynamic_acf = True
        with self.assertRaisesRegex(ValueError, "requires static ACF memory"):
            Model(args)

    def test_variable_token_memory_gate_restores_channel_order_and_reports_mode(self):
        args = _args(channels=3, pred_len=12)
        args.mamba_variable_mixer = True
        args.mamba_acf_memory_gate_mode = "variable_token"
        args.mamba_acf_channel_order = "strength"
        model = Model(args).eval()
        model.set_acf_prior(torch.ones(3, 400), lags=args.mamba_acf_lags)
        # Keep this shape/ordering contract test CPU-only; the installed Mamba
        # kernels in the test environment require CUDA for the variable scan.
        model._run_variable_branch = lambda tokens: tokens
        hidden = torch.randn(2, args.seq_len, args.d_model)
        values = torch.randn(2, args.seq_len, args.enc_in)
        forecast = model._forecast_normalized(hidden, values)
        self.assertEqual(tuple(forecast.shape), (2, args.pred_len, args.enc_in))
        diagnostics = model.acf_runtime_diagnostics()
        self.assertEqual(diagnostics["acf_memory_gate_variable_token"], 1.0)
        profile = model.resource_profile()
        self.assertEqual(profile["acf_memory_gate_mode"], "variable_token")
        self.assertTrue(profile["acf_memory_gate_variable_token"])

    def test_direct_anchor_selects_per_channel_max_abs_available_lag(self):
        args = _args(channels=2, pred_len=4)
        args.seq_len = 8
        args.mamba_acf_lags = [1, 3, 5]
        args.mamba_patch_len = 8
        args.mamba_acf_anchor_copy = True
        args.mamba_use_acf_memory = False
        args.AutoCon_acf_source = "raw"
        model = Model(args)
        prior = torch.zeros(2, 6)
        prior[:, 1] = torch.tensor([0.2, 0.8])
        prior[:, 3] = torch.tensor([0.9, 0.1])
        prior[:, 5] = torch.tensor([0.5, -0.7])
        model.set_acf_prior(prior, lags=args.mamba_acf_lags)
        values = torch.arange(16, dtype=torch.float32).reshape(1, 8, 2)

        anchor, selected_lags, strength, valid = model._static_acf_anchor_memory(
            values, args.pred_len
        )
        # h=1 selects lag 3/1 for channels 0/1; later steps lose lag 1,
        # and by h=4 only lag 5 is legal.
        expected_lags = torch.tensor([
            [3, 1], [3, 5], [3, 5], [5, 5]
        ])
        expected = torch.tensor([
            [[10.0, 15.0], [12.0, -9.0], [14.0, -11.0], [12.0, -13.0]]
        ])
        torch.testing.assert_close(selected_lags.cpu(), expected_lags)
        torch.testing.assert_close(anchor, expected)
        torch.testing.assert_close(valid, torch.ones_like(valid))
        torch.testing.assert_close(strength, torch.tensor([
            [0.9, 0.8], [0.9, 0.7], [0.9, 0.7], [0.5, 0.7]
        ]))
        profile = model.resource_profile()
        self.assertTrue(profile["acf_anchor_copy_enabled"])
        self.assertEqual(profile["acf_anchor_copy_parameters"], 1)
        self.assertEqual(profile["acf_anchor_copy_selection"], "per_channel_max_abs_acf")

    def test_direct_anchor_masks_unavailable_lags_and_has_rezero_gradient(self):
        args = _args(channels=1, pred_len=4)
        args.seq_len = 8
        args.mamba_acf_lags = [3, 20]
        args.mamba_patch_len = 8
        args.mamba_acf_anchor_copy = True
        args.mamba_use_acf_memory = False
        args.AutoCon_acf_source = "raw"
        model = Model(args)
        prior = torch.zeros(1, 21)
        prior[:, 3] = 0.8
        prior[:, 20] = 0.95
        model.set_acf_prior(prior, lags=args.mamba_acf_lags)
        values = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1).requires_grad_()
        anchor, selected_lags, _, valid = model._static_acf_anchor_memory(values, 4)
        self.assertEqual(tuple(anchor.shape), (1, 4, 1))
        # lag 20 is unavailable throughout; lag 3 remains legal only for h<=3.
        self.assertTrue(torch.equal(valid.squeeze(-1).bool(), torch.tensor([True, True, True, False])))
        # The all-invalid row keeps a deterministic safe placeholder (the
        # first configured lag), but its validity mask forces zero contribution.
        self.assertTrue(torch.equal(selected_lags.squeeze(-1), torch.tensor([3, 3, 3, 3])))
        self.assertEqual(float(anchor[0, -1, 0].item()), 0.0)

        hidden = torch.zeros(1, args.seq_len, args.d_model)
        with torch.no_grad():
            model.acf_anchor_scale.fill_(0.4)
        forecast = model._forecast_normalized(hidden, values)
        forecast.sum().backward()
        self.assertIsNotNone(model.acf_anchor_scale.grad)
        self.assertTrue(torch.isfinite(model.acf_anchor_scale.grad).all())
        diagnostics = model.acf_runtime_diagnostics()
        self.assertGreater(diagnostics["acf_anchor_valid_fraction"], 0.0)
        self.assertEqual(diagnostics["acf_anchor_lag_mean"], 3.0)

    def test_direct_anchor_requires_static_raw_source(self):
        args = _args(channels=1)
        args.mamba_acf_anchor_copy = True
        args.AutoCon_acf_source = "residual"
        with self.assertRaisesRegex(ValueError, "requires AutoCon_acf_source=raw"):
            Model(args)
        args = _args(channels=1)
        args.mamba_acf_anchor_copy = True
        args.AutoCon_acf_source = "raw"
        args.mamba_dynamic_acf = True
        with self.assertRaisesRegex(ValueError, "requires static train-only ACF"):
            Model(args)

    def test_horizon_calibration_is_nested_and_trainable(self):
        baseline_args = _args(channels=2, pred_len=24)
        baseline_args.mamba_use_acf_memory = True
        baseline = Model(baseline_args)
        calibrated_args = _args(channels=2, pred_len=24)
        calibrated_args.mamba_acf_horizon_calibration = True
        calibrated_args.mamba_acf_horizon_calibration_rank = 2
        calibrated = Model(calibrated_args)
        self.assertEqual(
            calibrated.acf_horizon_calibration.parameter_count,
            2 * len(calibrated.acf_lags),
        )
        self.assertTrue(any(
            parameter.requires_grad
            for parameter in calibrated.acf_horizon_calibration.parameters()
        ))
        prior = torch.ones(2, 400)
        baseline.set_acf_prior(prior, lags=baseline_args.mamba_acf_lags)
        calibrated.set_acf_prior(prior, lags=calibrated_args.mamba_acf_lags)
        values = torch.randn(3, 96, 2)
        base_seasonal, base_rel = baseline._static_acf_memory(values, 24)
        cal_seasonal, cal_rel = calibrated._static_acf_memory(values, 24)
        self.assertTrue(torch.allclose(base_seasonal, cal_seasonal, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(base_rel, cal_rel, atol=1e-6, rtol=1e-6))
        profile = calibrated.resource_profile()
        self.assertTrue(profile["acf_horizon_calibration_enabled"])
        self.assertEqual(profile["acf_horizon_calibration_rank"], 2)
        self.assertEqual(profile["acf_horizon_calibration_parameters"], 6)

    def test_reliability_interpolation_is_nested_and_has_a_single_gradient(self):
        torch.manual_seed(311)
        baseline_args = _args(channels=2, pred_len=24)
        baseline = Model(baseline_args)
        torch.manual_seed(311)
        adapted_args = _args(channels=2, pred_len=24)
        adapted_args.mamba_acf_reliability_interpolation = True
        adapted = Model(adapted_args)
        prior = torch.ones(2, 400)
        baseline.set_acf_prior(prior, lags=baseline_args.mamba_acf_lags)
        adapted.set_acf_prior(prior, lags=adapted_args.mamba_acf_lags)
        values = torch.randn(3, 96, 2)
        base_seasonal, base_rel = baseline._static_acf_memory(values, 24)
        adapted_seasonal, adapted_rel = adapted._static_acf_memory(values, 24)
        self.assertTrue(torch.equal(base_seasonal, adapted_seasonal))
        self.assertTrue(torch.equal(base_rel, adapted_rel))
        self.assertEqual(adapted.acf_reliability_interpolation_scale.item(), 0.0)
        with torch.no_grad():
            adapted.acf_reliability_interpolation_scale.fill_(0.5)
        opened, _ = adapted._static_acf_memory(values, 24)
        self.assertFalse(torch.allclose(opened, base_seasonal))
        opened.square().mean().backward()
        gradient = adapted.acf_reliability_interpolation_scale.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertNotEqual(float(gradient.abs().item()), 0.0)
        profile = adapted.resource_profile()
        self.assertTrue(profile["acf_reliability_interpolation_enabled"])
        self.assertEqual(profile["acf_reliability_interpolation_parameters"], 1)
        self.assertTrue(profile["acf_reliability_interpolation_path_active"])

    def test_reliability_interpolation_requires_static_raw_memory(self):
        invalid = []
        args = _args(channels=1)
        args.mamba_use_acf_memory = False
        args.mamba_acf_reliability_interpolation = True
        invalid.append((args, "mamba_use_acf_memory"))
        args = _args(channels=1)
        args.mamba_dynamic_acf = True
        args.mamba_acf_reliability_interpolation = True
        invalid.append((args, "static"))
        args = _args(channels=1)
        args.AutoCon_acf_source = "residual"
        args.mamba_acf_reliability_interpolation = True
        invalid.append((args, "AutoCon_acf_source=raw"))
        args = _args(channels=1)
        args.mamba_acf_step_reliability = True
        args.mamba_acf_reliability_interpolation = True
        invalid.append((args, "step_reliability"))
        for candidate, message in invalid:
            with self.assertRaisesRegex(ValueError, message):
                Model(candidate)

    def test_horizon_calibration_rejects_dynamic_memory(self):
        args = _args(channels=2)
        args.mamba_dynamic_acf = True
        args.mamba_acf_horizon_calibration = True
        with self.assertRaises(ValueError):
            Model(args)

    def test_horizon_calibration_resizes_to_resolved_sidecar_lags(self):
        args = _args(channels=2, pred_len=24)
        args.mamba_use_acf_memory = True
        args.mamba_acf_horizon_calibration = True
        args.mamba_acf_lags = [24, 168, 336]
        model = Model(args)
        with torch.no_grad():
            model.acf_horizon_calibration.coefficients.copy_(
                torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
            )
        model.set_acf_prior(torch.ones(2, 400), lags=[168, 24])
        self.assertEqual(model.acf_lags, (24, 168))
        self.assertEqual(model.acf_horizon_calibration.lag_count, 2)
        self.assertEqual(model.acf_horizon_calibration.parameter_count, 4)
        self.assertTrue(torch.allclose(
            model.acf_horizon_calibration.coefficients,
            torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        ))
        seasonal, _ = model._static_acf_memory(torch.randn(2, 96, 2), 24)
        self.assertEqual(tuple(seasonal.shape), (2, 24, 2))

    def test_acf_representation_fusion_is_opt_in_and_profiled(self):
        disabled = Model(_args(channels=3))
        self.assertFalse(disabled.acf_repr_fusion_enabled)
        self.assertIsNone(disabled.acf_repr_projection)
        self.assertFalse(disabled.resource_profile()["acf_representation_fusion_enabled"])

        args = _args(channels=3)
        args.mamba_acf_repr_fusion = True
        args.AutoCon_acf_source = "raw"
        enabled = Model(args)
        self.assertTrue(enabled.acf_repr_fusion_enabled)
        self.assertIsNotNone(enabled.acf_repr_projection)
        self.assertTrue(enabled.acf_repr_scale.requires_grad)
        profile = enabled.resource_profile()
        self.assertTrue(profile["acf_representation_lag_index_is_causal"])
        self.assertTrue(profile["acf_representation_transform_is_causal"])
        self.assertTrue(profile["acf_representation_context_is_causal"])
        self.assertGreater(profile["acf_representation_parameters"], 0)

        enabled.acf_source = "residual"
        residual_profile = enabled.resource_profile()
        self.assertTrue(residual_profile["acf_representation_lag_index_is_causal"])
        self.assertFalse(residual_profile["acf_representation_transform_is_causal"])
        self.assertFalse(residual_profile["acf_representation_context_is_causal"])

        plain_args = _args(channels=3)
        plain_args.AutoCon = False
        plain_args.mamba_acf_repr_fusion = True
        plain = Model(plain_args)
        self.assertTrue(plain.acf_repr_fusion_requested)
        self.assertFalse(plain.acf_repr_fusion_enabled)

    def test_profile_separates_acf_context_from_full_grid_causality(self):
        args = _args(channels=3)
        args.mamba_bidirectional = False
        args.mamba_causal_embedding = True
        args.mamba_scales = [1]
        args.mamba_acf_repr_fusion = True
        args.AutoCon_acf_source = "raw"
        causal = Model(args).resource_profile()
        self.assertTrue(causal["acf_representation_context_is_causal"])
        self.assertFalse(causal["representation_grid_is_strictly_causal"])
        self.assertIn(
            "window_full_normalization",
            causal["representation_grid_causality_limits"],
        )
        self.assertTrue(causal["representation_normalization_is_window_global"])

        args.mamba_token_stride = 2
        strided = Model(args).resource_profile()
        self.assertTrue(strided["acf_representation_context_is_causal"])
        self.assertFalse(strided["representation_grid_is_strictly_causal"])
        self.assertIn(
            "strided_pooling_and_interpolation",
            strided["representation_grid_causality_limits"],
        )

        args.mamba_token_stride = 1
        args.mamba_scales = [1, 2]
        multiscale = Model(args).resource_profile()
        self.assertFalse(multiscale["representation_grid_is_strictly_causal"])
        self.assertIn(
            "multiscale_pooling_and_interpolation",
            multiscale["representation_grid_causality_limits"],
        )

        args.mamba_scales = [1]
        args.AutoCon_acf_source = "residual"
        residual = Model(args).resource_profile()
        self.assertFalse(residual["acf_representation_context_is_causal"])
        self.assertFalse(residual["representation_grid_is_strictly_causal"])
        self.assertIn(
            "centered_acf_representation_transform",
            residual["representation_grid_causality_limits"],
        )

    def test_window_normalization_changes_a_prefix_when_only_future_changes(self):
        """Current modes normalize from the full window, not a prefix."""

        first = torch.zeros(1, 8, 1)
        second = first.clone()
        second[:, -1, :] = 1.0
        for mode in ("LastVal", "Mean", "ReVIN", "MeanStd"):
            holder = SimpleNamespace(AutoCon_wnorm=mode)
            first_normalized, _ = Model._normalize(holder, first)
            second_normalized, _ = Model._normalize(holder, second)
            self.assertGreater(
                (first_normalized[:, :4] - second_normalized[:, :4])
                .abs()
                .max()
                .item(),
                0.0,
                msg=mode,
            )

    def test_meanstd_matches_patchtst_population_statistics_and_restores(self):
        """The explicit parity mode uses PatchTST's exact window contract."""

        values = torch.randn(3, 11, 4)
        holder = SimpleNamespace(AutoCon_wnorm="MeanStd")
        normalized, restore = Model._normalize(holder, values)
        mean = values.mean(dim=1, keepdim=True)
        std = torch.sqrt(
            (values - mean).var(dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        expected = (values - mean) / std
        self.assertTrue(torch.allclose(normalized, expected, atol=1e-7, rtol=0.0))
        restored = Model._restore(holder, normalized, restore)
        self.assertTrue(torch.allclose(restored, values, atol=1e-6, rtol=0.0))

    def test_revin_keeps_shared_autocon_sample_standard_deviation_contract(self):
        values = torch.randn(3, 11, 4)
        holder = SimpleNamespace(AutoCon_wnorm="ReVIN")
        normalized, restore = Model._normalize(holder, values)
        mean = values.mean(dim=1, keepdim=True)
        std = values.std(dim=1, keepdim=True, unbiased=True)
        expected = (values - mean) / (std + 1e-5)
        self.assertTrue(torch.allclose(normalized, expected, atol=1e-7, rtol=0.0))
        restored = Model._restore(holder, normalized, restore)
        self.assertTrue(torch.allclose(restored, values, atol=1e-6, rtol=0.0))

    def test_meanstd_is_supported_by_mamba_autocon_window_helper(self):
        from models.Mamba2AutoCon import _MambaForecastModel

        values = torch.randn(2, 9, 3)
        holder = SimpleNamespace(AutoCon_wnorm="MeanStd")
        short_x, long_x, restore = _MambaForecastModel._normalize_window(
            holder, values
        )
        self.assertTrue(torch.equal(short_x, long_x))
        self.assertTrue(torch.allclose(short_x.mean(dim=1), torch.zeros(2, 3), atol=1e-6))
        mean, std = restore
        expected_std = torch.sqrt(
            (values - mean).var(dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        self.assertTrue(torch.allclose(std, expected_std, atol=1e-7, rtol=0.0))
        self.assertTrue(torch.allclose(short_x * std + mean, values, atol=1e-6))

    def test_acf_representation_context_does_not_read_future_positions(self):
        args = _args(channels=2, pred_len=8)
        args.seq_len = 12
        args.patch_len = 4
        args.patch_stride = 2
        args.mamba_patch_len = 4
        args.mamba_patch_stride = 2
        args.mamba_acf_lags = [2, 20]
        args.mamba_acf_repr_fusion = True
        args.mamba_acf_temperature = 4.0
        args.acf_source = "raw"
        model = Model(args)
        model.acf_source = "raw"
        model.set_compact_acf_prior(
            torch.tensor([[0.8, 0.4], [-0.6, 0.2]]), [2, 20]
        )
        first = torch.arange(24, dtype=torch.float32).reshape(1, 12, 2)
        second = first.clone()
        second[:, 6:, :] += 1000.0
        context_first = model._acf_representation_context(first)
        context_second = model._acf_representation_context(second)
        self.assertEqual(tuple(context_first.shape), (1, 12, 4))
        self.assertTrue(torch.isfinite(context_first).all())
        # Positions before the changed prefix can only use t-lag values and
        # therefore must be identical.  The long lag remains explicitly
        # unavailable rather than reading a clamped value.
        torch.testing.assert_close(context_first[:, :6], context_second[:, :6])
        self.assertLess(float(context_first[..., 3].max()), 1.0)

    def test_compact_acf_prior_artifact_round_trip(self):
        args = _args(channels=3, pred_len=24)
        args.mamba_bidirectional = False
        source = Model(args)
        full_prior = torch.zeros(args.enc_in, 97)
        full_prior[:, 24] = torch.tensor([0.8, -0.4, 0.2])
        full_prior[:, 72] = torch.tensor([-0.3, 0.5, 0.1])
        source.set_acf_prior(full_prior, lags=[24, 72])
        source_state = source.export_acf_prior_state()
        self.assertIsNotNone(source_state)
        np.testing.assert_allclose(source_state["acf_full_values"], full_prior.numpy())

        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "acf_prior.npz"
            self.assertTrue(source.save_acf_prior(artifact))
            restored = Model(args)
            self.assertTrue(restored.load_acf_prior(artifact))
            torch.testing.assert_close(restored.acf_prior, source.acf_prior)
            torch.testing.assert_close(restored.acf_prior_lags, source.acf_prior_lags)
            restored_state = restored.export_acf_prior_state()
            self.assertIsNotNone(restored_state)
            np.testing.assert_allclose(restored_state["acf_full_values"], full_prior.numpy())
            original_memory, original_reliability = source._static_acf_memory(
                torch.randn(2, args.seq_len, args.enc_in), args.pred_len
            )
            restored_memory, restored_reliability = restored._static_acf_memory(
                torch.randn(2, args.seq_len, args.enc_in), args.pred_len
            )
            # Compare the compiled coefficients/reliability rather than the
            # random input-dependent seasonal values.
            torch.testing.assert_close(
                restored.acf_memory_static_coefficients,
                source.acf_memory_static_coefficients,
            )
            torch.testing.assert_close(
                restored.acf_memory_static_step_reliability,
                source.acf_memory_static_step_reliability,
            )
            self.assertEqual(tuple(original_memory.shape), tuple(restored_memory.shape))
            self.assertEqual(tuple(original_reliability.shape), tuple(restored_reliability.shape))

            mismatched_args = _args(channels=3, pred_len=12)
            mismatched = Model(mismatched_args)
            with self.assertRaisesRegex(ValueError, "pred_len"):
                mismatched.load_acf_prior(artifact)

            float_lag_artifact = Path(temporary) / "float_lags.npz"
            with np.load(artifact, allow_pickle=False) as payload:
                fields = {name: payload[name] for name in payload.files}
            fields["lags"] = np.asarray([24.0, 72.0], dtype=np.float32)
            np.savez_compressed(float_lag_artifact, **fields)
            with self.assertRaisesRegex(ValueError, "integer dtype"):
                restored.load_acf_prior(float_lag_artifact)

            fractional_geometry = Path(temporary) / "fractional_geometry.npz"
            with np.load(artifact, allow_pickle=False) as payload:
                fields = {name: payload[name] for name in payload.files}
            fields["seq_len"] = np.asarray(float(args.seq_len) + 0.5)
            np.savez_compressed(fractional_geometry, **fields)
            with self.assertRaisesRegex(ValueError, "seq_len"):
                restored.load_acf_prior(fractional_geometry)

            out_of_range = Path(temporary) / "out_of_range.npz"
            with np.load(artifact, allow_pickle=False) as payload:
                fields = {name: payload[name] for name in payload.files}
            fields["acf_values"] = fields["acf_values"].copy()
            fields["acf_values"][0, 0] = 1.01
            np.savez_compressed(out_of_range, **fields)
            with self.assertRaisesRegex(ValueError, r"within \[-1, 1\]"):
                restored.load_acf_prior(out_of_range)

    def test_compact_acf_prior_artifact_rejects_complete_table_mismatch(self):
        args = _args(channels=2, pred_len=24)
        source = Model(args)
        full_prior = torch.zeros(args.enc_in, 97)
        full_prior[:, 24] = torch.tensor([0.8, -0.4])
        full_prior[:, 72] = torch.tensor([-0.3, 0.5])
        source.set_acf_prior(full_prior, lags=[24, 72])

        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "acf_prior.npz"
            self.assertTrue(source.save_acf_prior(artifact))
            with np.load(artifact, allow_pickle=False) as payload:
                fields = {name: payload[name] for name in payload.files}
            near_equal_fields = dict(fields)
            near_equal_fields["acf_full_values"] = fields["acf_full_values"].copy()
            near_equal_fields["acf_full_values"][0, 24] += np.float32(5e-7)
            near_equal = Path(temporary) / "near_equal_acf_prior.npz"
            np.savez_compressed(near_equal, **near_equal_fields)
            self.assertTrue(Model(args).load_acf_prior(near_equal))

            compact_only_fields = dict(fields)
            compact_only_fields.pop("acf_full_values")
            compact_only = Path(temporary) / "compact_only_acf_prior.npz"
            np.savez_compressed(compact_only, **compact_only_fields)
            compact_only_model = Model(args)
            self.assertTrue(compact_only_model.load_acf_prior(compact_only))
            self.assertEqual(compact_only_model.acf_full_prior.numel(), 0)

            fields["acf_full_values"] = fields["acf_full_values"].copy()
            fields["acf_full_values"][0, 24] += np.float32(0.01)
            corrupted = Path(temporary) / "corrupted_acf_prior.npz"
            np.savez_compressed(corrupted, **fields)

            with self.assertRaisesRegex(
                ValueError, "full ACF prior selected values do not match"
            ):
                Model(args).load_acf_prior(corrupted)

    def test_direct_compact_prior_rejects_complete_table_mismatch(self):
        args = _args(channels=1, pred_len=8)
        model = Model(args)
        compact = torch.tensor([[0.8, -0.4]])
        full = torch.zeros(1, 97)
        full[:, 24] = 0.7
        full[:, 72] = -0.4
        with self.assertRaisesRegex(
            ValueError, "full ACF prior selected values do not match"
        ):
            model.set_compact_acf_prior(compact, [24, 72], full_acf_values=full)

    def test_direct_compact_prior_accepts_unsorted_aligned_complete_table(self):
        args = _args(channels=1, pred_len=8)
        model = Model(args)
        # The compact columns follow the caller's [72, 24] order; the public
        # setter canonicalizes both columns and the full-table comparison.
        compact = torch.tensor([[-0.4, 0.9]])
        full = torch.zeros(1, 97)
        full[:, 24] = 0.9
        full[:, 72] = -0.4
        model.set_compact_acf_prior(
            compact, [72, 24], full_acf_values=full
        )
        self.assertEqual(tuple(model.acf_prior_lags.tolist()), (24, 72))
        torch.testing.assert_close(model.acf_prior, torch.tensor([[0.9, -0.4]]))

    def test_compact_acf_prior_keeps_unsorted_lag_values_aligned(self):
        args = _args(channels=1, pred_len=8)
        model = Model(args)
        model.set_compact_acf_prior(torch.tensor([[0.9, -0.4]]), [72, 24])
        self.assertEqual(tuple(model.acf_prior_lags.tolist()), (24, 72))
        torch.testing.assert_close(model.acf_prior, torch.tensor([[-0.4, 0.9]]))

    def test_static_memory_without_prior_fails_loudly(self):
        args = _args(channels=2, pred_len=12)
        args.mamba_bidirectional = False
        model = Model(args)
        hidden = torch.zeros(1, args.seq_len, args.d_model)
        values = torch.zeros(1, args.seq_len, args.enc_in)
        with self.assertRaisesRegex(RuntimeError, "no train-only prior"):
            model._forecast_normalized(hidden, values)

    def test_channel_pool_remains_trainable_for_channel_independent_autocon(self):
        args = _args(channels=3, channel_independent=True)
        args.AutoCon = True
        model = Model(args)
        self.assertTrue(all(parameter.requires_grad for parameter in model.channel_pool.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.acf_gate.parameters()))

    def test_independent_parameter_count_is_channel_invariant(self):
        small_args = _args(channels=3, channel_independent=True)
        large_args = _args(channels=862, channel_independent=True)
        small_args.mamba_bidirectional = False
        large_args.mamba_bidirectional = False
        small = Model(small_args)
        large = Model(large_args)
        small_count = sum(parameter.numel() for parameter in small.parameters())
        large_count = sum(parameter.numel() for parameter in large.parameters())
        # c_out follows enc_in in this controlled comparison, so no output
        # projection introduces a channel-dependent parameter matrix.
        self.assertEqual(small_count, large_count)
        self.assertEqual(
            small.resource_profile()["main_scan_streams_per_batch_item"],
            3,
        )
        self.assertEqual(
            large.resource_profile()["main_scan_streams_per_batch_item"],
            862,
        )

    def test_shared_and_variable_parameter_counts_expose_channel_cost(self):
        shared_small = Model(_args(channels=3))
        shared_large = Model(_args(channels=862))
        variable_small_args = _args(channels=3)
        variable_small_args.mamba_variable_mixer = True
        variable_large_args = _args(channels=862)
        variable_large_args.mamba_variable_mixer = True
        variable_small = Model(variable_small_args)
        variable_large = Model(variable_large_args)
        shared_small_count = sum(parameter.numel() for parameter in shared_small.parameters())
        shared_large_count = sum(parameter.numel() for parameter in shared_large.parameters())
        variable_small_count = sum(parameter.numel() for parameter in variable_small.parameters())
        variable_large_count = sum(parameter.numel() for parameter in variable_large.parameters())
        self.assertGreater(shared_large_count, shared_small_count)
        self.assertGreater(variable_large_count, variable_small_count)
        self.assertEqual(
            variable_large.resource_profile()["shared_mamba_component_channel_rank_bound"],
            variable_large.d_model,
        )
        self.assertEqual(
            variable_large.resource_profile()["variable_scan_tokens_per_stream"],
            862,
        )

    def test_grouped_variable_tokens_pool_without_padding_dilution(self):
        args = _args(channels=5)
        args.mamba_variable_mixer = True
        args.mamba_variable_group_size = 2
        model = Model(args)
        tokens = torch.zeros(1, 5, args.d_model)
        tokens[0, :, 0] = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0])
        grouped = model._group_variable_tokens(tokens)
        self.assertEqual(tuple(grouped.shape), (1, 3, args.d_model))
        self.assertTrue(torch.allclose(
            grouped[0, :, 0], torch.tensor([2.0, 6.0, 9.0])
        ))
        profile = model.resource_profile()
        self.assertEqual(profile["variable_token_group_size"], 2)
        self.assertEqual(profile["variable_token_group_count"], 3)
        self.assertEqual(profile["variable_token_group_padding"], 1)
        self.assertEqual(profile["variable_scan_tokens_per_stream"], 3)
        self.assertAlmostEqual(profile["variable_scan_token_reduction"], 0.4)

    def test_group_size_one_is_exact_legacy_variable_token_contract(self):
        args = _args(channels=5)
        args.mamba_variable_mixer = True
        model = Model(args)
        tokens = torch.randn(2, 5, args.d_model)
        self.assertTrue(torch.equal(model._group_variable_tokens(tokens), tokens))
        profile = model.resource_profile()
        self.assertEqual(profile["variable_token_group_size"], 1)
        self.assertEqual(profile["variable_token_group_count"], 5)
        self.assertEqual(profile["variable_scan_tokens_per_stream"], 5)
        self.assertEqual(profile["variable_scan_token_reduction"], 0.0)

    def test_group_size_requires_variable_mixer(self):
        args = _args(channels=5)
        args.mamba_variable_group_size = 2
        with self.assertRaisesRegex(ValueError, "requires mamba_variable_mixer"):
            Model(args)

    def test_acf_strength_order_is_stable_signed_peak_order_and_reversible(self):
        args = _args(channels=4)
        args.mamba_variable_mixer = True
        args.mamba_acf_channel_order = "strength"
        args.mamba_acf_lags = [24, 48, 72]
        model = Model(args)
        prior = torch.zeros(4, 73)
        prior[:, 24] = torch.tensor([0.2, -0.9, 0.6, 0.6])
        prior[:, 48] = torch.tensor([-0.3, 0.1, 0.2, -0.1])
        prior[:, 72] = torch.tensor([0.4, 0.2, -0.1, 0.2])
        model.set_acf_prior(prior, lags=args.mamba_acf_lags)
        self.assertTrue(torch.equal(
            model.acf_channel_order_indices,
            torch.tensor([1, 2, 3, 0]),
        ))
        self.assertTrue(torch.equal(
            model.acf_channel_inverse_indices,
            torch.tensor([3, 0, 1, 2]),
        ))
        self.assertTrue(torch.equal(
            model.acf_channel_order_indices.index_select(
                0, model.acf_channel_inverse_indices
            ),
            torch.arange(4),
        ))
        self.assertTrue(torch.allclose(
            model.acf_channel_dominant_peak,
            torch.tensor([0.4, -0.9, 0.6, 0.6]),
        ))
        self.assertTrue(torch.equal(
            model.acf_channel_dominant_peak_lag,
            torch.tensor([72, 24, 24, 24]),
        ))
        profile = model.resource_profile()
        self.assertEqual(profile["variable_token_order"], "strength")
        self.assertTrue(profile["variable_token_order_uses_train_acf_prior"])
        self.assertTrue(profile["variable_token_order_prior_attached"])

    def test_strength_order_without_train_prior_is_identity(self):
        args = _args(channels=4)
        args.mamba_variable_mixer = True
        args.mamba_acf_channel_order = "strength"
        model = Model(args)
        order, inverse = model._train_prior_channel_order(torch.device("cpu"))
        self.assertIsNone(order)
        self.assertIsNone(inverse)
        self.assertEqual(model.resource_profile()["variable_token_order"], "strength")
        self.assertFalse(model.resource_profile()["variable_token_order_prior_attached"])

    def test_similarity_order_is_deterministic_signed_profile_and_reversible(self):
        args = _args(channels=4)
        args.mamba_variable_mixer = True
        args.mamba_acf_channel_order = "similarity"
        args.mamba_acf_lags = [24, 48, 72]
        model = Model(args)
        prior = torch.zeros(4, 73)
        # Channels 0 and 1 are a positively aligned pair, channel 3 is
        # closer to that pair than to the anti-phase channel 2.  The start
        # channel is the strongest profile, with input-index tie breaking.
        prior[:, 24] = torch.tensor([1.0, 0.95, -1.0, 0.0])
        prior[:, 48] = torch.tensor([0.0, 0.05, 0.0, 0.9])
        prior[:, 72] = torch.tensor([0.0, 0.0, 0.0, 0.1])
        model.set_acf_prior(prior, lags=args.mamba_acf_lags)
        self.assertTrue(torch.equal(
            model.acf_channel_order_indices,
            torch.tensor([0, 1, 3, 2]),
        ))
        self.assertTrue(torch.equal(
            model.acf_channel_inverse_indices,
            torch.tensor([0, 1, 3, 2]),
        ))
        self.assertTrue(torch.equal(
            model.acf_channel_order_indices.index_select(
                0, model.acf_channel_inverse_indices
            ),
            torch.arange(4),
        ))
        profile = model.resource_profile()
        self.assertEqual(profile["variable_token_order"], "similarity")
        self.assertEqual(
            profile["variable_token_order_rule"],
            "signed_acf_profile_cosine_greedy",
        )
        self.assertTrue(profile["variable_token_order_uses_train_acf_prior"])
        self.assertTrue(profile["variable_token_order_prior_attached"])

    def test_similarity_order_without_train_prior_is_identity(self):
        args = _args(channels=4)
        args.mamba_variable_mixer = True
        args.mamba_acf_channel_order = "similarity"
        model = Model(args)
        order, inverse = model._train_prior_channel_order(torch.device("cpu"))
        self.assertIsNone(order)
        self.assertIsNone(inverse)
        self.assertEqual(model.resource_profile()["variable_token_order"], "similarity")
        self.assertFalse(model.resource_profile()["variable_token_order_prior_attached"])

    def test_input_order_remains_identity_after_train_prior(self):
        args = _args(channels=4)
        args.mamba_variable_mixer = True
        model = Model(args)
        prior = torch.zeros(4, 73)
        prior[:, 24] = torch.tensor([0.2, 0.9, 0.6, 0.5])
        prior[:, 48] = torch.tensor([0.3, 0.1, 0.2, 0.4])
        prior[:, 72] = torch.tensor([0.4, 0.2, 0.1, 0.3])
        model.set_acf_prior(prior, lags=[24, 48, 72])
        expected = torch.arange(4)
        self.assertTrue(torch.equal(model.acf_channel_order_indices, expected))
        self.assertTrue(torch.equal(model.acf_channel_inverse_indices, expected))

    def test_strength_order_requires_variable_mixer(self):
        args = _args()
        args.mamba_acf_channel_order = "strength"
        with self.assertRaisesRegex(ValueError, "requires mamba_variable_mixer"):
            Model(args)

    def test_similarity_order_requires_variable_mixer(self):
        args = _args()
        args.mamba_acf_channel_order = "similarity"
        with self.assertRaisesRegex(ValueError, "requires mamba_variable_mixer"):
            Model(args)

    def test_invalid_acf_channel_order_is_rejected(self):
        args = _args()
        args.mamba_acf_channel_order = "random"
        with self.assertRaisesRegex(ValueError, "must be input, strength, or similarity"):
            Model(args)

    def test_profile_reports_exact_multiscale_scan_lengths(self):
        args = _args(channels=7)
        args.mamba_scales = [1, 2, 5]
        args.mamba_bidirectional = False
        profile = Model(args).resource_profile()
        self.assertEqual(profile["main_scan_tokens_per_scale"], [96, 48, 20])
        self.assertEqual(profile["main_scan_tokens_per_stream"], 96)
        self.assertEqual(profile["main_scan_tokens_total_per_batch_item"], 164)
        self.assertEqual(profile["main_scan_operations_per_batch_item"], 3)
        self.assertEqual(profile["temporal_token_stride"], 1)

    def test_multiscale_pool_uses_real_tail_values_and_terminal_marks(self):
        """A partial scale chunk has no padded value or mismatched timestamp."""

        args = _args(channels=1)
        args.seq_len = 5
        args.mamba_scales = [1, 3]
        args.mamba_patch_len = 5
        args.mamba_bidirectional = False
        model = Model(args)
        values = torch.arange(5, dtype=torch.float32).reshape(1, 5, 1)
        marks = torch.arange(5, dtype=torch.float32).reshape(1, 5, 1)
        calls = []

        def fake_embed(branch_values, branch_marks):
            calls.append((branch_values.detach().clone(), branch_marks.detach().clone()))
            return branch_values.new_zeros(
                branch_values.shape[0], branch_values.shape[1], model.d_model
            )

        with patch.object(model, "_embed", side_effect=fake_embed), \
                patch.object(model, "_run_branch", side_effect=lambda hidden, blocks: hidden):
            hidden = model._encode_channels(values, marks)

        self.assertEqual(tuple(hidden.shape), (1, args.seq_len, args.d_model))
        self.assertEqual([int(branch_values.shape[1]) for branch_values, _ in calls], [5, 2])
        torch.testing.assert_close(calls[0][0][0, :, 0], torch.arange(5, dtype=torch.float32))
        torch.testing.assert_close(calls[0][1][0, :, 0], torch.arange(5, dtype=torch.float32))
        torch.testing.assert_close(calls[1][0][0, :, 0], torch.tensor([1.0, 3.5]))
        torch.testing.assert_close(calls[1][1][0, :, 0], torch.tensor([2.0, 4.0]))

    def test_default_divisible_multiscale_values_remain_unchanged(self):
        """The default 96/[1, 2] scale path keeps its established values."""

        args = _args(channels=1)
        args.mamba_scales = [1, 2]
        args.mamba_bidirectional = False
        model = Model(args)
        values = torch.arange(args.seq_len, dtype=torch.float32).reshape(1, -1, 1)

        torch.testing.assert_close(model._pool_scale(values, 1), values)
        torch.testing.assert_close(
            model._pool_scale(values, 2)[0, :, 0],
            torch.arange(0.5, args.seq_len, 2.0),
        )
        marks = torch.arange(args.seq_len, dtype=torch.float32).reshape(1, -1, 1)
        torch.testing.assert_close(
            model._pool_scale_marks(marks, 2)[0, :, 0],
            torch.arange(1.0, args.seq_len, 2.0),
        )

    def test_token_stride_is_causal_and_updates_multiscale_profile(self):
        args = _args(channels=7)
        args.seq_len = 97
        args.mamba_token_stride = 8
        args.mamba_scales = [1, 2, 3]
        args.mamba_bidirectional = False
        model = Model(args)

        marks = torch.arange(97, dtype=torch.float32).reshape(1, 97, 1)
        base_marks = model._token_stride_marks(marks, args.mamba_token_stride)
        self.assertTrue(torch.equal(
            base_marks.squeeze(-1).squeeze(0),
            torch.tensor([7, 15, 23, 31, 39, 47, 55, 63, 71, 79, 87, 95, 96]),
        ))
        scale_two_marks = model._token_stride_marks(base_marks, 2)
        self.assertTrue(torch.equal(
            scale_two_marks.squeeze(-1).squeeze(0),
            torch.tensor([15, 31, 47, 63, 79, 95, 96]),
        ))
        profile = model.resource_profile()
        self.assertEqual(profile["temporal_token_stride"], 8)
        self.assertEqual(profile["temporal_token_count_before_multiscale"], 13)
        self.assertEqual(profile["main_scan_tokens_per_scale"], [13, 7, 5])
        self.assertEqual(profile["main_scan_tokens_total_per_batch_item"], 25)

    def test_token_stride_reaches_each_mocked_branch_with_causal_values_and_marks(self):
        """Exercise the actual pooled inputs passed to every temporal branch."""

        args = _args(channels=1)
        args.seq_len = 97
        args.mamba_token_stride = 8
        args.mamba_scales = [1, 2, 3]
        args.mamba_bidirectional = False
        model = Model(args)
        values = torch.arange(97, dtype=torch.float32).reshape(1, 97, 1)
        marks = torch.arange(97, dtype=torch.float32).reshape(1, 97, 1)
        calls = []

        def fake_embed(branch_values, branch_marks):
            calls.append((branch_values.detach().clone(), branch_marks.detach().clone()))
            return branch_values.new_zeros(
                branch_values.shape[0], branch_values.shape[1], model.d_model
            )

        with patch.object(model, "_embed", side_effect=fake_embed), \
                patch.object(model, "_run_branch", side_effect=lambda hidden, blocks: hidden):
            hidden = model._encode_channels(values, marks)

        self.assertEqual(tuple(hidden.shape), (1, 97, args.d_model))
        self.assertEqual([int(branch_values.shape[1]) for branch_values, _ in calls], [13, 7, 5])
        expected_values = [
            [3.5, 11.5, 19.5, 27.5, 35.5, 43.5, 51.5, 59.5,
             67.5, 75.5, 83.5, 91.5, 96.0],
            [7.5, 23.5, 39.5, 55.5, 71.5, 87.5, 96.0],
            [11.5, 35.5, 59.5, 83.5, 96.0],
        ]
        expected_marks = [
            [7, 15, 23, 31, 39, 47, 55, 63, 71, 79, 87, 95, 96],
            [15, 31, 47, 63, 79, 95, 96],
            [23, 47, 71, 95, 96],
        ]
        for (branch_values, branch_marks), values_expected, marks_expected in zip(
            calls, expected_values, expected_marks
        ):
            torch.testing.assert_close(branch_values[0, :, 0], torch.tensor(values_expected))
            torch.testing.assert_close(
                branch_marks[0, :, 0], torch.tensor(marks_expected, dtype=branch_marks.dtype)
            )

    def test_multiscale_direct_pool_weights_partial_base_tail_by_real_count(self):
        """A partial base chunk is weighted by its real observations only."""

        args = _args(channels=1)
        args.seq_len = 105
        args.mamba_token_stride = 8
        args.mamba_scales = [1, 2]
        args.mamba_bidirectional = False
        model = Model(args)
        values = torch.arange(105, dtype=torch.float32).reshape(1, 105, 1)
        marks = torch.arange(105, dtype=torch.float32).reshape(1, 105, 1)
        calls = []

        def fake_embed(branch_values, branch_marks):
            calls.append((branch_values.detach().clone(), branch_marks.detach().clone()))
            return branch_values.new_zeros(
                branch_values.shape[0], branch_values.shape[1], model.d_model
            )

        with patch.object(model, "_embed", side_effect=fake_embed), \
                patch.object(model, "_run_branch", side_effect=lambda hidden, blocks: hidden):
            hidden = model._encode_channels(values, marks)

        self.assertEqual(tuple(hidden.shape), (1, 105, args.d_model))
        self.assertEqual([int(branch_values.shape[1]) for branch_values, _ in calls], [14, 7])
        torch.testing.assert_close(
            calls[0][0][0, :, 0],
            torch.tensor([
                3.5, 11.5, 19.5, 27.5, 35.5, 43.5, 51.5,
                59.5, 67.5, 75.5, 83.5, 91.5, 99.5, 104.0,
            ]),
        )
        torch.testing.assert_close(
            calls[0][1][0, :, 0],
            torch.tensor(
                [7, 15, 23, 31, 39, 47, 55, 63, 71, 79, 87, 95, 103, 104],
                dtype=calls[0][1].dtype,
            ),
        )
        torch.testing.assert_close(
            calls[1][0][0, :, 0],
            torch.tensor([7.5, 23.5, 39.5, 55.5, 71.5, 87.5, 100.0]),
        )
        torch.testing.assert_close(
            calls[1][1][0, :, 0],
            torch.tensor([15, 31, 47, 63, 79, 95, 104], dtype=calls[1][1].dtype),
        )

    def test_token_stride_and_scales_reject_an_invalid_post_pool_scale(self):
        args = _args()
        args.mamba_token_stride = 96
        args.mamba_scales = [1, 2]
        with self.assertRaisesRegex(ValueError, "base tokens"):
            Model(args)

    def test_invalid_token_stride_is_rejected_before_gpu_execution(self):
        args = _args()
        args.mamba_token_stride = 0
        with self.assertRaisesRegex(ValueError, "mamba_token_stride must be positive"):
            Model(args)
        args.mamba_token_stride = 97
        with self.assertRaisesRegex(ValueError, "cannot exceed seq_len"):
            Model(args)

    def test_low_rank_head_uses_compressed_temporal_length(self):
        args = _args()
        args.pred_len = 96
        args.mamba_token_stride = 8
        args.mamba_low_rank_head = True
        args.mamba_temporal_rank = 11
        model = Model(args)
        self.assertEqual(model.temporal_input_length, 12)
        self.assertEqual(model.temporal_head.input_projection.in_features, 12)
        with self.assertRaisesRegex(ValueError, "smaller than min"):
            invalid = _args()
            invalid.pred_len = 96
            invalid.mamba_token_stride = 8
            invalid.mamba_low_rank_head = True
            invalid.mamba_temporal_rank = 12
            Model(invalid)

    def test_resource_profile_exposes_temporal_projection_accounting(self):
        full_args = _args(channels=7)
        full_args.pred_len = 96
        full = Model(full_args).resource_profile()

        low_args = _args(channels=7)
        low_args.pred_len = 96
        low_args.mamba_low_rank_head = True
        low_args.mamba_temporal_rank = 16
        low = Model(low_args).resource_profile()

        self.assertFalse(full["temporal_head_factorized"])
        self.assertIsNone(full["temporal_projection_rank"])
        self.assertEqual(full["temporal_head_macs_per_channel"], 96 * 96)
        self.assertEqual(full["linear_residual_macs_per_channel"], 96 * 96)
        self.assertTrue(low["temporal_head_factorized"])
        self.assertEqual(low["temporal_projection_rank"], 16)
        self.assertEqual(low["temporal_head_macs_per_channel"], 96 * 16 + 16 * 96)
        self.assertEqual(low["linear_residual_macs_per_channel"], 96 * 16 + 16 * 96)
        self.assertLess(
            low["temporal_projection_macs_per_window"],
            full["temporal_projection_macs_per_window"],
        )

    def test_acf_conditioned_variable_tokens_are_train_prior_only_and_differentiable(self):
        args = _args(channels=3, pred_len=24)
        args.mamba_variable_mixer = True
        args.mamba_acf_channel_conditioning = True
        args.mamba_acf_lags = [1, 2, 3]
        model = Model(args)
        model.set_acf_prior(torch.tensor([
            [1.0, 0.5, -0.2, 0.8],
            [1.0, -0.4, 0.3, -0.9],
            [1.0, 0.1, 0.2, -0.3],
        ]))

        summary = model._acf_channel_summary(torch.device("cpu"), torch.float32)
        self.assertEqual(tuple(summary.shape), (args.enc_in, 4))
        self.assertTrue(torch.allclose(
            summary[0], torch.tensor([0.36666667, 0.5, 0.27777779, 0.8]),
            atol=1e-6,
        ))
        tokens = torch.randn(2, args.enc_in, args.d_model, requires_grad=True)
        # ReZero makes an explicitly enabled but not-yet-trained branch an
        # exact identity, preserving the old variable-token initialization.
        self.assertTrue(torch.equal(model._acf_conditioned_variable_tokens(tokens), tokens))
        with torch.no_grad():
            model.acf_conditioning_scale.fill_(0.5)
        conditioned = model._acf_conditioned_variable_tokens(tokens)
        self.assertEqual(tuple(conditioned.shape), tuple(tokens.shape))
        self.assertFalse(torch.equal(conditioned, tokens))
        conditioned.square().mean().backward()
        self.assertIsNotNone(tokens.grad)
        self.assertIsNotNone(model.acf_conditioning_scale.grad)
        self.assertTrue(torch.isfinite(model.acf_conditioning_scale.grad).all())
        self.assertIsNotNone(model.acf_channel_conditioner[0].weight.grad)
        self.assertTrue(torch.isfinite(model.acf_channel_conditioner[0].weight.grad).all())
        profile = model.resource_profile()
        self.assertTrue(profile["variable_tokens_use_train_acf_prior"])
        self.assertEqual(profile["acf_channel_summary_width"], 4)

    def test_acf_channel_conditioning_requires_variable_mixer(self):
        args = _args()
        args.mamba_acf_channel_conditioning = True
        with self.assertRaisesRegex(ValueError, "requires mamba_variable_mixer"):
            Model(args)

    def test_acf_channel_gate_requires_channel_gate(self):
        args = _args()
        args.mamba_acf_channel_gate = True
        with self.assertRaisesRegex(ValueError, "requires mamba_channel_gate"):
            Model(args)

    def test_acf_channel_gate_is_independent_of_variable_mixer(self):
        args = _args(channels=5)
        args.mamba_channel_gate = True
        args.mamba_acf_channel_gate = True
        model = Model(args)
        model.set_acf_prior(torch.tensor([
            [0.0, 0.8, -0.2, 0.1],
            [0.0, -0.5, 0.4, 0.2],
            [0.0, 0.1, 0.7, -0.3],
            [0.0, 0.2, -0.6, 0.4],
            [0.0, 0.3, 0.2, 0.5],
        ]), lags=[1, 2, 3])
        profile = model.resource_profile()
        self.assertTrue(profile["acf_channel_gate_enabled"])
        self.assertTrue(profile["acf_channel_gate_prior_attached"])
        self.assertTrue(profile["acf_channel_gate_path_active"])
        self.assertGreater(profile["acf_channel_gate_parameters"], 0)
        self.assertFalse(profile["variable_token_channels"])

    def test_static_acf_memory_matches_reference_formula(self):
        for stepwise in (False, True):
            args = _args(channels=4, pred_len=20)
            args.seq_len = 32
            args.label_len = 16
            args.mamba_acf_lags = [3, 9, 40]
            args.mamba_acf_step_reliability = stepwise
            model = Model(args)
            prior = torch.zeros(4, 41)
            prior[:, 3] = torch.tensor([0.8, 0.1, -0.5, 0.2])
            prior[:, 9] = torch.tensor([-0.2, 0.7, 0.3, -0.8])
            prior[:, 40] = torch.tensor([0.4, -0.6, 0.9, 0.05])
            model.set_acf_prior(prior, lags=args.mamba_acf_lags)
            x = torch.randn(2, args.seq_len, args.enc_in, requires_grad=True)

            seasonal, reliability = model._static_acf_memory(x, args.pred_len)
            memory, corr, valid = model._causal_memory(x, args.pred_len)
            valid_weight = valid.to(dtype=memory.dtype)[None, :, None, :]
            logits = torch.log(corr.abs().clamp_min(1e-6))[:, None, :, :]
            logits = logits / model.acf_temperature
            logits = logits.masked_fill(
                ~valid[None, :, None, :], torch.finfo(memory.dtype).min
            )
            weights = torch.softmax(logits, dim=-1) * valid_weight
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-4)
            expected_seasonal = (
                memory * corr.sign()[:, None, :, :] * weights
            ).sum(dim=-1)
            expected_reliability = model._acf_memory_reliability(
                corr, weights, valid, stepwise=stepwise
            )
            if not stepwise:
                expected_reliability = expected_reliability.unsqueeze(1)
            torch.testing.assert_close(seasonal, expected_seasonal)
            # Static reliability is batch-independent and intentionally keeps
            # a singleton batch dimension; the forecast path broadcasts it to
            # the current batch.  Compare after that explicit broadcast.
            torch.testing.assert_close(
                reliability.expand_as(expected_reliability), expected_reliability
            )
            seasonal.sum().backward()
            self.assertTrue(torch.isfinite(x.grad).all())

            # A different input length/horizon must bypass the model-length
            # cache while retaining the same algebraic contract.
            short = torch.randn(2, 20, args.enc_in)
            short_seasonal, short_reliability = model._static_acf_memory(short, 10)
            short_memory, short_corr, short_valid = model._causal_memory(short, 10)
            short_valid_weight = short_valid.to(dtype=short_memory.dtype)[None, :, None, :]
            short_logits = torch.log(short_corr.abs().clamp_min(1e-6))[:, None, :, :]
            short_logits = short_logits / model.acf_temperature
            short_logits = short_logits.masked_fill(
                ~short_valid[None, :, None, :], torch.finfo(short_memory.dtype).min
            )
            short_weights = torch.softmax(short_logits, dim=-1) * short_valid_weight
            short_weights = short_weights / short_weights.sum(dim=-1, keepdim=True).clamp_min(1e-4)
            expected_short = (
                short_memory * short_corr.sign()[:, None, :, :] * short_weights
            ).sum(dim=-1)
            expected_short_rel = model._acf_memory_reliability(
                short_corr, short_weights, short_valid, stepwise=stepwise
            )
            if not stepwise:
                expected_short_rel = expected_short_rel.unsqueeze(1)
            torch.testing.assert_close(short_seasonal, expected_short)
            torch.testing.assert_close(
                short_reliability.expand_as(expected_short_rel), expected_short_rel
            )

    @staticmethod
    def _bare_memory_model(*, channels=2, seq_len=8, pred_len=4,
                           lags=(1, 3), source="residual", dynamic=True):
        """Construct the memory subsystem without requiring mamba_ssm."""

        model = object.__new__(Model)
        torch.nn.Module.__init__(model)
        model.channels = channels
        model.seq_len = seq_len
        model.pred_len = pred_len
        model.acf_lags = tuple(lags)
        model.acf_source = source
        model.dynamic_acf = dynamic
        model.acf_temperature = 4.0
        model.acf_step_reliability = False
        model.acf_prior = torch.empty(0)
        model.acf_memory_static_coefficients = torch.empty(0)
        model.acf_memory_static_step_reliability = torch.empty(0)
        model.acf_memory_static_aggregate_reliability = torch.empty(0)
        model.acf_memory_static_indices = torch.empty(0, dtype=torch.long)
        model.acf_memory_static_valid = torch.empty(0, dtype=torch.bool)
        model._acf_memory_static_length = None
        model._acf_memory_static_horizon = None
        model._acf_memory_static_lags = None
        model._acf_memory_static_temperature = None
        model._acf_memory_static_prior_identity = None
        model._acf_memory_static_prior_version = None
        return model

    def test_dynamic_memory_uses_the_selected_source_even_with_a_train_prior(self):
        values = torch.tensor([
            [0.0, 1.0], [1.0, 0.0], [2.0, -1.0], [3.0, 0.0],
            [4.0, 1.0], [5.0, 0.0], [6.0, -1.0], [7.0, 0.0],
        ]).unsqueeze(0)
        for source in ("residual", "trend"):
            model = self._bare_memory_model(source=source, dynamic=True)
            model.acf_prior = torch.tensor([[0.95, -0.8], [0.4, 0.9]])
            memory_source = model._memory_signal(values)
            _, corr, _ = model._causal_memory(values, model.pred_len)
            expected = torch.stack(
                [model._signed_acf(memory_source, lag) for lag in model.acf_lags],
                dim=-1,
            )
            torch.testing.assert_close(corr, expected)
            self.assertFalse(torch.allclose(
                corr, model.acf_prior.unsqueeze(0).expand_as(corr)
            ))

    def test_dynamic_forecast_branch_precedes_attached_static_prior(self):
        model = self._bare_memory_model(
            channels=2, seq_len=8, pred_len=3, lags=(1, 3),
            source="raw", dynamic=True,
        )
        model.d_model = 4
        model.channel_independent = False
        model.use_acf_memory = True
        model.variable_mixer_enabled = False
        model.patch_mixer_enabled = False
        model.local_mixer_enabled = False
        model.value_head = torch.nn.Linear(model.d_model, model.channels)
        model.temporal_head = torch.nn.Linear(model.seq_len, model.pred_len)
        model.linear_residual = torch.nn.Linear(model.seq_len, model.pred_len)
        model.residual_gate = torch.nn.Linear(model.d_model, model.pred_len)
        model.acf_gate = torch.nn.Linear(model.d_model, model.pred_len)
        model.acf_prior = torch.tensor([[0.9, -0.7], [0.6, 0.8]])
        hidden = torch.zeros(1, model.seq_len, model.d_model)
        values = torch.randn(1, model.seq_len, model.channels)

        with patch.object(
            model, "_static_acf_memory",
            side_effect=AssertionError("dynamic mode used static memory"),
        ) as static_memory, patch.object(
            model, "_causal_memory", wraps=model._causal_memory
        ) as dynamic_memory:
            forecast = model._forecast_normalized(hidden, values)

        self.assertEqual(tuple(forecast.shape), (1, model.pred_len, model.channels))
        static_memory.assert_not_called()
        dynamic_memory.assert_called_once()

    def test_static_memory_cache_rebuilds_after_inplace_prior_update(self):
        model = self._bare_memory_model(
            channels=1, seq_len=8, pred_len=3, lags=(1, 2),
            source="raw", dynamic=False,
        )
        model.acf_prior = torch.tensor([[0.9, -0.6]])
        model._refresh_static_acf_memory_cache()
        values = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1)
        before, _ = model._static_acf_memory(values, model.pred_len)
        with torch.no_grad():
            model.acf_prior.zero_()
        after, reliability = model._static_acf_memory(values, model.pred_len)

        self.assertFalse(torch.allclose(before, after))
        self.assertTrue(torch.allclose(after, torch.zeros_like(after)))
        self.assertTrue(torch.allclose(reliability, torch.zeros_like(reliability)))
        self.assertEqual(
            model._acf_memory_static_prior_version,
            model._acf_prior_tensor_version(model.acf_prior),
        )

    def test_channel_lag_bank_selects_deterministic_top_slots_per_channel(self):
        model = self._bare_memory_model(
            channels=2, seq_len=8, pred_len=3, lags=(1, 2, 3),
            source="raw", dynamic=False,
        )
        model.use_acf_memory = True
        model.acf_channel_lag_bank = True
        model.acf_channel_lag_top_k = 1
        model.acf_prior = torch.tensor([
            [0.9, 0.1, 0.2],
            [0.1, 0.8, 0.3],
        ])

        support = model._build_static_acf_channel_support()
        self.assertEqual(tuple(support.shape), (2, 3))
        self.assertTrue(torch.equal(
            support,
            torch.tensor([[True, False, False], [False, True, False]]),
        ))
        model._refresh_static_acf_memory_cache()
        self.assertTrue(torch.equal(model.acf_memory_static_channel_support, support))

    def test_channel_lag_bank_changes_only_static_memory_support(self):
        values = torch.arange(16, dtype=torch.float32).reshape(1, 8, 2)
        prior = torch.tensor([
            [0.9, 0.1, 0.2],
            [0.1, 0.8, 0.3],
        ])
        dense = self._bare_memory_model(
            channels=2, seq_len=8, pred_len=3, lags=(1, 2, 3),
            source="raw", dynamic=False,
        )
        sparse = self._bare_memory_model(
            channels=2, seq_len=8, pred_len=3, lags=(1, 2, 3),
            source="raw", dynamic=False,
        )
        for model, enabled in ((dense, False), (sparse, True)):
            model.use_acf_memory = True
            model.acf_channel_lag_bank = enabled
            model.acf_channel_lag_top_k = 1
            model.acf_prior = prior.clone()
            model._refresh_static_acf_memory_cache()
        dense_memory, dense_rel = dense._static_acf_memory(values, 3)
        sparse_memory, sparse_rel = sparse._static_acf_memory(values, 3)
        self.assertFalse(torch.allclose(dense_memory, sparse_memory))
        self.assertFalse(torch.allclose(dense_rel, sparse_rel))
        self.assertEqual(tuple(sparse_rel.shape), (1, 1, 2))
        self.assertTrue(torch.isfinite(sparse_memory).all())

    def test_anchor_cache_rebuilds_once_after_stale_identity(self):
        """A device move must not trigger anchor selection on every forward."""

        args = _args(channels=1, pred_len=3)
        args.seq_len = 8
        args.mamba_acf_lags = [1, 3]
        args.mamba_patch_len = 8
        args.mamba_acf_anchor_copy = True
        args.mamba_use_acf_memory = False
        args.AutoCon_acf_source = "raw"
        model = Model(args)
        model.set_acf_prior(torch.tensor([[0.8, 0.4, 0.6, 0.2]]), lags=args.mamba_acf_lags)
        values = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1)

        # ``Module.to`` can replace registered buffers and therefore change
        # the prior identity while retaining an otherwise valid cache.
        model._acf_memory_static_prior_identity = -1
        with patch.object(
            model,
            "_build_static_acf_anchor_terms",
            wraps=model._build_static_acf_anchor_terms,
        ) as builder:
            first, *_ = model._static_acf_anchor_memory(values, args.pred_len)
            second, *_ = model._static_acf_anchor_memory(values, args.pred_len)

        self.assertEqual(builder.call_count, 1)
        torch.testing.assert_close(first, second)

    def test_relation_only_prior_does_not_compile_forecast_caches(self):
        """A sidecar used only by the relation objective must stay metadata-only."""

        args = _args(channels=2, pred_len=4)
        args.mamba_use_acf_memory = False
        args.mamba_acf_anchor_copy = False
        args.AutoCon_acf_source = "raw"
        model = Model(args)
        model.set_acf_prior(torch.ones(2, 400), lags=[1, 3])

        self.assertEqual(model.acf_memory_static_coefficients.numel(), 0)
        self.assertEqual(model.acf_memory_static_indices.numel(), 0)
        self.assertEqual(model.acf_memory_static_anchor_indices.numel(), 0)
        self.assertIsNone(model._acf_memory_static_length)
        self.assertIsNone(model._acf_memory_static_prior_identity)

    def test_anchor_only_prior_compiles_anchor_but_not_soft_memory(self):
        """The direct retrieval branch should not pay for unused soft terms."""

        args = _args(channels=2, pred_len=4)
        args.mamba_use_acf_memory = False
        args.mamba_acf_anchor_copy = True
        args.AutoCon_acf_source = "raw"
        model = Model(args)
        model.set_acf_prior(torch.ones(2, 400), lags=[1, 3])

        self.assertEqual(model.acf_memory_static_coefficients.numel(), 0)
        self.assertEqual(model.acf_memory_static_indices.numel(), 0)
        self.assertEqual(
            tuple(model.acf_memory_static_anchor_indices.shape),
            (args.pred_len, args.enc_in),
        )
        self.assertEqual(model._acf_memory_static_length, args.seq_len)

    def test_eval_acf_diagnostics_are_lazy_but_explicitly_opt_in(self):
        model = Model(_args(channels=1, pred_len=3)).eval()
        model._set_acf_runtime_diagnostic("acf_seasonal_rms", torch.ones(2))
        self.assertIsNone(model.acf_runtime_diagnostics()["acf_seasonal_rms"])

        model.acf_collect_eval_diagnostics = True
        model._set_acf_runtime_diagnostic("acf_seasonal_rms", torch.ones(2))
        self.assertEqual(model.acf_runtime_diagnostics()["acf_seasonal_rms"], 1.0)

    def test_runtime_diagnostics_batch_transfer_preserves_python_snapshot(self):
        model = Model(_args(channels=1, pred_len=3)).train()
        model._set_acf_runtime_diagnostic("acf_seasonal_rms", torch.tensor([3.0, 4.0]))
        model._set_acf_runtime_diagnostic("acf_reliability_mean", torch.tensor([2.0, 4.0]))
        raw_values = model._acf_runtime_diagnostics
        self.assertTrue(torch.is_tensor(raw_values["acf_seasonal_rms"]))
        self.assertTrue(torch.is_tensor(raw_values["acf_reliability_mean"]))
        snapshot = model.acf_runtime_diagnostics()
        self.assertIsInstance(snapshot["acf_seasonal_rms"], float)
        self.assertAlmostEqual(snapshot["acf_seasonal_rms"], 3.5355339, places=5)
        self.assertAlmostEqual(snapshot["acf_reliability_mean"], 3.0, places=6)

    def test_effective_memory_relative_rms_is_detached_telemetry(self):
        """Relative ACF strength is observable without changing forecast tensors."""

        model = object.__new__(Model)
        torch.nn.Module.__init__(model)
        model.train()
        base_forecast = torch.tensor([[2.0, -2.0]], requires_grad=True)
        effective_memory = torch.tensor([[0.2, -0.2]], requires_grad=True)
        base_snapshot = base_forecast.detach().clone()
        model._record_acf_effective_memory_strength(
            effective_memory, base_forecast
        )

        diagnostics = model.acf_runtime_diagnostics()
        self.assertAlmostEqual(
            diagnostics["acf_effective_memory_to_forecast_rms"], 0.1, places=6
        )
        torch.testing.assert_close(base_forecast.detach(), base_snapshot)
        self.assertIsNone(base_forecast.grad)
        self.assertIsNone(effective_memory.grad)

    def test_reliability_mode_and_anchor_lag_support_are_explicit(self):
        """Profile records stepwise mode and the independent anchor support policy."""

        aggregate_args = _args(channels=2, pred_len=3)
        aggregate = Model(aggregate_args)
        aggregate_profile = aggregate.resource_profile()
        self.assertFalse(aggregate_profile["acf_step_reliability"])
        self.assertEqual(aggregate_profile["acf_reliability_mode"], "aggregate")

        stepwise_args = _args(channels=2, pred_len=3)
        stepwise_args.mamba_acf_step_reliability = True
        stepwise = Model(stepwise_args)
        stepwise_profile = stepwise.resource_profile()
        self.assertTrue(stepwise_profile["acf_step_reliability"])
        self.assertEqual(stepwise_profile["acf_reliability_mode"], "stepwise")
        stepwise.set_acf_prior(
            torch.ones(2, 337), lags=stepwise_args.mamba_acf_lags
        )
        self.assertTrue(stepwise.resource_profile()["acf_step_reliability_path_active"])

        args = _args(channels=2, pred_len=3)
        args.seq_len = 8
        args.mamba_patch_len = 4
        args.mamba_patch_stride = 4
        args.mamba_acf_lags = [1, 2, 3]
        args.mamba_use_acf_memory = True
        args.mamba_acf_channel_lag_bank = True
        args.mamba_acf_channel_lag_top_k = 1
        args.mamba_acf_anchor_copy = True
        model = Model(args)
        model.set_acf_prior(
            # ``set_acf_prior`` consumes a complete table indexed by lag;
            # column zero is the lag-0 placeholder and is not selected.
            torch.tensor([
                [0.0, 0.95, 0.10, 0.20],
                [0.0, 0.10, 0.80, 0.20],
            ]),
            lags=args.mamba_acf_lags,
        )
        support = model.acf_memory_static_channel_support
        # The bank scores lag 1 highest for channel 0 after legal-step
        # weighting, while the anchor chooses lag 3 at the final legal step.
        self.assertTrue(torch.equal(support[0], torch.tensor([True, False, False])))
        self.assertEqual(int(model.acf_memory_static_anchor_lags[2, 0]), 3)
        profile = model.resource_profile()
        self.assertEqual(
            profile["acf_anchor_copy_lag_support"], "full_configured_lag_bank"
        )
        self.assertFalse(profile["acf_anchor_copy_uses_channel_lag_bank"])


@unittest.skipUnless(torch.cuda.is_available(), "installed mamba_ssm requires CUDA")
class MambaACFContractTests(unittest.TestCase):
    def _run(self, *, channels=3, c_out=None, channel_independent=False, pred_len=24,
             low_rank_head=False, local_mixer=False, token_stride=1, scales=None,
             scale_conditioning=False, model_cls=Model):
        args = _args(channels, c_out, channel_independent=channel_independent, pred_len=pred_len)
        args.mamba_low_rank_head = low_rank_head
        args.mamba_temporal_rank = 16
        args.mamba_local_mixer = local_mixer
        args.mamba_token_stride = token_stride
        if scales is not None:
            args.mamba_scales = list(scales)
        args.mamba_acf_scale_conditioning = bool(scale_conditioning)
        if scale_conditioning:
            args.AutoCon_acf_source = "raw"
        if local_mixer:
            args.mamba_local_width = 12
            args.mamba_local_rank = 4
        model = model_cls(args).cuda()
        # CPU prior is the normal train-only fitting path.  set_acf_prior must
        # move it safely to the model device.
        model.set_acf_prior(torch.randn(channels, 400), lags=args.mamba_acf_lags)
        if scale_conditioning:
            # Open the ReZero path for this contract test; production runs
            # start at zero so the ablation is initially behavior-preserving.
            with torch.no_grad():
                model.acf_scale_conditioning_scale.fill_(0.25)
        x = torch.randn(2, args.seq_len, channels, device="cuda", requires_grad=True)
        x_mark = torch.randn(2, args.seq_len, 4, device="cuda")
        dec = torch.zeros(2, args.label_len + args.pred_len, channels, device="cuda")
        dec_mark = torch.randn(2, args.label_len + args.pred_len, 4, device="cuda")
        prediction, representation = model(x, x_mark, dec, dec_mark)
        (prediction.square().mean() + representation.square().mean()).backward()
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue(torch.isfinite(representation).all())
        return model, x, prediction, representation

    def test_shared_output_and_representation_contract(self):
        _, _, prediction, representation = self._run(channels=7, pred_len=720)
        self.assertEqual(tuple(prediction.shape), (2, 720, 7))
        self.assertEqual(tuple(representation.shape), (2, 96, 16))

    def test_token_stride_preserves_cuda_output_and_representation_contract(self):
        _, _, prediction, representation = self._run(
            channels=7, pred_len=96, token_stride=8, scales=[1, 2]
        )
        self.assertEqual(tuple(prediction.shape), (2, 96, 7))
        self.assertEqual(tuple(representation.shape), (2, 96, 16))

    def test_acf_scale_conditioning_forward_and_gradient_contract(self):
        model, _, prediction, representation = self._run(
            channels=3,
            pred_len=96,
            scales=[1, 2],
            scale_conditioning=True,
        )
        self.assertEqual(tuple(prediction.shape), (2, 96, 3))
        self.assertEqual(tuple(representation.shape), (2, 96, 16))
        scale_grad = model.acf_scale_conditioning_scale.grad
        self.assertIsNotNone(scale_grad)
        self.assertTrue(torch.isfinite(scale_grad).all())
        self.assertGreater(float(scale_grad.abs()), 0.0)

    def test_acf_scale_conditioning_channel_independent_contract(self):
        _, _, prediction, representation = self._run(
            channels=3,
            pred_len=96,
            channel_independent=True,
            scales=[1, 2],
            scale_conditioning=True,
        )
        self.assertEqual(tuple(prediction.shape), (2, 96, 3))
        self.assertEqual(tuple(representation.shape), (2, 96, 16))

    def test_mamba1_token_stride_forward_and_gradient_contract(self):
        model, x, prediction, representation = self._run(
            channels=3,
            pred_len=96,
            token_stride=8,
            scales=[1, 2],
            scale_conditioning=True,
            model_cls=MambaACF1Model,
        )
        self.assertEqual(tuple(prediction.shape), (2, 96, 3))
        self.assertEqual(tuple(representation.shape), (2, 96, 16))
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertIsNotNone(model.value_head.weight.grad)
        self.assertTrue(torch.isfinite(model.value_head.weight.grad).all())

    def test_ms_projection_and_independent_ablation_contract(self):
        _, _, prediction, _ = self._run(channels=7, c_out=1, channel_independent=False)
        self.assertEqual(tuple(prediction.shape), (2, 24, 1))
        independent_model, _, prediction, _ = self._run(channels=3, channel_independent=True)
        self.assertEqual(tuple(prediction.shape), (2, 24, 3))
        self.assertIsNotNone(independent_model.embedding.value_embedding.tokenConv.weight.grad)
        self.assertTrue(torch.isfinite(independent_model.embedding.value_embedding.tokenConv.weight.grad).all())
        self.assertIsNotNone(independent_model.value_head.weight.grad)
        self.assertTrue(torch.isfinite(independent_model.value_head.weight.grad).all())

    def test_channel_modes_report_scan_and_rank_contract(self):
        """Resource metadata makes the high-dimensional fairness trade-off explicit."""

        args = _args(channels=7, pred_len=24)
        args.mamba_scales = [1, 2]
        args.mamba_bidirectional = True
        shared = Model(args).cuda().eval()
        shared_profile = shared.resource_profile()
        self.assertEqual(shared_profile["channel_mode"], "shared_temporal")
        self.assertEqual(shared_profile["embedding_input_channels"], 7)
        self.assertEqual(shared_profile["main_scan_streams_per_batch_item"], 4)
        self.assertEqual(shared_profile["main_scan_tokens_per_stream"], 96)
        self.assertEqual(shared_profile["shared_mamba_component_channel_rank_bound"], 16)
        self.assertTrue(shared_profile["linear_residual_is_channel_preserving"])

        args = _args(channels=7, pred_len=24, channel_independent=True)
        args.mamba_scales = [1, 2]
        args.mamba_bidirectional = True
        independent = Model(args).cuda().eval()
        independent_profile = independent.resource_profile()
        self.assertEqual(independent_profile["channel_mode"], "channel_independent")
        self.assertEqual(independent_profile["embedding_input_channels"], 1)
        self.assertEqual(independent_profile["main_scan_streams_per_batch_item"], 28)
        self.assertEqual(independent_profile["main_scan_tokens_per_stream"], 96)
        self.assertEqual(independent_profile["shared_mamba_component_channel_rank_bound"], 0)
        self.assertEqual(
            independent_profile["channel_preserving_forecast_branches"],
            ["independent_temporal"],
        )

        args = _args(channels=7, pred_len=24)
        args.mamba_variable_mixer = True
        args.mamba_variable_mixer_depth = 2
        args.mamba_bidirectional = True
        variable = Model(args).cuda().eval()
        variable_profile = variable.resource_profile()
        self.assertEqual(variable_profile["channel_mode"], "shared_temporal_plus_variable_tokens")
        self.assertEqual(variable_profile["main_scan_streams_per_batch_item"], 2)
        self.assertEqual(variable_profile["variable_scan_streams_per_batch_item"], 2)
        self.assertEqual(variable_profile["variable_scan_tokens_per_stream"], 7)
        self.assertEqual(variable_profile["shared_mamba_component_channel_rank_bound"], 16)
        self.assertEqual(
            variable_profile["channel_preserving_forecast_branches"],
            ["variable_tokens"],
        )

    def test_channel_independent_forecast_is_permutation_equivariant(self):
        """Independent temporal weights must not depend on CSV variable order."""

        args = _args(channels=5, pred_len=24, channel_independent=True)
        args.mamba_use_acf_memory = False
        model = Model(args).cuda().eval()
        x = torch.randn(2, args.seq_len, args.enc_in, device="cuda")
        mark = torch.randn(2, args.seq_len, 4, device="cuda")
        dec = torch.zeros(2, args.label_len + args.pred_len, args.enc_in, device="cuda")
        dec_mark = torch.randn(2, args.label_len + args.pred_len, 4, device="cuda")
        permutation = torch.tensor([3, 0, 4, 1, 2], device="cuda")
        with torch.no_grad():
            prediction, representation = model(x, mark, dec, dec_mark)
            permuted_prediction, permuted_representation = model(
                x[:, :, permutation], mark, dec[:, :, permutation], dec_mark
            )
        self.assertTrue(
            torch.allclose(
                permuted_prediction,
                prediction[:, :, permutation],
                atol=2e-5,
                rtol=2e-5,
            )
        )
        # The pooled AutoCon representation is invariant to a channel reorder.
        self.assertTrue(
            torch.allclose(
                permuted_representation,
                representation,
                atol=2e-5,
                rtol=2e-5,
            )
        )

    def test_long_lags_report_direct_availability_only(self):
        model, x, _, _ = self._run(channels=3, pred_len=96)
        _, _, valid = model._causal_memory(x.detach(), 96)
        expected = torch.tensor([0.25, 0.25, 0.0], device=valid.device)
        self.assertTrue(torch.allclose(valid.float().mean(dim=0), expected))

    def test_stepwise_memory_reliability_tracks_legal_horizon_steps(self):
        # With a single lag, confidence is one when the source is legal and
        # zero after the source leaves the encoder window.  This is the
        # distinction the optional stepwise gate is intended to preserve.
        corr = torch.tensor([[[0.8]]])
        weights = torch.ones(1, 4, 1, 1)
        valid = torch.tensor([[True], [True], [False], [False]])
        stepwise = Model._acf_memory_reliability(corr, weights, valid, stepwise=True)
        aggregate = Model._acf_memory_reliability(corr, weights, valid, stepwise=False)
        self.assertEqual(tuple(stepwise.shape), (1, 4, 1))
        self.assertTrue(torch.allclose(stepwise.squeeze(-1), torch.tensor([[0.8, 0.8, 0.0, 0.0]])))
        self.assertTrue(torch.allclose(aggregate, torch.tensor([[0.4]])))

    def test_invalid_scale_is_rejected_before_gpu_execution(self):
        args = _args()
        args.mamba_scales = [1, 128]
        with self.assertRaises(ValueError):
            Model(args)

    def test_low_rank_head_preserves_mamba_acf_forward_contract(self):
        full_model, _, full_prediction, _ = self._run(
            channels=7, pred_len=720, low_rank_head=False
        )
        low_rank_model, _, low_rank_prediction, _ = self._run(
            channels=7, pred_len=720, low_rank_head=True
        )
        self.assertEqual(tuple(low_rank_prediction.shape), (2, 720, 7))
        self.assertTrue(low_rank_model.temporal_head.is_low_rank)
        self.assertTrue(low_rank_model.linear_residual.is_low_rank)
        self.assertIsInstance(full_model.temporal_head, torch.nn.Linear)
        self.assertIsInstance(full_model.linear_residual, torch.nn.Linear)
        self.assertLess(
            sum(parameter.numel() for parameter in low_rank_model.parameters()),
            sum(parameter.numel() for parameter in full_model.parameters()),
        )

    def test_local_patch_branch_preserves_cuda_forward_and_autocon_contract(self):
        model, _, prediction, representation = self._run(
            channels=7, pred_len=24, local_mixer=True
        )
        self.assertEqual(tuple(prediction.shape), (2, 24, 7))
        self.assertEqual(tuple(representation.shape), (2, 96, 16))
        self.assertIsNotNone(model.local_residual_scale.grad)
        self.assertIsNotNone(model.local_repr_scale.grad)
        self.assertTrue(torch.isfinite(model.local_residual_scale.grad).all())
        self.assertTrue(torch.isfinite(model.local_repr_scale.grad).all())

    def test_variable_token_branch_preserves_contract_and_has_gradient(self):
        args = _args(channels=7, pred_len=96)
        args.mamba_variable_mixer = True
        args.mamba_variable_mixer_depth = 1
        args.mamba_variable_permutation = True
        model = Model(args).cuda().train()
        model.set_acf_prior(torch.tensor([0.8, 0.2, -0.3]), lags=args.mamba_acf_lags)
        x = torch.randn(2, args.seq_len, args.enc_in, device="cuda", requires_grad=True)
        mark = torch.randn(2, args.seq_len, 4, device="cuda")
        dec = torch.zeros(2, args.label_len + args.pred_len, args.enc_in, device="cuda")
        dec_mark = torch.randn_like(torch.zeros(2, args.label_len + args.pred_len, 4, device="cuda"))
        prediction, representation = model(x, mark, dec, dec_mark)
        (prediction.square().mean() + representation.square().mean()).backward()
        self.assertEqual(tuple(prediction.shape), (2, args.pred_len, args.enc_in))
        self.assertEqual(tuple(representation.shape), (2, args.seq_len, args.d_model))
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertIsNotNone(model.variable_residual_scale.grad)
        self.assertTrue(torch.isfinite(model.variable_residual_scale.grad).all())
        # ReZero starts the optional branch at zero.  Its scale receives the
        # first update; after that update the branch parameters must participate
        # in backpropagation rather than remaining permanently disconnected.
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.step()
        model.zero_grad(set_to_none=True)
        second_prediction, _ = model(x, mark, dec, dec_mark)
        second_prediction.square().mean().backward()
        self.assertIsNotNone(model.variable_embedding.weight.grad)
        self.assertTrue(torch.isfinite(model.variable_embedding.weight.grad).all())

    def test_strength_order_is_restored_after_variable_scan(self):
        args = _args(channels=4, pred_len=24)
        args.mamba_variable_mixer = True
        args.mamba_acf_channel_order = "strength"
        args.mamba_acf_lags = [24, 48, 72]
        model = Model(args).cuda().eval()
        prior = torch.zeros(4, 73)
        prior[:, 24] = torch.tensor([0.2, -0.9, 0.6, 0.6])
        prior[:, 48] = torch.tensor([-0.3, 0.1, 0.2, -0.1])
        prior[:, 72] = torch.tensor([0.4, 0.2, -0.1, 0.2])
        model.set_acf_prior(prior, lags=args.mamba_acf_lags)
        order, inverse = model._train_prior_channel_order(torch.device("cuda"))
        self.assertIsNotNone(order)
        self.assertIsNotNone(inverse)
        tokens = torch.randn(2, args.enc_in, args.d_model, device="cuda")
        with torch.no_grad():
            scanned = model.variable_norm(
                model._run_variable_branch(tokens.index_select(1, order))
            )
            restored = scanned.index_select(1, inverse)
        expected = torch.empty_like(scanned)
        expected[:, order, :] = scanned
        self.assertTrue(torch.allclose(restored, expected, atol=2e-5, rtol=2e-5))
        self.assertEqual(tuple(restored.shape), tuple(tokens.shape))

    def test_one_dimensional_acf_prior_broadcasts_per_channel(self):
        args = _args(channels=5, pred_len=24)
        model = Model(args).cuda()
        model.set_acf_prior(torch.tensor([0.5, -0.2, 0.1]), lags=[24, 168, 336])
        self.assertEqual(tuple(model.acf_prior.shape), (args.enc_in, 3))
        self.assertTrue(torch.allclose(model.acf_prior[0], model.acf_prior[-1]))

    def test_channel_patch_branch_preserves_contract_and_has_gradient(self):
        args = _args(channels=7, pred_len=96)
        args.mamba_patch_mixer = True
        args.mamba_patch_mixer_depth = 1
        args.mamba_patch_len = 16
        args.mamba_patch_stride = 8
        args.mamba_patch_permutation = True
        model = Model(args).cuda().train()
        model.set_acf_prior(
            torch.zeros(args.enc_in, max(args.mamba_acf_lags) + 1),
            lags=args.mamba_acf_lags,
        )
        x = torch.randn(2, args.seq_len, args.enc_in, device="cuda", requires_grad=True)
        mark = torch.randn(2, args.seq_len, 4, device="cuda")
        dec = torch.zeros(2, args.label_len + args.pred_len, args.enc_in, device="cuda")
        dec_mark = torch.randn(2, args.label_len + args.pred_len, 4, device="cuda")
        prediction, representation = model(x, mark, dec, dec_mark)
        (prediction.square().mean() + representation.square().mean()).backward()
        self.assertEqual(tuple(prediction.shape), (2, args.pred_len, args.enc_in))
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertIsNotNone(model.patch_residual_scale.grad)
        self.assertTrue(torch.isfinite(model.patch_residual_scale.grad).all())

    def test_variable_repr_fusion_preserves_cuda_forward_and_relation_gradient(self):
        args = _args(channels=3, pred_len=24)
        args.mamba_variable_mixer = True
        args.mamba_variable_repr_fusion = True
        model = Model(args).cuda().train()
        model.set_acf_prior(
            torch.zeros(args.enc_in, max(args.mamba_acf_lags) + 1),
            lags=args.mamba_acf_lags,
        )
        # At least three ordered windows are needed: with only two windows and
        # one selected lag, every anchor has exactly one candidate and the
        # relation cross-entropy is identically zero.
        batch_size = 3
        x = torch.randn(
            batch_size,
            args.seq_len,
            args.enc_in,
            device="cuda",
            requires_grad=True,
        )
        mark = torch.randn(batch_size, args.seq_len, 4, device="cuda")
        dec = torch.zeros(
            batch_size,
            args.label_len + args.pred_len,
            args.enc_in,
            device="cuda",
        )
        dec_mark = torch.randn(
            batch_size,
            args.label_len + args.pred_len,
            4,
            device="cuda",
        )
        prediction, representation = model(x, mark, dec, dec_mark)
        relation = ACFRelationLoss(
            torch.tensor([[0.0, 0.8] + [0.0] * (args.seq_len - 1)]),
            relation_scope="global",
            selected_lags=[1],
        ).cuda()
        _, global_relation = relation(
            representation, torch.tensor([0, 1, 2], device="cuda")
        )
        objective = prediction.square().mean() + global_relation.mean()
        objective.backward()
        self.assertEqual(
            tuple(prediction.shape), (batch_size, args.pred_len, args.enc_in)
        )
        self.assertEqual(
            tuple(representation.shape), (batch_size, args.seq_len, args.d_model)
        )
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue(torch.isfinite(representation).all())
        self.assertIsNotNone(model.variable_embedding.weight.grad)
        self.assertGreater(
            float(model.variable_embedding.weight.grad.abs().sum()), 0.0
        )
        self.assertIsNotNone(model.variable_repr_projection[0].weight.grad)

    def test_invalid_patch_width_is_rejected_before_gpu_execution(self):
        args = _args()
        args.mamba_patch_mixer = True
        args.mamba_patch_len = args.seq_len + 1
        with self.assertRaises(ValueError):
            Model(args)

    def test_local_patch_branch_preserves_contract_and_reports_channel_cost(self):
        args = _args(channels=7, pred_len=24)
        args.mamba_local_mixer = True
        args.mamba_local_patch_len = 16
        args.mamba_local_patch_stride = 8
        args.mamba_local_width = 12
        args.mamba_local_rank = 4
        args.mamba_local_depth = 2
        model = Model(args)
        x = torch.randn(3, args.seq_len, args.enc_in, requires_grad=True)
        tokens = model._local_patch_features(x)
        self.assertEqual(
            tuple(tokens.shape),
            (3, args.enc_in, model.local_patch_count, args.mamba_local_width),
        )
        correction = model._local_forecast(x, tokens)
        self.assertEqual(tuple(correction.shape), (3, args.pred_len, args.enc_in))
        profile = model.resource_profile()
        self.assertTrue(profile["local_patch_enabled"])
        self.assertTrue(profile["local_patch_channel_preserving"])
        self.assertEqual(profile["local_patch_cross_channel_parameters"], 0)
        self.assertEqual(
            profile["local_patch_tokens_per_batch_item"],
            args.enc_in * model.local_patch_count,
        )
        correction.square().mean().backward()
        self.assertIsNotNone(model.local_residual_scale.grad)
        self.assertTrue(torch.isfinite(model.local_residual_scale.grad).all())

    def test_local_patch_representation_adapter_is_autocon_only(self):
        args = _args(channels=4, pred_len=24)
        args.mamba_local_mixer = True
        args.mamba_local_width = 12
        args.mamba_local_rank = 4
        model = Model(args)
        self.assertTrue(model.local_repr_projection.weight.requires_grad)
        self.assertTrue(model.local_repr_scale.requires_grad)

        args.AutoCon = False
        plain = Model(args)
        self.assertFalse(plain.local_repr_projection.weight.requires_grad)
        self.assertFalse(plain.local_repr_scale.requires_grad)

    def test_local_patch_branch_rejects_independent_mode_and_bad_rank(self):
        args = _args(channels=4, pred_len=24, channel_independent=True)
        args.mamba_local_mixer = True
        with self.assertRaisesRegex(ValueError, "channel_independent"):
            Model(args)

        args = _args(channels=4, pred_len=8)
        args.mamba_local_mixer = True
        args.mamba_local_rank = 8
        with self.assertRaisesRegex(ValueError, "mamba_local_rank"):
            Model(args)


if __name__ == "__main__":
    unittest.main()
