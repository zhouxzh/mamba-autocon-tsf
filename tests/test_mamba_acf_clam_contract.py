"""Contract tests for the optional causal local-ACF memory modulation."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

import models.MambaACF as mamba_acf


class _IdentityMamba(nn.Module):
    """Small CPU/GPU stand-in so these tests do not require Triton kernels."""

    def __init__(self, **_kwargs):
        super().__init__()

    def forward(self, values):
        return values


def _args(*, channels=2, seq_len=32, pred_len=8, lags=(4, 12, 24), local=True,
          min_pairs=16):
    return SimpleNamespace(
        seq_len=seq_len,
        label_len=max(1, seq_len // 2),
        pred_len=pred_len,
        enc_in=channels,
        c_out=channels,
        d_model=8,
        d_ff=16,
        e_layers=1,
        dropout=0.0,
        AutoCon=False,
        AutoCon_wnorm="LastVal",
        AutoCon_acf_source="raw",
        embed="timeF",
        freq="h",
        wo_freq="",
        mamba_d_state=4,
        mamba_d_conv=2,
        mamba_expand=2,
        mamba_headdim=4,
        mamba_scales=[1],
        mamba_token_stride=1,
        mamba_bidirectional=False,
        mamba_channel_independent=False,
        mamba_channel_gate=False,
        mamba_channel_gate_rank=4,
        mamba_acf_scale_conditioning=False,
        mamba_acf_channel_gate=False,
        mamba_causal_embedding=False,
        mamba_variable_mixer=False,
        mamba_variable_mixer_depth=1,
        mamba_variable_group_size=1,
        mamba_variable_permutation=False,
        mamba_acf_channel_order="input",
        mamba_acf_channel_conditioning=False,
        mamba_acf_repr_fusion=False,
        mamba_patch_mixer=False,
        mamba_patch_mixer_depth=1,
        mamba_patch_len=8,
        mamba_patch_stride=4,
        mamba_patch_order="time_major",
        mamba_patch_permutation=False,
        mamba_local_mixer=False,
        mamba_local_patch_len=8,
        mamba_local_patch_stride=4,
        mamba_local_width=8,
        mamba_local_rank=4,
        mamba_local_depth=1,
        mamba_low_rank_head=False,
        mamba_temporal_rank=4,
        mamba_acf_lags=list(lags),
        mamba_use_acf_memory=True,
        mamba_dynamic_acf=False,
        mamba_acf_temperature=4.0,
        mamba_acf_step_reliability=False,
        mamba_acf_horizon_calibration=False,
        mamba_acf_horizon_calibration_rank=2,
        mamba_acf_memory_gate_mode="hidden",
        mamba_acf_anchor_copy=False,
        mamba_acf_local_agreement=local,
        mamba_acf_local_min_pairs=min_pairs,
    )


def _make_model(args):
    # mamba_ssm is an optional CUDA dependency.  The CLAM helper and forecast
    # head are ordinary torch operations, so an identity mixer is sufficient
    # to test their contracts on either CPU or CUDA.
    with patch.object(mamba_acf, "_Mamba2", _IdentityMamba), \
            patch.object(mamba_acf, "_MAMBA2_IMPORT_ERROR", None):
        return mamba_acf.Model(args)


def _full_prior(values, lags):
    """Build the full ACF table expected by ``set_acf_prior``."""

    values = torch.as_tensor(values, dtype=torch.float32)
    lags = tuple(int(lag) for lag in lags)
    table = torch.zeros(values.shape[0], max(lags) + 1, dtype=torch.float32)
    table[:, list(lags)] = values
    return table


class MambaACFCLAMTests(unittest.TestCase):
    def test_prediction_only_path_preserves_forecast_and_omits_representation(self):
        args = _args(local=False)
        args.AutoCon = True
        args.mamba_use_acf_memory = False
        model = _make_model(args).eval()
        values = torch.randn(2, args.seq_len, args.enc_in)
        marks = torch.randn(2, args.seq_len, 4)
        decoder = torch.zeros(2, args.label_len + args.pred_len, args.enc_in)
        decoder_marks = torch.randn(2, args.label_len + args.pred_len, 4)

        prediction, representation = model(values, marks, decoder, decoder_marks)
        prediction_only, omitted = model(
            values, marks, decoder, decoder_marks, return_repr=False
        )
        torch.testing.assert_close(prediction, prediction_only)
        self.assertIsNotNone(representation)
        self.assertIsNone(omitted)

        _, forced = model(
            values, marks, decoder, decoder_marks,
            onlyrepr=True, return_repr=False,
        )
        torch.testing.assert_close(representation, forced)

    def test_support_requires_at_least_sixteen_pairs(self):
        args = _args(seq_len=32, lags=(8, 17, 31), local=True)
        model = _make_model(args)
        eligible, pair_count = model._acf_local_agreement_support(32)
        self.assertEqual(eligible, (8,))
        self.assertEqual(pair_count, 24)

        profile = model.resource_profile()
        self.assertTrue(profile["acf_local_agreement_requested"])
        self.assertFalse(profile["acf_local_agreement_path_active"])
        self.assertEqual(profile["acf_local_agreement_eligible_lag_count"], 1)
        self.assertEqual(profile["acf_local_agreement_pair_count"], 24)
        self.assertEqual(profile["acf_local_agreement_parameters"], 1)

    def test_constructor_rejects_non_raw_local_source(self):
        args = _args(local=True)
        args.AutoCon_acf_source = "residual"
        with self.assertRaisesRegex(ValueError, "AutoCon_acf_source=raw"):
            _make_model(args)

    def test_constructor_accepts_lowercase_source_alias(self):
        args = _args(local=True)
        delattr(args, "AutoCon_acf_source")
        args.autocon_acf_source = "raw"
        model = _make_model(args)
        self.assertEqual(model.acf_source, "raw")

    def test_constructor_prefers_lowercase_source_when_aliases_conflict(self):
        args = _args(local=True)
        args.AutoCon_acf_source = "residual"
        args.autocon_acf_source = "raw"
        model = _make_model(args)
        self.assertEqual(model.acf_source, "raw")

    def test_constructor_rejects_non_raw_lowercase_source_alias(self):
        args = _args(local=True)
        delattr(args, "AutoCon_acf_source")
        args.autocon_acf_source = "residual"
        with self.assertRaisesRegex(ValueError, "AutoCon_acf_source=raw"):
            _make_model(args)

    def test_no_eligible_lag_falls_back_exactly_to_static_memory(self):
        args_on = _args(seq_len=32, lags=(20, 24), local=True)
        args_off = _args(seq_len=32, lags=(20, 24), local=False)
        model_on = _make_model(args_on)
        model_off = _make_model(args_off)
        prior = _full_prior([[0.8, -0.6], [0.4, 0.9]], args_on.mamba_acf_lags)
        model_on.set_acf_prior(prior, lags=args_on.mamba_acf_lags)
        model_off.set_acf_prior(prior, lags=args_off.mamba_acf_lags)
        values = torch.randn(3, 32, 2)

        agreement, eligible, pairs = model_on._acf_local_agreement_values(values)
        self.assertIsNone(agreement)
        self.assertFalse(eligible.any())
        self.assertEqual(pairs, 0)

        seasonal_on, reliability_on = model_on._static_acf_memory(values, 8)
        seasonal_off, reliability_off = model_off._static_acf_memory(values, 8)
        self.assertTrue(torch.equal(seasonal_on, seasonal_off))
        self.assertTrue(torch.equal(reliability_on, reliability_off))
        profile = model_on.resource_profile()
        self.assertFalse(profile["acf_local_agreement_path_active"])

    def test_zero_prior_slot_keeps_baseline_mass_at_zero_scale(self):
        for channel_independent in (False, True):
            args_on = _args(seq_len=32, lags=(4, 12, 24), local=True)
            args_off = _args(seq_len=32, lags=(4, 12, 24), local=False)
            args_on.mamba_channel_independent = channel_independent
            args_off.mamba_channel_independent = channel_independent
            model_on = _make_model(args_on)
            model_off = _make_model(args_off)
            prior = _full_prior(
                [[0.9, 0.0, 0.3], [-0.8, 0.6, 0.0]],
                args_on.mamba_acf_lags,
            )
            model_on.set_acf_prior(prior, lags=args_on.mamba_acf_lags)
            model_off.set_acf_prior(prior, lags=args_off.mamba_acf_lags)
            values = torch.randn(2, 32, 2)
            baseline, _ = model_off._static_acf_memory(values, 8)
            nested, _ = model_on._static_acf_memory(values, 8)
            torch.testing.assert_close(nested, baseline, rtol=1e-5, atol=1e-6)

    def test_zero_init_is_nested_and_open_scale_has_gradient(self):
        args_on = _args(seq_len=32, pred_len=8, lags=(4, 12, 24), local=True)
        args_off = _args(seq_len=32, pred_len=8, lags=(4, 12, 24), local=False)
        model_on = _make_model(args_on)
        model_off = _make_model(args_off)
        prior = _full_prior(
            [[0.9, -0.7, 0.3], [-0.8, 0.6, 0.4]],
            args_on.mamba_acf_lags,
        )
        model_on.set_acf_prior(prior, lags=args_on.mamba_acf_lags)
        model_off.set_acf_prior(prior, lags=args_off.mamba_acf_lags)
        positions = torch.arange(32, dtype=torch.float32)
        values = torch.stack([
            torch.sin(2.0 * torch.pi * positions / 4.0),
            torch.cos(2.0 * torch.pi * positions / 6.0),
        ], dim=-1).unsqueeze(0).repeat(2, 1, 1)

        self.assertEqual(model_on.acf_local_agreement_scale.item(), 0.0)
        base, _ = model_off._static_acf_memory(values, 8)
        nested, _ = model_on._static_acf_memory(values, 8)
        torch.testing.assert_close(nested, base, rtol=1e-5, atol=1e-6)

        with torch.no_grad():
            model_on.acf_local_agreement_scale.fill_(0.4)
        opened, _ = model_on._static_acf_memory(values.requires_grad_(), 8)
        self.assertFalse(torch.allclose(opened, base, rtol=1e-6, atol=1e-7))
        opened.square().mean().backward()
        self.assertIsNotNone(model_on.acf_local_agreement_scale.grad)
        self.assertTrue(torch.isfinite(model_on.acf_local_agreement_scale.grad).all())
        self.assertGreater(float(model_on.acf_local_agreement_scale.grad.abs()), 0.0)

    def test_local_agreement_keeps_prior_sign(self):
        args = _args(seq_len=32, lags=(4, 12), local=True)
        model = _make_model(args)
        prior = _full_prior([[0.8, -0.7], [-0.5, 0.6]], args.mamba_acf_lags)
        model.set_acf_prior(prior, lags=args.mamba_acf_lags)
        positions = torch.arange(32, dtype=torch.float32)
        signal = torch.sin(2.0 * torch.pi * positions / 4.0)
        values = signal.view(1, -1, 1).expand(1, -1, 2).clone()
        agreement, eligible, _ = model._acf_local_agreement_values(values)
        self.assertTrue(eligible.all())
        # Lag 4 has positive local correlation; multiplying by the prior sign
        # must preserve opposite signs across the two channels.
        self.assertGreater(float(agreement[0, 0, 0]), 0.0)
        self.assertLess(float(agreement[0, 1, 0]), 0.0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class MambaACFCLAMCudaTests(unittest.TestCase):
    def test_cuda_forecast_and_rezero_gradient(self):
        args = _args(seq_len=32, pred_len=8, lags=(4, 12, 24), local=True)
        model = _make_model(args).cuda().train()
        model.set_acf_prior(
            _full_prior(
                [[0.9, -0.7, 0.3], [-0.8, 0.6, 0.4]],
                args.mamba_acf_lags,
            ),
            lags=args.mamba_acf_lags,
        )
        hidden = torch.randn(2, 32, args.d_model, device="cuda", requires_grad=True)
        values = torch.randn(2, 32, args.enc_in, device="cuda", requires_grad=True)
        with torch.no_grad():
            model.acf_local_agreement_scale.fill_(0.25)
        forecast = model._forecast_normalized(hidden, values)
        self.assertEqual(tuple(forecast.shape), (2, 8, 2))
        self.assertTrue(torch.isfinite(forecast).all())
        forecast.square().mean().backward()
        gradient = model.acf_local_agreement_scale.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue(torch.isfinite(values.grad).all())


if __name__ == "__main__":
    unittest.main()
