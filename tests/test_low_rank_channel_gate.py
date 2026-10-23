"""CPU contract tests for the low-rank cross-variable gate."""

from __future__ import annotations

import unittest

import torch

from layers.LowRankChannelGate import LowRankChannelGate


class LowRankChannelGateTests(unittest.TestCase):
    def test_shape_and_factorized_parameterization(self):
        channels, dim, rank = 7, 11, 3
        module = LowRankChannelGate(channels, dim=dim, rank=rank)
        x = torch.randn(2, 13, channels, dim)
        y = module(x)

        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertEqual(module.rank, rank)
        self.assertTrue(
            all(tuple(parameter.shape) != (channels, channels)
                for parameter in module.parameters())
        )
        factor_parameters = sum(parameter.numel() for parameter in module.parameters())
        self.assertLess(factor_parameters, channels * channels + 2 * channels * rank)

    def test_forward_has_finite_gradients(self):
        module = LowRankChannelGate(channels=5, dim=6, rank=2)
        x = torch.randn(3, 9, 5, 6, requires_grad=True)
        loss = module(x).square().mean()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        for parameter in module.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_single_channel_is_supported(self):
        module = LowRankChannelGate(channels=1, dim=4, rank=16)
        self.assertEqual(module.rank, 1)
        x = torch.randn(2, 8, 1, 4, requires_grad=True)
        y = module(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        y.mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_zero_initialized_residual_is_identity(self):
        module = LowRankChannelGate(channels=6, dim=3, rank=2)
        x = torch.randn(2, 10, 6, 3)
        self.assertEqual(float(module.residual_scale.detach()), 0.0)
        y = module(x)
        torch.testing.assert_close(y, x, rtol=0.0, atol=1e-7)

    def test_acf_adapter_is_factorized_and_identity_at_initialization(self):
        channels, dim, rank = 7, 5, 3
        module = LowRankChannelGate(
            channels, dim=dim, rank=rank, acf_summary_dim=4
        )
        x = torch.randn(2, 11, channels, dim)
        summary = torch.randn(channels, 4)
        y = module(x, acf_summary=summary)

        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertGreater(module.acf_parameter_count, 0)
        self.assertTrue(
            all(tuple(parameter.shape) != (channels, channels)
                for parameter in module.parameters())
        )
        torch.testing.assert_close(y, x, rtol=0.0, atol=1e-7)

    def test_acf_adapter_detaches_train_split_descriptor_and_has_gradients(self):
        module = LowRankChannelGate(
            channels=5, dim=3, rank=2, acf_summary_dim=4
        )
        with torch.no_grad():
            module.residual_scale.fill_(0.4)
        x = torch.randn(3, 9, 5, 3, requires_grad=True)
        summary = torch.randn(5, 4, requires_grad=True)
        loss = module(x, acf_summary=summary).square().mean()
        loss.backward()

        self.assertIsNone(summary.grad)
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertIsNotNone(module.acf_summary_to_latent[0].weight.grad)
        self.assertTrue(torch.isfinite(module.acf_summary_to_latent[0].weight.grad).all())

    def test_missing_acf_summary_matches_plain_gate(self):
        plain = LowRankChannelGate(channels=6, dim=2, rank=3)
        conditioned = LowRankChannelGate(
            channels=6, dim=2, rank=3, acf_summary_dim=4
        )
        conditioned.summary_to_latent.load_state_dict(
            plain.summary_to_latent.state_dict()
        )
        conditioned.latent_to_affine.load_state_dict(
            plain.latent_to_affine.state_dict()
        )
        with torch.no_grad():
            conditioned.residual_scale.copy_(plain.residual_scale)
            plain.residual_scale.fill_(0.35)
            conditioned.residual_scale.copy_(plain.residual_scale)
        plain.eval()
        conditioned.eval()
        x = torch.randn(2, 8, 6, 2)
        torch.testing.assert_close(
            plain(x), conditioned(x, acf_summary=None), rtol=0.0, atol=0.0
        )

    def test_feature_dim_alias_and_shape_validation(self):
        module = LowRankChannelGate(channels=3, feature_dim=5, rank=2)
        self.assertEqual(module.dim, 5)
        with self.assertRaises(ValueError):
            module(torch.randn(2, 4, 3, 6))
        with self.assertRaises(ValueError):
            module(torch.randn(2, 4, 2, 5))

    def test_acf_adapter_is_factorized_and_reports_its_own_parameters(self):
        channels, rank, summary_width = 7, 8, 4
        module = LowRankChannelGate(
            channels=channels,
            dim=1,
            rank=rank,
            acf_summary_dim=summary_width,
        )

        # The effective rank is capped below C, and the optional adapter must
        # not quietly introduce a dense C-by-C channel matrix.
        self.assertEqual(module.rank, channels - 1)
        self.assertTrue(
            all(
                tuple(parameter.shape) != (channels, channels)
                for parameter in module.parameters()
            )
        )
        expected_acf_parameters = (
            summary_width * module.rank
            + module.rank
            + module.rank * module.rank
            + module.rank
            + 1
        )
        self.assertEqual(module.acf_parameter_count, expected_acf_parameters)


if __name__ == "__main__":
    unittest.main()
