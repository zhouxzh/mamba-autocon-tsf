"""Numerical checks for the AutoCon local objective.

Run with the repository's training environment, for example:
``conda run --no-capture-output -n mamba python -m unittest -v tests.test_autocon_loss``.
"""

import numpy as np
import torch
import unittest

from layers.losses import (
    ACFRelationLoss,
    ForecastACFConsistencyLoss,
    ForecastLagPairConsistencyLoss,
    ForecastLagPairAffineLoss,
    ForecastLagPairContrastiveLoss,
    ChannelPatchLagContrastiveLoss,
    PooledPatchLagContrastiveLoss,
    PooledPatchVarianceLoss,
    AutoCon,
    AutoConCI,
    _legacy_relation_map,
    local_autocorr_mask,
)


def _reference_local_loss(features, labels, acf_values, temperature=1.0):
    scores = torch.bmm(features, features.transpose(1, 2)) / temperature
    batch, steps, _ = scores.shape
    distances = (labels.unsqueeze(1) - labels.unsqueeze(2)).abs()
    acf = torch.as_tensor(acf_values, dtype=features.dtype, device=features.device)
    in_range = distances < acf.numel()
    relations = acf[distances.long().clamp(max=acf.numel() - 1)]
    relations = relations.masked_fill(~in_range, 0.0)
    diagonal = torch.eye(steps, device=features.device, dtype=torch.bool)
    candidates = in_range & ~diagonal.unsqueeze(0)
    positive = local_autocorr_mask(relations, ~candidates)
    positive = positive * candidates.to(features.dtype)

    candidate_logits = scores.masked_fill(~candidates, -np.inf)
    has_candidate = candidates.any(dim=2, keepdim=True)
    shift = candidate_logits.max(dim=2, keepdim=True).values
    shift = torch.where(has_candidate, shift, torch.zeros_like(shift))
    logits = scores - shift
    denominator = (logits.exp() * candidates.to(features.dtype)).sum(
        2, keepdim=True
    ).clamp_min(torch.finfo(features.dtype).tiny)
    log_prob = logits - torch.log(denominator)
    count = positive.sum(2).clamp_min(1.0)
    return -(relations * positive * log_prob).sum(2) / count


class AutoConLossTests(unittest.TestCase):
    def test_local_loss_uses_candidate_axis_stabilization(self):
        torch.manual_seed(17)
        features = torch.randn(3, 6, 5)
        labels = torch.stack([torch.randperm(6) for _ in range(3)])
        acf_values = np.linspace(1.0, 0.05, 32, dtype=np.float32)
        objective = AutoCon(3, 6, acf_values, temperature=1.0, base_temperature=1.0)

        actual = objective.local_contrastive_loss(features, labels)
        expected = _reference_local_loss(features, labels, acf_values)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_edge_cases_are_finite(self):
        objective = AutoCon(4, 1, np.ones(4), temperature=1.0, base_temperature=1.0)
        one_step = torch.randn(2, 1, 3)
        labels = torch.zeros(2, dtype=torch.long)
        local, global_ = objective(one_step, labels)
        self.assertEqual(tuple(local.shape), (2, 1))
        self.assertEqual(tuple(global_.shape), (2,))
        self.assertTrue(torch.isfinite(local).all() and torch.isfinite(global_).all())

    def test_local_loss_has_finite_gradients(self):
        torch.manual_seed(23)
        features = torch.randn(4, 8, 3, requires_grad=True)
        labels = torch.stack([torch.randperm(8) for _ in range(4)])
        objective = AutoCon(4, 8, np.linspace(1.0, 0.1, 16), temperature=1.0, base_temperature=1.0)
        objective.local_contrastive_loss(features, labels).mean().backward()
        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_non_diagonal_perfect_acf_is_a_valid_positive(self):
        """An ACF value of one must not erase a distinct timestamp pair."""

        acf_values = np.array([1.0, 1.0, 0.2], dtype=np.float32)
        labels = torch.tensor([[0, 1, 2]])
        distances = (labels.unsqueeze(1) - labels.unsqueeze(2)).abs()
        diagonal = torch.eye(3, dtype=torch.bool).unsqueeze(0)
        relations, candidates = _legacy_relation_map(
            torch.from_numpy(acf_values), distances, diagonal
        )
        positives = local_autocorr_mask(relations, ~candidates)
        positives = positives * candidates

        self.assertFalse(bool(candidates[0, 0, 0]))
        self.assertTrue(bool(candidates[0, 0, 1]))
        self.assertEqual(float(positives[0, 0, 1]), 1.0)

        features = torch.tensor(
            [[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]]
        )
        objective = AutoCon(1, 3, acf_values, temperature=1.0, base_temperature=1.0)
        actual = objective.local_contrastive_loss(features, labels)
        expected = _reference_local_loss(features, labels, acf_values)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_out_of_range_distances_contribute_no_legacy_loss(self):
        """Unknown lags are excluded instead of reusing the final ACF value."""

        torch.manual_seed(29)
        features = torch.randn(3, 3, 4, requires_grad=True)
        objective = AutoCon(
            3, 3, np.array([1.0, 0.8], dtype=np.float32),
            temperature=1.0, base_temperature=1.0,
        )
        local_labels = torch.tensor([[0, 10, 20]]).repeat(3, 1)
        global_labels = torch.tensor([0, 10, 20])
        local = objective.local_contrastive_loss(features, local_labels)
        global_ = objective.avg_global_contrastive_loss(features, global_labels)

        self.assertTrue(torch.equal(local, torch.zeros_like(local)))
        self.assertTrue(torch.equal(global_, torch.zeros_like(global_)))
        (local.mean() + global_.mean()).backward()
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_autocon_ci_masks_out_of_range_distances(self):
        """The channel-independent legacy branch follows the same mask rule."""

        torch.manual_seed(41)
        features = torch.randn(4, 3, 2, requires_grad=True)
        objective = AutoConCI(
            2, 3, np.array([[1.0, 0.8], [1.0, 0.8]], dtype=np.float32),
            temperature=1.0, base_temperature=1.0,
        )
        local_labels = torch.tensor([[0, 10, 20]]).repeat(4, 1)
        local = objective.local_contrastive_loss(features, local_labels)
        global_ = objective.avg_global_contrastive_loss(
            features, torch.tensor([0, 10])
        )

        self.assertTrue(torch.equal(local, torch.zeros_like(local)))
        self.assertTrue(torch.equal(global_, torch.zeros_like(global_)))
        (local.mean() + global_.mean()).backward()
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_autocon_ci_global_keeps_channelwise_relations(self):
        """CI global loss retains one relation/logit matrix per channel."""

        torch.manual_seed(47)
        features = torch.randn(6, 3, 5)
        labels = torch.tensor([0, 1, 2])
        channel_acf = np.array(
            [[1.0, 0.95, 0.10], [1.0, 0.05, 0.90]], dtype=np.float32
        )
        objective = AutoConCI(
            3, 3, channel_acf, temperature=0.7, base_temperature=0.7
        )
        channelwise = objective.avg_global_contrastive_loss(features, labels)
        self.assertEqual(tuple(channelwise.shape), (2, 3))

        # A shared mean ACF would erase the deliberately opposite channel
        # relations.  The restored CI contract must therefore produce a
        # different result from that collapsed table.
        collapsed = AutoConCI(
            3, 3, channel_acf.mean(axis=0), temperature=0.7, base_temperature=0.7
        ).avg_global_contrastive_loss(features, labels)
        self.assertGreater(float((channelwise - collapsed).abs().max()), 1e-6)


class ACFRelationLossTests(unittest.TestCase):
    def _objective(self):
        acf = torch.zeros(2, 17)
        acf[:, 0] = 1.0
        acf[:, 4] = 0.9
        acf[:, 8] = -0.7
        return ACFRelationLoss(acf, temperature=0.3, relation_temperature=0.4)

    def test_forecast_acf_consistency_is_causal_and_differentiable(self):
        acf = torch.zeros(2, 13)
        acf[:, 0] = 1.0
        acf[:, 4] = 0.9
        objective = ForecastACFConsistencyLoss(
            acf, horizon=8, top_k=1, min_lag=2, min_abs=0.5
        )
        forecast = torch.randn(3, 8, 2, requires_grad=True)
        loss = objective(forecast)
        self.assertEqual(tuple(loss.shape), (3,))
        self.assertTrue(torch.isfinite(loss).all())
        loss.mean().backward()
        self.assertGreater(float(forecast.grad.abs().sum()), 0.0)
        self.assertEqual(objective.selected_lags.tolist(), [4])

    def test_forecast_acf_consistency_returns_zero_without_supported_prior(self):
        acf = torch.zeros(2, 13)
        objective = ForecastACFConsistencyLoss(acf, horizon=8, min_abs=0.5)
        result = objective(torch.randn(2, 8, 2))
        self.assertTrue(torch.equal(result, torch.zeros_like(result)))

    def test_forecast_lag_pair_consistency_is_prediction_connected(self):
        acf = torch.zeros(1, 8)
        acf[:, 2] = 1.0
        objective = ForecastLagPairConsistencyLoss(acf, selected_lags=[2])
        aligned = torch.tensor(
            [[[1.0], [2.0], [3.0]], [[1.0], [2.0], [3.0]]],
            requires_grad=True,
        )
        opposed = torch.tensor(
            [[[1.0], [2.0], [3.0]], [[3.0], [1.0], [1.0]]],
            requires_grad=True,
        )
        aligned_loss = objective(aligned, torch.tensor([0, 2]))
        opposed_loss = objective(opposed, torch.tensor([0, 2]))
        self.assertLess(float(aligned_loss.mean().detach()), 1e-6)
        self.assertGreater(float(opposed_loss.mean().detach()), 3.0)
        opposed_loss.mean().backward()
        self.assertGreater(float(opposed.grad.abs().sum()), 0.0)
        self.assertEqual(objective.last_pair_count, 1.0)
        self.assertEqual(objective.last_active_anchor_fraction, 1.0)

    def test_forecast_lag_pair_consistency_is_permutation_invariant(self):
        torch.manual_seed(59)
        acf = torch.zeros(2, 12)
        acf[:, 4] = torch.tensor([0.7, -0.5])
        objective = ForecastLagPairConsistencyLoss(acf, selected_lags=[4])
        forecast = torch.randn(4, 6, 2)
        origins = torch.tensor([0, 4, 16, 20])
        expected = objective(forecast, origins).mean()
        permutation = torch.tensor([2, 0, 3, 1])
        actual = objective(forecast[permutation], origins[permutation]).mean()
        self.assertTrue(torch.allclose(expected, actual, atol=1e-6, rtol=1e-6))

    def test_forecast_lag_pair_consistency_is_zero_without_pair_support(self):
        acf = torch.zeros(1, 8)
        acf[:, 3] = 0.9
        objective = ForecastLagPairConsistencyLoss(acf, selected_lags=[3])
        forecast = torch.randn(3, 5, 1, requires_grad=True)
        loss = objective(forecast, torch.tensor([0, 1, 8]))
        self.assertTrue(torch.equal(loss, torch.zeros_like(loss)))
        loss.mean().backward()
        self.assertTrue(torch.isfinite(forecast.grad).all())
        self.assertEqual(float(forecast.grad.abs().sum()), 0.0)

    def test_forecast_lag_pair_affine_loss_is_signed_and_prediction_connected(self):
        acf = torch.zeros(1, 4); acf[:, 1] = -0.8
        objective = ForecastLagPairAffineLoss(acf, selected_lags=[1])
        forecast = torch.tensor([[[1.0], [2.0], [3.0]], [[-1.0], [-2.0], [-3.0]], [[3.0], [2.0], [1.0]]], requires_grad=True)
        loss = objective(forecast, torch.tensor([0, 1, 7]))
        self.assertTrue(torch.isfinite(loss).all())
        self.assertGreater(float(loss.mean().detach()), 0.0)
        loss.mean().backward()
        self.assertGreater(float(forecast.grad.abs().sum()), 0.0)

    def test_forecast_lag_pair_contrastive_prefers_acf_positive_over_negative(self):
        acf = torch.zeros(1, 8)
        acf[:, 2] = 0.9
        objective = ForecastLagPairContrastiveLoss(
            acf, selected_lags=[2], channel_selected_lags=[[2, 0]], temperature=0.2
        )
        aligned = torch.tensor(
            [[[1.0], [2.0], [3.0]], [[1.0], [2.0], [3.0]], [[3.0], [2.0], [1.0]]],
            requires_grad=True,
        )
        opposed = torch.tensor(
            [[[1.0], [2.0], [3.0]], [[3.0], [2.0], [1.0]], [[1.0], [2.0], [3.0]]],
            requires_grad=True,
        )
        aligned_loss = objective(aligned, torch.tensor([0, 2, 7])).mean()
        opposed_loss = objective(opposed, torch.tensor([0, 2, 7])).mean()
        self.assertLess(float(aligned_loss.detach()), float(opposed_loss.detach()))
        opposed_loss.backward()
        self.assertGreater(float(opposed.grad.abs().sum()), 0.0)
        self.assertGreater(objective.last_active_anchor_fraction, 0.0)

    def test_forecast_lag_pair_contrastive_confidence_weighting_preserves_scale(self):
        acf = np.zeros((1, 8), dtype=np.float32)
        acf[0, 2] = 0.2
        acf[0, 4] = 0.9
        forecast = torch.randn(4, 6, 1)
        origins = torch.tensor([0, 2, 4, 6])
        plain = ForecastLagPairContrastiveLoss(
            acf, selected_lags=[2, 4], channel_selected_lags=[[2, 4]], temperature=0.2,
        )
        weighted = ForecastLagPairContrastiveLoss(
            acf, selected_lags=[2, 4], channel_selected_lags=[[2, 4]], temperature=0.2,
            acf_confidence_weighting=True,
        )
        plain_loss = plain(forecast, origins)
        weighted_loss = weighted(forecast, origins)
        self.assertTrue(torch.isfinite(weighted_loss).all())
        self.assertEqual(weighted.acf_confidence_weighting, True)
        self.assertEqual(tuple(plain_loss.shape), tuple(weighted_loss.shape))
        self.assertGreater(float((plain_loss - weighted_loss).abs().max()), 0.0)

    def test_forecast_lag_pair_contrastive_accepts_stability_weights(self):
        acf = np.zeros((1, 8), dtype=np.float32)
        acf[0, 2], acf[0, 4] = .7, .5
        forecast = torch.randn(4, 6, 1)
        origins = torch.tensor([0, 2, 4, 6])
        low_lag = ForecastLagPairContrastiveLoss(
            acf, selected_lags=[2, 4], channel_selected_lags=[[2, 4]],
            channel_positive_weights=[[1., 0.]], temperature=.2,
        )
        high_lag = ForecastLagPairContrastiveLoss(
            acf, selected_lags=[2, 4], channel_selected_lags=[[2, 4]],
            channel_positive_weights=[[0., 1.]], temperature=.2,
        )
        self.assertTrue(torch.isfinite(low_lag(forecast, origins)).all())
        self.assertTrue(torch.isfinite(high_lag(forecast, origins)).all())
        with self.assertRaisesRegex(ValueError, "match channel_selected_lags"):
            ForecastLagPairContrastiveLoss(
                acf, selected_lags=[2], channel_selected_lags=[[2]],
                channel_positive_weights=[[1., 1.]],
            )

    def test_channel_patch_lag_contrastive_accepts_stability_weights(self):
        acf = np.zeros((2, 8), dtype=np.float32)
        acf[0, 2], acf[0, 4] = .7, .5
        acf[1, 2], acf[1, 4] = .6, .8
        representation = torch.randn(4, 2, 3, 5)
        origins = torch.tensor([0, 2, 4, 6])
        low_lag = ChannelPatchLagContrastiveLoss(
            acf, selected_lags=[2, 4], channel_selected_lags=[[2, 4], [2, 4]],
            channel_positive_weights=[[1., 0.], [1., 0.]], temperature=.2,
        )
        high_lag = ChannelPatchLagContrastiveLoss(
            acf, selected_lags=[2, 4], channel_selected_lags=[[2, 4], [2, 4]],
            channel_positive_weights=[[0., 1.], [0., 1.]], temperature=.2,
        )
        self.assertTrue(torch.isfinite(low_lag(representation, origins)).all())
        self.assertTrue(torch.isfinite(high_lag(representation, origins)).all())
        self.assertGreater(low_lag.last_active_anchor_fraction, 0.0)

    def test_channel_patch_positionwise_similarity_is_finite_and_differentiable(self):
        torch.manual_seed(83)
        acf = np.zeros((1, 8), dtype=np.float32); acf[0, 2] = .9
        representation = torch.randn(4, 1, 3, 5, requires_grad=True)
        origins = torch.tensor([0, 2, 4, 6])
        trajectory = ChannelPatchLagContrastiveLoss(
            acf, selected_lags=[2], channel_selected_lags=[[2]], temperature=.2,
        )
        positionwise = ChannelPatchLagContrastiveLoss(
            acf, selected_lags=[2], channel_selected_lags=[[2]], temperature=.2,
            patch_similarity="positionwise",
        )
        last_patch = ChannelPatchLagContrastiveLoss(
            acf, selected_lags=[2], channel_selected_lags=[[2]], temperature=.2,
            patch_similarity="last_patch",
        )
        first = trajectory(representation, origins)
        second = positionwise(representation, origins)
        third = last_patch(representation, origins)
        self.assertTrue(torch.isfinite(second).all())
        self.assertTrue(torch.isfinite(third).all())
        self.assertGreater(float((first - second).abs().max().detach()), 1e-6)
        (second + third).mean().backward()
        self.assertGreater(float(representation.grad.abs().sum()), 0.0)
        with self.assertRaisesRegex(ValueError, "patch_similarity"):
            ChannelPatchLagContrastiveLoss(acf, selected_lags=[2], patch_similarity="invalid")

    def test_forecast_lag_pair_contrastive_supports_negative_acf_pairs(self):
        acf = torch.zeros(1, 8)
        acf[:, 2] = -0.9
        objective = ForecastLagPairContrastiveLoss(acf, selected_lags=[2], temperature=0.2)
        forecast = torch.tensor(
            [[[1.0], [2.0], [3.0]], [[3.0], [2.0], [1.0]], [[1.0], [3.0], [1.0]]],
            requires_grad=True,
        )
        loss = objective(forecast, torch.tensor([0, 2, 7]))
        self.assertTrue(torch.isfinite(loss).all())
        loss.mean().backward()
        self.assertGreater(float(forecast.grad.abs().sum()), 0.0)

    def test_forecast_lag_pair_contrastive_is_zero_without_negative_candidate(self):
        acf = torch.zeros(1, 8)
        acf[:, 2] = 0.9
        objective = ForecastLagPairContrastiveLoss(acf, selected_lags=[2])
        forecast = torch.randn(2, 4, 1, requires_grad=True)
        loss = objective(forecast, torch.tensor([0, 2]))
        self.assertTrue(torch.equal(loss, torch.zeros_like(loss)))
        loss.mean().backward()
        self.assertEqual(float(forecast.grad.abs().sum()), 0.0)

    def test_channel_patch_lag_contrastive_prefers_channelwise_acf_positive(self):
        acf = torch.zeros(1, 8)
        acf[:, 2] = 0.9
        objective = ChannelPatchLagContrastiveLoss(
            acf, selected_lags=[2], channel_selected_lags=torch.tensor([[2]]),
            temperature=0.2,
        )
        aligned = torch.tensor(
            [[[[1.0, 0.0], [0.0, 1.0]]], [[[1.0, 0.0], [0.0, 1.0]]], [[[0.0, 1.0], [1.0, 0.0]]]],
            requires_grad=True,
        )
        opposed = torch.tensor(
            [[[[1.0, 0.0], [0.0, 1.0]]], [[[0.0, 1.0], [1.0, 0.0]]], [[[1.0, 0.0], [0.0, 1.0]]]],
            requires_grad=True,
        )
        origins = torch.tensor([0, 2, 7])
        self.assertLess(
            float(objective(aligned, origins).mean().detach()),
            float(objective(opposed, origins).mean().detach()),
        )
        objective(opposed, origins).mean().backward()
        self.assertGreater(float(opposed.grad.abs().sum()), 0.0)

    def test_pooled_patch_lag_contrastive_prefers_acf_positive_representation(self):
        acf = torch.zeros(8); acf[2] = 0.9
        objective = PooledPatchLagContrastiveLoss(acf, selected_lags=[2], temperature=0.2)
        aligned = torch.tensor(
            [[[1.0, 0.0], [2.0, 0.0]], [[1.0, 0.0], [2.0, 0.0]], [[2.0, 0.0], [1.0, 0.0]]],
            requires_grad=True,
        )
        opposed = torch.tensor(
            [[[1.0, 0.0], [2.0, 0.0]], [[2.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [2.0, 0.0]]],
            requires_grad=True,
        )
        self.assertLess(float(objective(aligned, torch.tensor([0, 2, 7])).mean().detach()),
                        float(objective(opposed, torch.tensor([0, 2, 7])).mean().detach()))
        objective(opposed, torch.tensor([0, 2, 7])).mean().backward()
        self.assertGreater(float(opposed.grad.abs().sum()), 0.0)

    def test_pooled_patch_variance_penalizes_collapsed_representations(self):
        objective = PooledPatchVarianceLoss(target_std=0.2)
        collapsed = torch.ones(4, 3, 5, requires_grad=True)
        diverse = torch.randn(4, 3, 5, requires_grad=True)
        collapsed_loss, diverse_loss = objective(collapsed), objective(diverse)
        self.assertGreater(float(collapsed_loss.detach()), float(diverse_loss.detach()))
        collapsed_loss.backward()
        self.assertTrue(torch.isfinite(collapsed.grad).all())

    def test_direct_constructor_matches_cli_boost_default(self):
        objective = ACFRelationLoss(torch.ones(1, 8), target_mode="soft_multi_peak")
        self.assertEqual(objective.selected_lag_boost, 0.0)

    def test_constructor_rejects_nonfinite_relation_hyperparameters(self):
        for keyword, value in (
            ("temperature", float("nan")),
            ("relation_temperature", float("inf")),
            ("sign_weight", float("nan")),
            ("sign_margin", float("inf")),
        ):
            with self.subTest(keyword=keyword):
                with self.assertRaises(ValueError):
                    ACFRelationLoss(torch.ones(1, 8), **{keyword: value})

    def test_constructor_rejects_truncated_relation_lag_geometry(self):
        with self.assertRaises(ValueError):
            ACFRelationLoss(torch.ones(1, 8), lag_tolerance=1.5)
        with self.assertRaises(ValueError):
            ACFRelationLoss(torch.ones(1, 8), min_lag=np.bool_(True))
        with self.assertRaises(ValueError):
            ACFRelationLoss(torch.ones(1, 8), selected_lags=[1.5])
        with self.assertRaises(ValueError):
            ACFRelationLoss(torch.ones(1, 8), selected_lags=[1, 1])

    def test_all_pair_relation_loss_is_permutation_invariant(self):
        torch.manual_seed(31)
        features = torch.randn(4, 8, 5)
        origins = torch.tensor([0, 4, 8, 31])
        objective = self._objective()
        local, global_ = objective(features, origins)
        permutation = torch.tensor([2, 0, 3, 1])
        local_permuted, global_permuted = objective(
            features[permutation], origins[permutation]
        )
        self.assertTrue(torch.allclose(local.mean(), local_permuted.mean(), atol=1e-6))
        self.assertTrue(torch.allclose(global_.mean(), global_permuted.mean(), atol=1e-6))

    def test_out_of_range_pairs_are_masked_and_gradients_are_finite(self):
        torch.manual_seed(37)
        features = torch.randn(3, 6, 4, requires_grad=True)
        origins = torch.tensor([0, 50, 100])
        objective = self._objective()
        local, global_ = objective(features, origins)
        loss = local.mean() + global_.mean()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_non_diagonal_perfect_relation_is_not_treated_as_identity(self):
        # ACF=1 at a non-zero lag must remain a valid relation.  The identity
        # mask is defined by matrix position, not by a relation value.
        acf = torch.tensor([[1.0, 0.2, 1.0, -0.5]])
        # Use a strict margin so the test does not depend on a random cosine
        # being below the default hinge threshold when only one candidate is
        # available for an anchor.
        objective = ACFRelationLoss(acf, sign_weight=1.0, sign_margin=1.0)
        features = torch.randn(3, 4, 3)
        _, global_ = objective(features, torch.tensor([0, 2, 9]))
        self.assertTrue(torch.isfinite(global_).all())
        self.assertGreater(float(objective.last_global_valid_fraction), 0.0)

    def test_global_active_anchor_normalization_is_explicit_opt_in(self):
        """Sparse global positives must not silently change legacy scaling."""

        torch.manual_seed(53)
        features = torch.randn(4, 3, 5)
        origins = torch.tensor([0, 4, 8, 30])
        acf = torch.ones(1, 40)
        common = dict(
            temperature=0.7,
            relation_temperature=0.6,
            selected_lags=[4],
            relation_scope="global",
        )
        legacy_default = ACFRelationLoss(acf, **common)
        legacy_explicit = ACFRelationLoss(
            acf, normalize_global_active_anchors=False, **common
        )
        normalized = ACFRelationLoss(
            acf, normalize_global_active_anchors=True, **common
        )

        default_loss = legacy_default.global_contrastive_loss(features, origins)
        explicit_loss = legacy_explicit.global_contrastive_loss(features, origins)
        normalized_loss = normalized.global_contrastive_loss(features, origins)
        active = default_loss.ne(0)

        self.assertTrue(torch.equal(default_loss, explicit_loss))
        self.assertEqual(int(active.sum()), 3)
        self.assertEqual(float(normalized_loss[-1]), 0.0)
        self.assertTrue(torch.allclose(
            normalized_loss[active], default_loss[active] * (4.0 / 3.0),
            atol=1e-6,
        ))
        self.assertTrue(torch.allclose(
            normalized_loss.mean(), default_loss[active].mean(), atol=1e-6
        ))

    def test_channel_reduction_is_explicit_and_preserves_strong_phase(self):
        # Opposite channel phases should not silently become an uninformative
        # zero when the strongest-channel option is requested.
        acf = torch.tensor([[1.0, 0.2, 0.1], [1.0, -0.9, 0.1]])
        mean_objective = ACFRelationLoss(acf, channel_reduce="mean")
        max_objective = ACFRelationLoss(acf, channel_reduce="max_abs")
        distances = torch.tensor([[1]])
        mean_relation, _ = mean_objective._lookup(distances)
        max_relation, _ = max_objective._lookup(distances)
        self.assertAlmostEqual(float(mean_relation.item()), -0.35, places=6)
        self.assertAlmostEqual(float(max_relation.item()), -0.9, places=6)
        self.assertEqual(mean_objective.channel_reduce, "mean")
        self.assertEqual(max_objective.channel_reduce, "max_abs")

    def test_invalid_channel_reduction_is_rejected(self):
        with self.assertRaises(ValueError):
            ACFRelationLoss(torch.ones(2, 4), channel_reduce="per_channel")

    def test_global_coverage_diagnostics_exclude_identity_pairs(self):
        # Only the 0<->1 origin pair is supported by the lag-zero-to-two ACF;
        # identities do not count as candidates or active anchors.
        objective = ACFRelationLoss(torch.ones(1, 3))
        features = torch.randn(3, 4, 3)
        objective(features, torch.tensor([0, 1, 10]))
        self.assertAlmostEqual(
            float(objective.last_global_valid_fraction), 1.0 / 3.0, places=6
        )
        self.assertAlmostEqual(
            float(objective.last_global_active_anchor_fraction), 2.0 / 3.0, places=6
        )

    def test_high_acf_conditional_exclusion_uses_high_acf_denominator(self):
        # For origins [0, 1, 2, 3], lags one and two produce ten high-ACF
        # candidate pairs.  Hard selection of lag one leaves four lag-two
        # pairs as false-negative candidates.  The conditional diagnostic is
        # therefore 4/10, while the historical candidate-denominated field is
        # 4/12.
        objective = ACFRelationLoss(
            torch.tensor([[1.0, 0.9, 0.8, 0.05]]),
            relation_scope="global",
            selected_lags=[1],
            soft_min_abs=0.5,
        )
        objective(torch.randn(4, 3, 4), torch.tensor([0, 1, 2, 3]))
        self.assertAlmostEqual(
            float(objective.last_global_high_acf_candidate_count), 10.0, places=6
        )
        self.assertAlmostEqual(
            float(objective.last_global_high_acf_excluded_count), 4.0, places=6
        )
        self.assertAlmostEqual(
            float(objective.last_global_high_acf_conditional_excluded_fraction),
            0.4,
            places=6,
        )
        self.assertAlmostEqual(
            float(objective.last_global_high_acf_excluded_fraction), 1.0 / 3.0,
            places=6,
        )

    def test_high_acf_diagnostic_threshold_is_independent_of_soft_target(self):
        # The telemetry threshold must not reuse soft_min_abs.  Otherwise a
        # soft target makes the exclusion rate zero by definition, concealing
        # the remaining strong-but-unselected candidate pairs.
        objective = ACFRelationLoss(
            torch.tensor([[1.0, 0.9, 0.2]]),
            relation_scope="global",
            selected_lags=[1],
            target_mode="soft_multi_peak",
            soft_min_abs=0.8,
            diagnostic_min_abs=0.1,
        )
        objective(torch.randn(3, 3, 4), torch.tensor([0, 1, 2]))
        self.assertAlmostEqual(
            float(objective.last_global_high_acf_candidate_count), 6.0, places=6
        )
        self.assertAlmostEqual(
            float(objective.last_global_high_acf_excluded_count), 2.0, places=6
        )
        self.assertAlmostEqual(
            float(objective.last_global_high_acf_conditional_excluded_fraction),
            1.0 / 3.0,
            places=6,
        )

    def test_global_sign_diagnostics_follow_signed_relation_targets(self):
        # These three vectors have positive cosine at lag one and negative
        # cosine at lag two, matching the signed ACF table.  Repeating each
        # vector over steps exercises the public global path and its pooling.
        vectors = torch.tensor(
            [[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [-0.2, 0.98, 0.0]]
        )
        features = vectors.unsqueeze(1).repeat(1, 2, 1)
        objective = ACFRelationLoss(
            torch.tensor([[1.0, 0.9, -0.8]]),
            relation_scope="global",
            selected_lags=[1, 2],
            sign_margin=0.1,
        )
        _, global_loss = objective(features, torch.tensor([0, 1, 2]))
        self.assertTrue(torch.isfinite(global_loss).all())
        self.assertAlmostEqual(
            float(objective.last_global_sign_target_count), 6.0, places=6
        )
        self.assertAlmostEqual(
            float(objective.last_global_sign_accuracy), 1.0, places=5
        )
        self.assertAlmostEqual(
            float(objective.last_global_sign_hinge), 0.0, places=5
        )

    def test_component_telemetry_preserves_pair_loss_and_gradients(self):
        """Opt-in component returns must not perturb the historical loss path."""

        torch.manual_seed(67)
        objective = ACFRelationLoss(
            torch.tensor([[1.0, 0.8, -0.5, 0.4]]),
            selected_lags=[1, 2],
            temperature=0.3,
            relation_temperature=0.4,
            sign_weight=0.25,
        )
        similarities = torch.randn(3, 3, requires_grad=True)
        relation = torch.tensor(
            [[0.0, 0.8, -0.5], [0.8, 0.0, 0.4], [-0.5, 0.4, 0.0]]
        )
        candidate = ~torch.eye(3, dtype=torch.bool)
        valid = candidate.clone()
        weights = relation.abs()

        legacy = objective._pair_loss(
            similarities, relation, valid, candidate_mask=candidate,
            target_weights=weights,
        )
        legacy_grad = torch.autograd.grad(legacy.mean(), similarities)[0]
        component_input = similarities.detach().clone().requires_grad_(True)
        total, magnitude_ce, signed_hinge = objective._pair_loss(
            component_input,
            relation,
            valid,
            candidate_mask=candidate,
            target_weights=weights,
            return_components=True,
        )
        component_grad = torch.autograd.grad(total.mean(), component_input)[0]

        torch.testing.assert_close(total, legacy)
        torch.testing.assert_close(total, magnitude_ce + signed_hinge)
        torch.testing.assert_close(component_grad, legacy_grad)
        self.assertTrue(torch.isfinite(magnitude_ce).all())
        self.assertTrue(torch.isfinite(signed_hinge).all())

    def test_component_telemetry_reconstructs_active_scope_losses(self):
        torch.manual_seed(71)
        features = torch.randn(4, 6, 5)
        origins = torch.tensor([0, 1, 2, 3])
        for scope in ("local", "global", "both"):
            objective = ACFRelationLoss(
                torch.tensor([[1.0, 0.9, -0.6, 0.3, 0.1, 0.05]]),
                relation_scope=scope,
                selected_lags=[1, 2],
                sign_weight=0.25,
            )
            local, global_ = objective(features, origins)
            for value in (
                objective.last_local_magnitude_ce,
                objective.last_local_signed_hinge_contribution,
                objective.last_global_magnitude_ce,
                objective.last_global_signed_hinge_contribution,
            ):
                self.assertTrue(torch.isfinite(value))
            if scope in {"local", "both"}:
                torch.testing.assert_close(
                    local.mean(),
                    objective.last_local_magnitude_ce
                    + objective.last_local_signed_hinge_contribution,
                )
            else:
                self.assertEqual(float(objective.last_local_magnitude_ce), 0.0)
                self.assertEqual(
                    float(objective.last_local_signed_hinge_contribution), 0.0
                )
            if scope in {"global", "both"}:
                torch.testing.assert_close(
                    global_.mean(),
                    objective.last_global_magnitude_ce
                    + objective.last_global_signed_hinge_contribution,
                )
            else:
                self.assertEqual(float(objective.last_global_magnitude_ce), 0.0)
                self.assertEqual(
                    float(objective.last_global_signed_hinge_contribution), 0.0
                )

    def test_global_scope_disables_dense_local_component(self):
        torch.manual_seed(43)
        features = torch.randn(3, 8, 4, requires_grad=True)
        objective = ACFRelationLoss(
            torch.ones(1, 17), relation_scope="global"
        )
        local, global_ = objective(features, torch.tensor([0, 4, 8]))
        self.assertTrue(torch.equal(local, torch.zeros_like(local)))
        self.assertGreater(float(global_.sum().detach()), 0.0)
        (global_.mean()).backward()
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_channelwise_global_relation_preserves_opposite_acf_phases(self):
        objective = ACFRelationLoss(
            torch.tensor([[1.0, 0.9, 0.2], [1.0, -0.9, 0.2]]),
            relation_scope="global", selected_lags=[1],
        )
        torch.manual_seed(149)
        features = torch.randn(3, 2, 4, requires_grad=True)
        loss = objective.channelwise_global_contrastive_loss(
            features, torch.tensor([0, 1, 5])
        )
        self.assertEqual(tuple(loss.shape), (3, 2))
        self.assertTrue(torch.isfinite(loss).all())
        loss.mean().backward()
        self.assertGreater(float(features.grad.abs().sum()), 0.0)

    def test_channelwise_global_relation_uses_channel_specific_peak_support(self):
        objective = ACFRelationLoss(
            torch.tensor([[1.0, 0.8, 0.2], [1.0, 0.2, -0.8]]),
            relation_scope="global", selected_lags=[1],
            channel_selected_lags=torch.tensor([[1], [2]]),
        )
        features = torch.randn(3, 2, 4, requires_grad=True)
        loss = objective.channelwise_global_contrastive_loss(
            features, torch.tensor([0, 1, 2])
        )
        # Channel 0 has a positive at lag 1; channel 1 only at lag 2.
        self.assertGreater(float(loss[0, 0].detach()), 0.0)
        self.assertGreater(float(loss[0, 1].detach()), 0.0)
        loss.mean().backward()
        self.assertGreater(float(features.grad.abs().sum()), 0.0)

    def test_channelwise_global_active_normalization_removes_sparse_channel_dilution(self):
        common = dict(
            acf_values=torch.tensor([[1.0, 0.8, 0.2], [1.0, 0.2, -0.8]]),
            relation_scope="global", selected_lags=[1],
            channel_selected_lags=torch.tensor([[1], [2]]),
        )
        torch.manual_seed(177)
        features = torch.randn(3, 2, 4)
        raw = ACFRelationLoss(**common)
        normalized = ACFRelationLoss(normalize_global_active_anchors=True, **common)
        raw_loss = raw.channelwise_global_contrastive_loss(features, torch.tensor([0, 1, 2]))
        normalized_loss = normalized.channelwise_global_contrastive_loss(features, torch.tensor([0, 1, 2]))
        self.assertGreater(float(normalized_loss.abs().sum()), float(raw_loss.abs().sum()))

    def test_local_relation_geometry_cache_invalidates_on_prior_mutation(self):
        objective = ACFRelationLoss(
            torch.tensor([[1.0, 0.8, 0.2, -0.1]]), relation_scope="local",
            selected_lags=[1],
        )
        first = objective._cached_local_lookup(6, torch.device("cpu"))
        second = objective._cached_local_lookup(6, torch.device("cpu"))
        self.assertIs(first[0], second[0])
        expected = objective._lookup_components(
            (torch.arange(6)[:, None] - torch.arange(6)[None, :]).abs()
        )
        for cached, direct in zip(first, expected):
            self.assertTrue(torch.equal(cached, direct))

        objective.acf_values[0, 1] = 0.4
        refreshed = objective._cached_local_lookup(6, torch.device("cpu"))
        self.assertIsNot(first[0], refreshed[0])
        self.assertFalse(torch.equal(first[0], refreshed[0]))

    def test_selected_lag_mask_keeps_only_peak_candidates(self):
        objective = ACFRelationLoss(
            torch.ones(1, 12), selected_lags=[4, 8], lag_tolerance=1
        )
        relation, valid = objective._lookup(torch.arange(0, 12))
        self.assertEqual(relation.shape, (12,))
        self.assertEqual(valid.nonzero(as_tuple=False).flatten().tolist(), [3, 4, 5, 7, 8, 9])

    def test_selected_lags_define_positives_but_keep_in_range_negatives(self):
        objective = ACFRelationLoss(
            torch.ones(1, 32), relation_scope="global", selected_lags=[2]
        )
        features = torch.randn(4, 3, 5, requires_grad=True)
        _, global_loss = objective(features, torch.tensor([0, 2, 10, 20]))
        global_loss.mean().backward()
        # Every non-identity in-range pair remains a denominator candidate,
        # while only the lag-2 pairs are positive targets.
        self.assertAlmostEqual(float(objective.last_global_candidate_fraction), 1.0)
        self.assertAlmostEqual(float(objective.last_global_valid_fraction), 1.0 / 6.0)
        self.assertAlmostEqual(float(objective.last_global_positive_count_mean), 0.5)
        self.assertGreater(float(features.grad.norm()), 0.0)

    def test_global_telemetry_marks_single_candidate_rows_as_degenerate(self):
        """One positive candidate cannot supply a magnitude contrastive gradient."""

        objective = ACFRelationLoss(
            torch.tensor([[1.0, 0.8, 0.2]]),
            relation_scope="global",
            selected_lags=[1],
        )
        # Both anchor rows have one in-range candidate, and it is their only
        # positive. Equal features also make the sign hinge inactive.
        features = torch.tensor(
            [[[1.0, 0.0]], [[1.0, 0.0]]], requires_grad=True
        )
        _, global_loss = objective(features, torch.tensor([0, 1]))
        self.assertTrue(torch.equal(global_loss, torch.zeros_like(global_loss)))
        self.assertAlmostEqual(
            float(objective.last_global_candidate_count_mean), 1.0, places=6
        )
        self.assertAlmostEqual(
            float(objective.last_global_contrastive_active_anchor_fraction), 0.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(objective.last_global_degenerate_positive_anchor_fraction), 1.0,
            places=6,
        )

        # Adding a third origin supplies a lag-2 denominator negative to the
        # edge anchors, so the new diagnostics distinguish usable rows.
        features = torch.tensor(
            [[[1.0, 0.0]], [[0.8, 0.6]], [[0.0, 1.0]]],
            requires_grad=True,
        )
        _, global_loss = objective(features, torch.tensor([0, 1, 2]))
        self.assertGreater(
            float(objective.last_global_contrastive_active_anchor_fraction), 0.0
        )
        self.assertLess(
            float(objective.last_global_degenerate_positive_anchor_fraction), 1.0
        )
        global_loss.mean().backward()
        self.assertIsNotNone(features.grad)
        self.assertGreater(float(features.grad.abs().sum()), 0.0)

    def test_relation_min_lag_masks_denominator_and_target_candidates(self):
        # The lower bound is a candidate rule, not merely a filter on the
        # selected positive list.  This prevents short overlapping windows
        # from re-entering the denominator in a no-overlap control.
        objective = ACFRelationLoss(
            torch.tensor([[1.0, 0.2, 0.8, 0.1, -0.7, 0.4]]),
            relation_scope="global",
            selected_lags=[2, 4],
            min_lag=4,
        )
        relation, in_range, selected = objective._lookup_components(
            torch.arange(0, 6)
        )
        self.assertEqual(in_range.nonzero(as_tuple=False).flatten().tolist(), [4, 5])
        self.assertEqual(selected.nonzero(as_tuple=False).flatten().tolist(), [4])
        candidate = in_range.clone()
        target, _ = objective._target_specification(relation, candidate, selected)
        self.assertEqual(target.nonzero(as_tuple=False).flatten().tolist(), [4])

    def test_soft_multi_peak_keeps_unselected_strong_acf_as_target(self):
        objective = ACFRelationLoss(
            torch.tensor([[1.0, 0.2, 0.8, 0.05, -0.7]]),
            relation_scope="global",
            selected_lags=[2],
            target_mode="soft_multi_peak",
            soft_min_abs=0.5,
            selected_lag_boost=1.0,
        )
        relation, in_range, selected = objective._lookup_components(
            torch.arange(0, 5)
        )
        target, weights = objective._target_specification(relation, in_range, selected)
        # lag 2 is selected and lag 4 is an unselected but strong negative
        # correlation; both are targets, while weak lag 1 is not.
        self.assertTrue(bool(target[2]))
        self.assertTrue(bool(target[4]))
        self.assertFalse(bool(target[1]))
        self.assertAlmostEqual(float(weights[2]), 1.6, places=6)
        self.assertAlmostEqual(float(weights[4]), 0.7, places=6)

    def test_soft_targets_with_no_nonzero_acf_are_finite_and_inactive(self):
        objective = ACFRelationLoss(
            torch.zeros(1, 8), relation_scope="global",
            target_mode="soft_multi_peak", soft_min_abs=0.0,
        )
        features = torch.randn(4, 3, 5, requires_grad=True)
        _, global_loss = objective(features, torch.tensor([0, 1, 2, 3]))
        self.assertTrue(torch.isfinite(global_loss).all())
        self.assertTrue(torch.equal(global_loss, torch.zeros_like(global_loss)))
        global_loss.mean().backward()
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_soft_full_is_supported_as_nonzero_control(self):
        objective = ACFRelationLoss(
            torch.tensor([[1.0, 0.0, 0.6]]), relation_scope="global",
            target_mode="soft_full",
        )
        relation, in_range, selected = objective._lookup_components(torch.arange(0, 3))
        target, _ = objective._target_specification(relation, in_range, selected)
        self.assertTrue(bool(target[2]))
        self.assertFalse(bool(target[1]))

    def test_invalid_relation_scope_and_lag_tolerance_are_rejected(self):
        with self.assertRaises(ValueError):
            ACFRelationLoss(torch.ones(1, 4), relation_scope="dense")
        with self.assertRaises(ValueError):
            ACFRelationLoss(torch.ones(1, 4), lag_tolerance=-1)
        with self.assertRaises(ValueError):
            ACFRelationLoss(torch.ones(1, 4), diagnostic_min_abs=1.1)
        with self.assertRaisesRegex(ValueError, "min_lag=4"):
            ACFRelationLoss(torch.ones(1, 4), min_lag=4)


if __name__ == "__main__":
    unittest.main()
