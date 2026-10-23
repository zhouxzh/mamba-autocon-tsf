"""Focused tests for the leakage-safe ACF guidance utilities."""

import unittest

import numpy as np

from utils.acf_guidance import (
    build_acf_relation_targets,
    causal_lag_features,
    compile_channel_lag_support,
    fit_signed_acf_prior,
)


class ACFGuidanceTests(unittest.TestCase):
    def test_channel_lag_support_canonicalizes_lags_and_uses_stable_ties(self):
        # Columns are deliberately supplied as lag 5, 1, 3.  After canonical
        # reordering, both channels tie between lags 1 and 3, so the lower
        # lag must win without depending on a backend-specific top-k order.
        values = np.asarray(
            [[0.2, 0.9, 0.9], [-0.2, -0.9, 0.9]], dtype=np.float32
        )
        support, fractions, lags = compile_channel_lag_support(
            values, [5, 1, 3], seq_len=8, horizon=1, top_k=1
        )

        self.assertEqual(lags, (1, 3, 5))
        np.testing.assert_allclose(fractions, [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(
            support,
            np.asarray([[1, 0, 0], [1, 0, 0]], dtype=np.uint8),
        )

    def test_channel_lag_support_rejects_insufficient_legal_slots(self):
        with self.assertRaisesRegex(ValueError, "legal forecast support"):
            compile_channel_lag_support(
                np.asarray([[0.8, 0.7]], dtype=np.float32),
                [20, 30],
                seq_len=8,
                horizon=3,
                top_k=1,
            )

    def test_acf_is_invariant_to_a_constant_level(self):
        steps = np.arange(120, dtype=np.float64)
        base = np.sin(2.0 * np.pi * steps / 12.0)
        shifted = np.stack([base, base + 1000.0], axis=1)
        prior = fit_signed_acf_prior(
            shifted, max_lag=24, top_k=2, confidence_z=None, detrend="none"
        )
        np.testing.assert_allclose(prior.acf[0], prior.acf[1], atol=1e-10)

    def test_chunked_acf_matches_single_chunk(self):
        rng = np.random.default_rng(4)
        values = rng.normal(size=(80, 5))
        chunked = fit_signed_acf_prior(
            values, max_lag=20, top_k=4, confidence_z=None, channel_chunk_size=2
        )
        whole = fit_signed_acf_prior(
            values, max_lag=20, top_k=4, confidence_z=None, channel_chunk_size=None
        )
        np.testing.assert_allclose(chunked.acf, whole.acf, atol=1e-12)
        self.assertEqual(chunked.peaks, whole.peaks)

    def test_fit_keeps_channel_specific_signed_peaks(self):
        steps = np.arange(240, dtype=np.float64)
        # Channel zero repeats in phase every 12 points.  Channel one is
        # anti-phase at lag 6, so dropping the sign would lose information.
        values = np.stack(
            [
                np.sin(2.0 * np.pi * steps / 12.0),
                np.sin(2.0 * np.pi * steps / 12.0),
            ],
            axis=1,
        )
        prior = fit_signed_acf_prior(
            values,
            max_lag=30,
            top_k=3,
            min_abs_acf=0.2,
            confidence_z=None,
            min_separation=2,
        )

        self.assertEqual(prior.acf.shape, (2, 31))
        self.assertEqual(prior.train_length, 240)
        self.assertIn(12, prior.channel_lags(0))
        self.assertIn(6, prior.channel_lags(1))
        self.assertLess(prior.acf[0, 6], 0.0)
        self.assertGreater(prior.acf[0, 12], 0.0)
        self.assertLess(prior.channel_weights(0)[0], 0.0)
        self.assertEqual(prior.common_lags(top_k=1), (6,))

    def test_availability_weighted_lags_prefer_usable_train_peaks(self):
        # Lag 12 has the stronger ACF, but with a history of eight and a
        # horizon of four it has no legal direct-memory source.  Lag 6 is
        # weaker but available for half the forecast horizon.
        prior = fit_signed_acf_prior(
            np.tile(np.asarray([[1.0], [0.0], [-1.0], [0.0]]), (12, 1)),
            max_lag=12,
            top_k=6,
            min_abs_acf=0.01,
            confidence_z=None,
            min_separation=1,
        )
        selected = prior.availability_weighted_lags(
            history_length=8,
            horizon=4,
            top_k=3,
            min_separation=1,
        )
        self.assertTrue(selected)
        self.assertNotEqual(selected[0], 12)
        self.assertLessEqual(selected[0], 8)
        self.assertTrue(all(lag <= 8 for lag in selected))

    def test_availability_weighted_lags_validate_forecast_contract(self):
        prior = fit_signed_acf_prior(
            np.arange(30, dtype=np.float64)[:, None], max_lag=8, confidence_z=None
        )
        with self.assertRaises(ValueError):
            prior.availability_weighted_lags(
                history_length=0, horizon=4, top_k=2
            )
        with self.assertRaises(ValueError):
            prior.availability_weighted_lags(
                history_length=4, horizon=0, top_k=2
            )

    def test_fit_rejects_bool_nonfinite_and_nonintegral_controls(self):
        values = np.arange(24, dtype=np.float64)[:, None]
        defaults = {
            "max_lag": 6,
            "top_k": 2,
            "min_lag": 1,
            "min_abs_acf": 0.1,
            "confidence_z": 1.96,
            "min_separation": 1,
            "channel_chunk_size": 2,
        }
        invalid = (
            ("max_lag_python_bool", {"max_lag": True}),
            ("max_lag_numpy_bool", {"max_lag": np.bool_(True)}),
            ("max_lag_fraction", {"max_lag": 1.5}),
            ("max_lag_nonfinite", {"max_lag": float("inf")}),
            ("top_k_python_bool", {"top_k": True}),
            ("top_k_numpy_bool", {"top_k": np.bool_(True)}),
            ("top_k_fraction", {"top_k": 1.5}),
            ("top_k_nonfinite", {"top_k": float("nan")}),
            ("min_lag_python_bool", {"min_lag": True}),
            ("min_lag_numpy_bool", {"min_lag": np.bool_(True)}),
            ("min_lag_fraction", {"min_lag": 1.5}),
            ("min_lag_nonfinite", {"min_lag": float("inf")}),
            ("min_separation_python_bool", {"min_separation": True}),
            ("min_separation_numpy_bool", {"min_separation": np.bool_(True)}),
            ("min_separation_fraction", {"min_separation": 1.5}),
            ("min_separation_nonfinite", {"min_separation": float("nan")}),
            ("chunk_python_bool", {"channel_chunk_size": True}),
            ("chunk_numpy_bool", {"channel_chunk_size": np.bool_(True)}),
            ("chunk_fraction", {"channel_chunk_size": 1.5}),
            ("chunk_nonfinite", {"channel_chunk_size": float("inf")}),
            ("min_abs_python_bool", {"min_abs_acf": True}),
            ("min_abs_numpy_bool", {"min_abs_acf": np.bool_(True)}),
            ("min_abs_nan", {"min_abs_acf": float("nan")}),
            ("min_abs_inf", {"min_abs_acf": float("inf")}),
            ("min_abs_negative", {"min_abs_acf": -0.01}),
            ("min_abs_above_one", {"min_abs_acf": 1.01}),
            ("confidence_python_bool", {"confidence_z": True}),
            ("confidence_numpy_bool", {"confidence_z": np.bool_(True)}),
            ("confidence_nan", {"confidence_z": float("nan")}),
            ("confidence_inf", {"confidence_z": float("inf")}),
            ("confidence_negative", {"confidence_z": -0.01}),
        )
        for name, override in invalid:
            with self.subTest(name=name), self.assertRaises(ValueError):
                fit_signed_acf_prior(values, **{**defaults, **override})

        prior = fit_signed_acf_prior(
            values,
            max_lag=6,
            top_k=0,
            min_lag=1,
            min_abs_acf=0.0,
            confidence_z=None,
            min_separation=1,
            channel_chunk_size=None,
        )
        self.assertEqual(prior.peaks, (tuple(),))

    def test_causal_forecast_features_report_unavailable_memory(self):
        history = np.arange(5, dtype=np.float64)[:, None]
        features, available = causal_lag_features(history, [1, 3], horizon=3)

        # At h=0, lag one is the final observed value and lag three is still
        # available.  Later horizons cannot read beyond the history boundary.
        np.testing.assert_allclose(features[0, 0], [4.0, 2.0])
        np.testing.assert_array_equal(available[0, 0], [True, True])
        np.testing.assert_array_equal(available[1, 0], [False, True])
        np.testing.assert_array_equal(available[2, 0], [False, True])

    def test_direct_forecast_lags_do_not_recycle_old_cycles(self):
        # A direct lag feature at L+h-lag must become unavailable once the
        # source leaves the observed history; it must not repeat an older
        # seasonal value for a later forecast step.
        history = np.arange(6, dtype=np.float64)[:, None]
        features, available = causal_lag_features(history, [2], horizon=4)
        np.testing.assert_allclose(features[:2, 0, 0], [4.0, 5.0])
        np.testing.assert_array_equal(available[:, 0, 0], [True, True, False, False])

    def test_per_channel_lags_and_weights(self):
        history = np.arange(10, dtype=np.float64).reshape(5, 2)
        lags = np.asarray([[1, 3], [2, -1]])
        weights = np.asarray([[2.0, 1.0], [-1.0, 0.0]])
        features, available = causal_lag_features(
            history, lags, horizon=1, weights=weights, fill_value=-9.0
        )

        # Channel 0: x[4] * 2 and x[2]. Channel 1: x[3] * -1; -1 is a
        # deliberate unused slot and is reported unavailable.
        np.testing.assert_allclose(features, [[[16.0, 4.0], [-7.0, -9.0]]])
        np.testing.assert_array_equal(available, [[[True, True], [True, False]]])

    def test_lag_features_reject_python_and_numpy_bool_lags(self):
        history = np.arange(6, dtype=np.float64)[:, None]
        invalid_lags = (
            True,
            np.bool_(True),
            [True],
            [np.bool_(True)],
            [True, 2],
            np.asarray([True]),
            np.asarray([[True]]),
        )
        for lags in invalid_lags:
            with self.subTest(lags=repr(lags)), self.assertRaises(ValueError):
                causal_lag_features(history, lags, horizon=1)

    def test_relation_targets_do_not_clip_unmeasured_pairs_by_default(self):
        acf = np.asarray([[1.0, 0.7, -0.4], [1.0, 0.2, 0.1]])
        relations = build_acf_relation_targets([0, 1, 5], acf)

        self.assertEqual(relations.shape, (3, 3, 2))
        np.testing.assert_allclose(relations[0, 1], [0.7, 0.2])
        np.testing.assert_allclose(relations[0, 2], [0.0, 0.0])
        np.testing.assert_allclose(
            build_acf_relation_targets([0, 5], acf, out_of_range="clip")[0, 1],
            [-0.4, 0.1],
        )

    def test_relation_targets_reject_python_and_numpy_bool_origins(self):
        acf = np.asarray([1.0, 0.5, 0.25])
        invalid_origins = (
            True,
            np.bool_(True),
            [True],
            [np.bool_(True)],
            [True, 1],
            np.asarray([True]),
            np.asarray([np.bool_(True), np.bool_(False)]),
        )
        for origins in invalid_origins:
            with self.subTest(origins=repr(origins)), self.assertRaises(ValueError):
                build_acf_relation_targets(origins, acf)

    def test_torch_and_numpy_lag_features_agree_when_available(self):
        try:
            import torch
        except ImportError:  # pragma: no cover - repository test env has torch.
            self.skipTest("PyTorch is unavailable")

        history = np.arange(12, dtype=np.float32).reshape(6, 2)
        numpy_features, numpy_mask = causal_lag_features(history, [1, 4], horizon=2)
        torch_features, torch_mask = causal_lag_features(
            torch.as_tensor(history), [1, 4], horizon=2
        )
        np.testing.assert_allclose(torch_features.numpy(), numpy_features)
        np.testing.assert_array_equal(torch_mask.numpy(), numpy_mask)

        prior = fit_signed_acf_prior(torch.as_tensor(history), max_lag=3, confidence_z=None)
        self.assertEqual(prior.n_channels, 2)


if __name__ == "__main__":
    unittest.main()
