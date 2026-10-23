"""Focused numerical tests for the high-dimensional streaming metrics path.

Run with the repository environment, for example::

    conda run --no-capture-output -n mamba python -m unittest -v tests.test_streaming_metrics
"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.metrics import (
    StreamingMetricAccumulator,
    inverse_scaled_arrays,
    metric,
    per_channel_metrics,
    write_streaming_artifacts,
)
from scripts.validate_training_phase import _validate_stream_metrics


class _Scaler:
    """Minimal StandardScaler-compatible object for deterministic unit tests."""

    def __init__(self, mean, scale):
        self.mean_ = np.asarray(mean, dtype=np.float64)
        self.scale_ = np.asarray(scale, dtype=np.float64)
        self.n_features_in_ = int(self.mean_.size)

    def inverse_transform(self, values):
        values = np.asarray(values, dtype=np.float64)
        return values * self.scale_ + self.mean_


class StreamingMetricsTests(unittest.TestCase):
    def _assert_metric_dict(self, actual, expected):
        names = ("mae", "mse", "rmse", "mape", "mspe")
        np.testing.assert_allclose(
            [actual[name] for name in names], expected, rtol=1e-11, atol=1e-12
        )

    def _arrays(self, channels=2, pred_len=3, windows=7):
        rng = np.random.default_rng(20260828)
        true = rng.normal(loc=0.2, scale=1.4, size=(windows, pred_len, channels))
        # Include an exact zero to exercise the same epsilon denominator as metric().
        true[1, 0, 0] = 0.0
        pred = true + rng.normal(loc=0.0, scale=0.15, size=true.shape)
        scaler = _Scaler(
            np.linspace(2.0, -1.0, channels),
            np.linspace(0.5, 1.5, channels),
        )
        return pred, true, scaler

    def test_accumulator_matches_batch_metrics_and_origin_subset(self):
        pred, true, scaler = self._arrays()
        accumulator = StreamingMetricAccumulator(
            scaler,
            output_channels=2,
            pred_len=3,
            origin_stride=2,
            sample_limit=3,
        )

        # Unequal batches verify that origin indexing is global, not reset per batch.
        for start, end in ((0, 1), (1, 4), (4, 7)):
            accumulator.update(pred[start:end], true[start:end])
        payload = accumulator.to_payload()

        original_pred, original_true = inverse_scaled_arrays(pred, true, scaler)
        self._assert_metric_dict(payload["scaled"], metric(pred, true))
        self._assert_metric_dict(payload["original"], metric(original_pred, original_true))

        expected_channels = np.column_stack(
            (per_channel_metrics(pred, true), per_channel_metrics(original_pred, original_true))
        )
        np.testing.assert_allclose(
            np.asarray(payload["channel_metrics"]), expected_channels, rtol=1e-11, atol=1e-12
        )

        origin_pred, origin_true = pred[::2], true[::2]
        origin_original_pred, origin_original_true = inverse_scaled_arrays(
            origin_pred, origin_true, scaler
        )
        self._assert_metric_dict(payload["origin"]["scaled"], metric(origin_pred, origin_true))
        self._assert_metric_dict(
            payload["origin"]["original"], metric(origin_original_pred, origin_original_true)
        )
        self.assertEqual(payload["window_count"], 7)
        self.assertEqual(payload["element_count"], 7 * 3 * 2)
        self.assertEqual(payload["origin_count"], 4)
        np.testing.assert_allclose(payload["sample_pred"], pred[[0, 2, 4]])
        np.testing.assert_allclose(payload["sample_true"], true[[0, 2, 4]])

    def test_single_output_uses_last_scaler_feature_for_inverse_metrics(self):
        pred, true, scaler = self._arrays(channels=1, pred_len=2, windows=4)
        # A one-channel forecast corresponds to the target column in an MS scaler.
        full_scaler = _Scaler([3.0, 11.0, -4.0], [2.0, 0.75, 0.25])
        accumulator = StreamingMetricAccumulator(
            full_scaler,
            output_channels=1,
            pred_len=2,
            origin_stride=2,
            sample_limit=0,
        )
        accumulator.update(pred[:2], true[:2])
        accumulator.update(pred[2:], true[2:])
        payload = accumulator.to_payload()

        expected_pred = pred * full_scaler.scale_[-1] + full_scaler.mean_[-1]
        expected_true = true * full_scaler.scale_[-1] + full_scaler.mean_[-1]
        self._assert_metric_dict(payload["original"], metric(expected_pred, expected_true))
        self.assertEqual(payload["sample_pred"].shape, (0, 2, 1))
        self.assertEqual(payload["sample_true"].shape, (0, 2, 1))

    def test_write_artifacts_is_self_consistent_and_bounds_samples(self):
        pred, true, scaler = self._arrays(windows=5)
        accumulator = StreamingMetricAccumulator(
            scaler,
            output_channels=2,
            pred_len=3,
            origin_stride=1,
            sample_limit=2,
        )
        accumulator.update(pred, true)

        with tempfile.TemporaryDirectory() as directory:
            result_dir = Path(directory)
            payload = write_streaming_artifacts(result_dir, accumulator)
            stream_path = result_dir / "stream_metrics.json"
            sample_path = result_dir / "stream_samples.npz"

            self.assertTrue(stream_path.is_file())
            self.assertTrue(sample_path.is_file())
            self.assertNotIn("sample_pred", payload)
            self.assertNotIn("sample_true", payload)
            on_disk = json.loads(stream_path.read_text(encoding="utf-8"))
            self.assertEqual(on_disk["schema_version"], 1)
            self.assertEqual(on_disk["aggregation"], "streaming_window_weighted")
            self.assertEqual(on_disk["sample_artifact"]["shape"], [2, 3, 2])
            digest = hashlib.sha256(sample_path.read_bytes()).hexdigest()
            self.assertEqual(on_disk["sample_artifact"]["sha256"], digest)

            metrics = np.load(result_dir / "metrics.npy", allow_pickle=False)
            original = np.load(result_dir / "original_metrics.npy", allow_pickle=False)
            channels = np.load(result_dir / "channel_metrics.npy", allow_pickle=False)
            np.testing.assert_allclose(metrics, [on_disk["scaled"][name] for name in on_disk["metric_order"]])
            np.testing.assert_allclose(original, [on_disk["original"][name] for name in on_disk["metric_order"]])
            self.assertEqual(channels.shape, (2, 6))
            with np.load(sample_path, allow_pickle=False) as samples:
                self.assertEqual(samples["pred"].shape, (2, 3, 2))
                self.assertEqual(samples["true"].shape, (2, 3, 2))

    def test_update_rejects_invalid_shapes_and_non_finite_values(self):
        _, true, scaler = self._arrays(windows=2)
        accumulator = StreamingMetricAccumulator(scaler, 2, 3)
        with self.assertRaises(ValueError):
            accumulator.update(true[:, :2], true[:, :2])
        bad = true.copy()
        bad[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            accumulator.update(bad, true)
        with self.assertRaises(ValueError):
            StreamingMetricAccumulator(scaler, 2, 3).to_payload()

    def test_validator_binds_origin_stride_and_origin_count(self):
        pred, true, scaler = self._arrays(windows=5)
        accumulator = StreamingMetricAccumulator(
            scaler, output_channels=2, pred_len=3, origin_stride=2, sample_limit=2
        )
        accumulator.update(pred, true)
        with tempfile.TemporaryDirectory() as directory:
            result_dir = Path(directory)
            write_streaming_artifacts(result_dir, accumulator)
            config = {"stream_sample_windows": 2}
            job = {
                "pred_len": 3,
                "config": {"c_out": 2, "stream_sample_windows": 2, "origin_stride": 2},
            }
            self.assertEqual(_validate_stream_metrics(result_dir, config, job), [])
            stream_path = result_dir / "stream_metrics.json"
            payload = json.loads(stream_path.read_text(encoding="utf-8"))
            payload["origin_stride"] = 3
            stream_path.write_text(json.dumps(payload), encoding="utf-8")
            errors = _validate_stream_metrics(result_dir, config, job)
            self.assertIn("stream_origin_stride_mismatch", errors)

    def test_non_streaming_config_does_not_bind_sample_limit(self):
        """A runner-disabled streaming option must keep the parser default."""
        pred, true, scaler = self._arrays(windows=3)
        accumulator = StreamingMetricAccumulator(
            scaler, output_channels=2, pred_len=3, origin_stride=1, sample_limit=0
        )
        accumulator.update(pred, true)
        with tempfile.TemporaryDirectory() as directory:
            result_dir = Path(directory)
            write_streaming_artifacts(result_dir, accumulator)
            # This mirrors the normal validation contract: stream_metrics is
            # off, so the sample limit is not an executable requirement.
            config = {
                "stream_metrics": False,
                "stream_sample_windows": 1,
                "origin_stride": 1,
            }
            job = {
                "pred_len": 3,
                "config": {
                    "c_out": 2,
                    "stream_sample_windows": 1,
                    "origin_stride": 1,
                },
            }
            self.assertEqual(_validate_stream_metrics(result_dir, config, job), [])


if __name__ == "__main__":
    unittest.main()
