"""Regression tests for split boundaries and train-only ACF fitting."""

import csv
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from data_provider.data_loader import Dataset_Custom
from exp.exp_long_term_forecasting_with_mamba import _MambaExperimentMixin


class _PriorCaptureModel:
    """Small stand-in for a MambaACF model's prior attachment API."""

    def set_acf_prior(self, acf, lags):
        self.acf = np.asarray(acf).copy()
        self.lags = tuple(int(lag) for lag in lags)


class _TrainOnlyExperiment:
    """Expose the mixin method without constructing an optional Mamba kernel."""

    def __init__(self, root_path):
        self.model = _PriorCaptureModel()
        self.flags = []
        self.args = SimpleNamespace(
            root_path=str(root_path),
            data_path="series.csv",
            seq_len=4,
            label_len=2,
            pred_len=3,
            features="S",
            target="target",
            embed="fixed",
            freq="h",
            train_ratio=0.6,
            seasonal_patterns=None,
            num_workers=0,
            AutoCon_acf_source="raw",
            mamba_acf_lags=[1, 2, 3],
            mamba_acf_max_lag=5,
            mamba_acf_top_k=3,
            mamba_acf_min_lag=1,
            mamba_acf_min_abs=0.0,
            # The experiment entry point casts this option to float.
            mamba_acf_confidence_z=0.0,
            mamba_acf_min_separation=1,
            mamba_acf_detrend="none",
            mamba_acf_chunk=2,
            mamba_acf_lag_mode="fixed",
        )

    def _get_data(self, flag):
        self.flags.append(flag)
        if flag != "train":
            raise AssertionError("the prior fitting path requested a non-train split")
        dataset = Dataset_Custom(
            root_path=self.args.root_path,
            data_path=self.args.data_path,
            flag=flag,
            size=[self.args.seq_len, self.args.label_len, self.args.pred_len],
            features=self.args.features,
            target=self.args.target,
            scale=True,
            timeenc=0,
            freq=self.args.freq,
            seasonal_patterns=self.args.seasonal_patterns,
            train_ratio=self.args.train_ratio,
        )
        return dataset, None


class _CsvFixtureMixin:
    @staticmethod
    def _write_series(directory, values):
        path = Path(directory) / "series.csv"
        start = datetime(2020, 1, 1)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["date", "target"])
            for index, value in enumerate(values):
                writer.writerow(
                    [
                        (start + timedelta(hours=index)).strftime("%Y-%m-%d %H:%M:%S"),
                        float(value),
                    ]
                )
        return path


class DatasetSplitBoundaryTests(_CsvFixtureMixin, unittest.TestCase):
    """Forecast labels stay within a split while history may cross its edge."""

    def test_all_forecast_windows_respect_split_boundaries(self):
        values = np.arange(30, dtype=np.float64)
        seq_len, label_len, pred_len = 4, 2, 3
        train_ratio = 0.6

        with tempfile.TemporaryDirectory() as directory:
            self._write_series(directory, values)
            datasets = {
                flag: Dataset_Custom(
                    root_path=directory,
                    data_path="series.csv",
                    flag=flag,
                    size=[seq_len, label_len, pred_len],
                    features="S",
                    target="target",
                    scale=False,
                    timeenc=0,
                    freq="h",
                    train_ratio=train_ratio,
                )
                for flag in ("train", "val", "test")
            }

            train_end = int(len(values) * train_ratio)
            test_start = len(values) - int(len(values) * 0.2)
            bounds = {
                "train": (0, train_end, 0),
                "val": (train_end, test_start, train_end - seq_len),
                "test": (test_start, len(values), test_start - seq_len),
            }

            for flag, dataset in datasets.items():
                split_start, split_end, history_start = bounds[flag]
                self.assertGreater(len(dataset), 0)
                for index in range(len(dataset)):
                    _, seq_x, seq_y, _, _ = dataset[index]
                    global_x_start = history_start + index
                    forecast_start = global_x_start + seq_len
                    forecast_end = forecast_start + pred_len

                    np.testing.assert_array_equal(
                        np.asarray(seq_x)[:, 0],
                        np.arange(global_x_start, global_x_start + seq_len),
                    )
                    np.testing.assert_array_equal(
                        np.asarray(seq_y)[-pred_len:, 0],
                        np.arange(forecast_start, forecast_end),
                    )
                    self.assertGreaterEqual(forecast_start, split_start)
                    self.assertLessEqual(forecast_end, split_end)

                # Validation and test intentionally borrow only historical
                # context from before their split boundary.
                if flag != "train":
                    _, first_x, _, _, _ = dataset[0]
                    self.assertEqual(float(first_x[0, 0]), split_start - seq_len)
                    self.assertEqual(float(first_x[-1, 0]), split_start - 1)


class TrainOnlyPriorTests(_CsvFixtureMixin, unittest.TestCase):
    """Changing held-out rows cannot change the fitted train ACF prior."""

    def test_prior_is_invariant_to_validation_and_test_mutations(self):
        train_rows = np.asarray(
            [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0],
            dtype=np.float64,
        )
        # With 20 rows and a 0.6 train ratio, rows [0, 11] are training rows.
        base = np.concatenate([train_rows, np.asarray([2.0] * 8)])
        changed = np.concatenate([train_rows, np.asarray([2000.0, -3000.0] * 4)])

        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            self._write_series(first_dir, base)
            self._write_series(second_dir, changed)

            first = _TrainOnlyExperiment(first_dir)
            second = _TrainOnlyExperiment(second_dir)
            _MambaExperimentMixin._attach_train_acf_prior(first)
            _MambaExperimentMixin._attach_train_acf_prior(second)

            np.testing.assert_array_equal(first.model.acf, second.model.acf)
            self.assertEqual(first.model.lags, second.model.lags)
            self.assertEqual(first.flags, ["train"])
            self.assertEqual(second.flags, ["train"])


if __name__ == "__main__":
    unittest.main()
