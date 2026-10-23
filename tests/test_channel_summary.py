import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.summarize_channel_metrics import (
    CHANNEL_METRICS,
    _summary_stats,
    load_per_channel_csv,
    load_stream_artifacts,
    make_group_summary,
    make_seed_summary,
)


class ChannelSummaryTests(unittest.TestCase):
    def _channel_frame(self):
        rows = []
        for seed, offset in ((11, 0.0), (22, 1.0)):
            for channel in range(3):
                mse = float((channel + 1) ** 2 + offset)
                rows.append({
                    "dataset": "toy",
                    "model": "Mamba1",
                    "features": "M",
                    "seq_len": 4,
                    "pred_len": 2,
                    "seed": seed,
                    # Deliberately omit c_out/enc_in/dec_in to exercise
                    # run-level c_out inference and defaults.
                    "channel": channel,
                    "mae": float(channel + 1 + offset),
                    "mse": mse,
                    "rmse": float(np.sqrt(mse)),
                    "original_mae": float(2 * (channel + 1) + offset),
                    "original_mse": float(4 * mse),
                    "original_rmse": float(2 * np.sqrt(mse)),
                })
        return pd.DataFrame(rows)

    def test_csv_macro_proxy_and_seed_ci(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "per_channel_metrics.csv"
            self._channel_frame().to_csv(path, index=False)
            channels = load_per_channel_csv(path)
            self.assertEqual(sorted(channels["c_out"].unique()), [3])
            seeds = make_seed_summary(channels)
            self.assertEqual(len(seeds), 2)
            first = seeds.loc[seeds["seed"] == 11].iloc[0]
            self.assertAlmostEqual(first["channel_macro_mae"], 2.0)
            self.assertAlmostEqual(first["channel_macro_mse"], 14.0 / 3.0)
            self.assertAlmostEqual(first["overall_micro_mse"], 14.0 / 3.0)
            self.assertAlmostEqual(first["overall_micro_rmse"], np.sqrt(14.0 / 3.0))
            self.assertEqual(first["micro_source"], "channel_equal_weight_proxy")

            summary = make_group_summary(seeds)
            self.assertEqual(len(summary), 1)
            grouped = summary.iloc[0]
            self.assertEqual(grouped["n_seeds"], 2)
            expected = _summary_stats(seeds["channel_macro_mse"], 0.05)
            self.assertEqual(grouped["channel_macro_mse_n"], expected[0])
            self.assertAlmostEqual(grouped["channel_macro_mse_mean"], expected[1])
            self.assertAlmostEqual(grouped["channel_macro_mse_std"], expected[2])
            self.assertAlmostEqual(grouped["channel_macro_mse_ci_low"], expected[3])
            self.assertAlmostEqual(grouped["channel_macro_mse_ci_high"], expected[4])

    def test_csv_rejects_missing_core_metric_and_duplicate_channel(self):
        frame = self._channel_frame()
        frame.loc[0, "mse"] = np.nan
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            frame.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_per_channel_csv(path)

            duplicate = self._channel_frame().iloc[:3].copy()
            duplicate = pd.concat([duplicate, duplicate.iloc[[0]]], ignore_index=True)
            duplicate_path = Path(directory) / "duplicate.csv"
            duplicate.to_csv(duplicate_path, index=False)
            with self.assertRaises(ValueError):
                load_per_channel_csv(duplicate_path)

    def _write_stream(self, root: Path, *, seed=7, bad=False):
        result = root / f"run_{seed}"
        result.mkdir(parents=True)
        config = {
            "data": "toy",
            "model": "Mamba2",
            "AutoCon": True,
            "features": "M",
            "seq_len": 4,
            "pred_len": 2,
            "seed": seed,
            "enc_in": 2,
            "dec_in": 2,
            "c_out": 2,
        }
        (result / "config.json").write_text(json.dumps(config), encoding="utf-8")
        channels = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]
        payload = {
            "schema_version": 1,
            "aggregation": "streaming_window_weighted",
            "window_count": 5,
            "element_count": 20,
            "output_channels": 2,
            "pred_len": 2,
            "origin_stride": 2,
            "origin_count": 3,
            "metric_order": ["mae", "mse", "rmse", "mape", "mspe"],
            "channel_metric_order": list(CHANNEL_METRICS),
            "scaled": {"mae": 2.0, "mse": 4.0, "rmse": 2.0, "mape": 8.0, "mspe": 16.0},
            "original": {"mae": 20.0, "mse": 40.0, "rmse": 2.0, "mape": 80.0, "mspe": 160.0},
            "channel_metrics": channels,
        }
        if bad:
            payload["scaled"]["mse"] = 99.0
        stream_path = result / "stream_metrics.json"
        stream_path.write_text(json.dumps(payload), encoding="utf-8")
        return stream_path

    def test_stream_provides_exact_micro_and_normalizes_autocon_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._write_stream(root)
            channels, micro = load_stream_artifacts([root])
            self.assertEqual(channels["model"].unique().tolist(), ["Mamba2AutoCon"])
            seeds = make_seed_summary(channels, micro)
            row = seeds.iloc[0]
            self.assertEqual(row["micro_source"], "stream_metrics")
            self.assertAlmostEqual(row["overall_micro_mse"], 4.0)
            self.assertAlmostEqual(row["overall_micro_rmse"], 2.0)
            # Channel macro RMSE is intentionally distinct from global RMSE.
            self.assertAlmostEqual(row["channel_macro_rmse"], 4.0)
            self.assertEqual(row["origin_stride"], 2)
            self.assertEqual(row["window_count"], 5)
            self.assertEqual(row["origin_count"], 3)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row["stream_sha256"])

    def test_conflicting_stream_and_overall_micro_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_stream(root)
            channels, micro = load_stream_artifacts([root])
            overall = micro.copy()
            overall["micro_source"] = "overall_csv"
            overall["overall_micro_mse"] = 5.0
            with self.assertRaises(ValueError):
                make_seed_summary(channels, micro, overall)


if __name__ == "__main__":
    unittest.main()
