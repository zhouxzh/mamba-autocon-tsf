import unittest

import numpy as np

from scripts.analyze_acf_forecast_utility import (
    _evaluate_split,
    _iter_window_chunks,
)


class ACFForecastUtilityTests(unittest.TestCase):
    def test_origin_stride_samples_window_origins_without_duplicates(self):
        values = np.arange(30, dtype=np.float32)[:, None]
        chunks = list(_iter_window_chunks(values, seq_len=4, horizon=3, batch_size=2, origin_stride=3))
        histories = np.concatenate([item[0][:, 0, 0] for item in chunks])
        self.assertTrue(np.array_equal(histories, np.arange(0, 24, 3, dtype=np.float32)))
        self.assertEqual(sum(item[0].shape[0] for item in chunks), 8)

    def test_evaluate_split_reports_sparse_count_and_direct_lag_availability(self):
        values = np.arange(80, dtype=np.float32).reshape(40, 2)
        rows = _evaluate_split(
            dataset="toy",
            split="test",
            values=values,
            seq_len=8,
            horizons=(4,),
            lags=(1, 8, 12),
            train_scale=np.ones(2, dtype=np.float64),
            batch_size=3,
            horizon_chunk_size=2,
            lag_batch_size=2,
            origin_stride=2,
        )
        last = next(row for row in rows if row["baseline"] == "last")
        self.assertEqual(last["n_windows"], 15)  # 29 available origins, sampled every 2
        self.assertEqual(last["evaluated_elements"], 15 * 4 * 2)
        lag_one = next(
            row for row in rows
            if row["baseline"] == "seasonal_lag"
            and row["evaluation_scope"] == "available_only"
            and row["lag"] == 1
        )
        self.assertEqual(lag_one["available_steps_per_window"], 1)
        self.assertEqual(lag_one["evaluated_elements"], 15 * 1 * 2)
        lag_long = next(
            row for row in rows
            if row["baseline"] == "seasonal_lag"
            and row["evaluation_scope"] == "available_only"
            and row["lag"] == 12
        )
        self.assertEqual(lag_long["available_steps_per_window"], 0)
        self.assertEqual(lag_long["evaluated_elements"], 0)


if __name__ == "__main__":
    unittest.main()
