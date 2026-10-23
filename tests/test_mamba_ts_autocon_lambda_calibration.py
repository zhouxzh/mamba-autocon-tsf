import unittest

from scripts.analyze_mamba_ts_autocon_lambda_calibration import analyze


class MambaTSAutoConLambdaCalibrationTests(unittest.TestCase):
    def _rows(self):
        return [
            {
                "dataset": "toy", "pred_len": 12, "seed": 1,
                "model": "MambaTSv3AutoCon", "autocon_lambda": 0.0,
                "best_val_mse": 1.0, "best_epoch": 4,
                "best_epoch_autocon_loss": "2.8", "elapsed_sec": "1.0",
            },
            {
                "dataset": "toy", "pred_len": 12, "seed": 1,
                "model": "MambaTSv3AutoCon", "autocon_lambda": 0.001,
                "best_val_mse": 0.99, "best_epoch": 4,
                "best_epoch_autocon_loss": "2.2", "elapsed_sec": "1.0",
            },
            {
                "dataset": "toy", "pred_len": 12, "seed": 1,
                "model": "MambaTSv3AutoCon", "autocon_lambda": 0.003,
                "best_val_mse": 1.01, "best_epoch": 4,
                "best_epoch_autocon_loss": "2.1", "elapsed_sec": "1.0",
            },
        ]

    def test_analyze_pairs_positive_lambdas_to_zero_control(self):
        deltas, summary = analyze(self._rows())
        self.assertEqual(len(deltas), 3)
        self.assertEqual(len(summary), 3)
        positive = {row["autocon_lambda"]: row for row in summary if row["autocon_lambda"] > 0}
        self.assertEqual(positive[0.001]["wins_vs_zero"], 1)
        self.assertEqual(positive[0.003]["wins_vs_zero"], 0)
        self.assertAlmostEqual(positive[0.001]["mean_relative_change_pct"], -1.0)

    def test_analyze_rejects_coordinate_without_zero_control(self):
        rows = self._rows()[1:]
        with self.assertRaisesRegex(ValueError, "zero-lambda control"):
            analyze(rows)


if __name__ == "__main__":
    unittest.main()
