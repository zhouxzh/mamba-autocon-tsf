import unittest

from scripts.analyze_mamba_ts_autocon_strong_baseline import pair


class MambaTSAutoConStrongBaselineTests(unittest.TestCase):
    def test_pairs_only_exact_coordinates(self):
        candidate = [{"dataset": "pems", "pred_len": 12, "seed": 1, "model": "MambaTSv3AutoCon", "best_val_mse": 0.9}]
        reference = [{"dataset": "pems", "pred_len": 12, "seed": 1, "model": "TimesNet", "best_val_mse": 1.0}]
        rows = pair(candidate, reference, candidate_model="MambaTSv3AutoCon", reference_model="TimesNet")
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["relative_change_pct"], -10.0)

    def test_rejects_missing_reference_coordinate(self):
        candidate = [{"dataset": "pems", "pred_len": 12, "seed": 1, "model": "MambaTSv3AutoCon", "best_val_mse": 0.9}]
        with self.assertRaisesRegex(ValueError, "exact coordinate mismatch"):
            pair(candidate, [], candidate_model="MambaTSv3AutoCon", reference_model="TimesNet")


if __name__ == "__main__":
    unittest.main()
