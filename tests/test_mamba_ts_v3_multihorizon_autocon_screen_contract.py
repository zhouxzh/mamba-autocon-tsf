import unittest

from scripts import analyze_mamba_ts_v3_multihorizon_autocon_screen as analyzer


class MambaTSv3MultiHorizonAutoConScreenContractTests(unittest.TestCase):
    def test_exact_coordinates_are_paired_at_each_horizon(self):
        rows = []
        for dataset in analyzer.DATASETS:
            for horizon in analyzer.HORIZONS:
                for seed in analyzer.SEEDS:
                    for phase, loss in ((analyzer.PHASES[0], 1.0), (analyzer.PHASES[1], 1.1), (analyzer.PHASES[2], 1.09)):
                        rows.append({"phase": phase, "dataset": dataset, "horizon": horizon, "seed": seed, "best_val_mse": loss})
        pairs = analyzer._pairs(rows, analyzer.PHASES[2], analyzer.PHASES[1])
        self.assertEqual(len(pairs), 24)
        self.assertEqual({(row["dataset"], row["horizon"]) for row in pairs}, {(dataset, horizon) for dataset in analyzer.DATASETS for horizon in analyzer.HORIZONS})
        self.assertEqual({item["n_pairs"] for item in analyzer._summary(pairs)}, {3})

    def test_missing_seed_is_rejected(self):
        rows = [
            {"phase": analyzer.PHASES[1], "dataset": "ETTh1", "horizon": 336, "seed": 2021, "best_val_mse": 1.0},
            {"phase": analyzer.PHASES[2], "dataset": "ETTh1", "horizon": 336, "seed": 2021, "best_val_mse": 1.0},
        ]
        with self.assertRaises(KeyError):
            analyzer._pairs(rows, analyzer.PHASES[2], analyzer.PHASES[1])


if __name__ == "__main__":
    unittest.main()
