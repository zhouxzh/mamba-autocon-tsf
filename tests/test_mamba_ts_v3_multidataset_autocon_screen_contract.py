import unittest

from scripts import analyze_mamba_ts_v3_multidataset_autocon_screen as analyzer


class MambaTSv3AutoConScreenContractTests(unittest.TestCase):
    def test_paired_delta_uses_exact_seed_coordinates(self):
        rows = []
        for dataset, reference, treatment in (
            ("ETTh1", 1.0, 0.99),
            ("ETTm1", 2.0, 2.01),
            ("electricity", 3.0, 3.0),
            ("traffic", 4.0, 3.9),
        ):
            for seed in (2021, 2022, 2023):
                rows.append({"phase": analyzer.PHASES[0], "dataset": dataset, "seed": seed, "best_val_mse": reference})
                rows.append({"phase": analyzer.PHASES[1], "dataset": dataset, "seed": seed, "best_val_mse": treatment})
        pairs = analyzer.paired(rows, analyzer.PHASES[1], analyzer.PHASES[0])
        self.assertEqual(len(pairs), 12)
        summary = analyzer.summarize(pairs)
        self.assertEqual({item["n_pairs"] for item in summary}, {3})
        self.assertEqual({item["dataset"] for item in summary}, {"ETTh1", "ETTm1", "electricity", "traffic"})

    def test_missing_coordinate_is_not_silently_imputed(self):
        rows = [
            {"phase": analyzer.PHASES[0], "dataset": "ETTh1", "seed": 2021, "best_val_mse": 1.0},
            {"phase": analyzer.PHASES[1], "dataset": "ETTh1", "seed": 2021, "best_val_mse": 1.0},
        ]
        with self.assertRaises(KeyError):
            analyzer.paired(rows, analyzer.PHASES[1], analyzer.PHASES[0])


if __name__ == "__main__":
    unittest.main()
