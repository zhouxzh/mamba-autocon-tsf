import unittest

from scripts import analyze_mamba_acf_multidataset_candidate_screen as analyzer


class MambaAcfCandidateScreenContractTests(unittest.TestCase):
    def test_telemetry_mean_accepts_scalar_and_sequence(self):
        self.assertAlmostEqual(analyzer._telemetry_mean({"x": 2.0}, "x"), 2.0)
        self.assertAlmostEqual(analyzer._telemetry_mean({"x": [1.0, 3.0]}, "x"), 2.0)
        self.assertIsNone(analyzer._telemetry_mean({"x": []}, "x"))

    def test_paired_uses_exact_dataset_seed_coordinates(self):
        rows = []
        for dataset, treatment, reference in (
            ("ETTh1", 1.0, 2.0),
            ("ETTm1", 3.0, 2.0),
            ("electricity", 4.0, 4.0),
            ("traffic", 2.0, 5.0),
        ):
            for seed in (2021, 2022, 2023):
                rows.append({"phase": "treatment", "dataset": dataset, "seed": seed, "best_val_mse": treatment})
                rows.append({"phase": "reference", "dataset": dataset, "seed": seed, "best_val_mse": reference})
        pairs = analyzer.paired(rows, "treatment", "reference")
        self.assertEqual(len(pairs), 12)
        self.assertEqual(sum(item["delta_mse"] < 0 for item in pairs), 6)
        summary = analyzer.pair_summary(pairs)
        self.assertEqual({item["dataset"] for item in summary}, {"ETTh1", "ETTm1", "electricity", "traffic"})
        self.assertEqual({item["n_pairs"] for item in summary}, {3})

    def test_confidence_interval_is_zero_for_single_pair(self):
        self.assertEqual(analyzer._ci95([1.0]), 0.0)


if __name__ == "__main__":
    unittest.main()
