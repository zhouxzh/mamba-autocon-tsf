"""Contract checks for the exploratory MambaACF relation-coverage screen."""

import json
import unittest
from pathlib import Path

from scripts import analyze_mamba_acf_relation_coverage_screen as analyzer


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = REPO_ROOT / "docs/research/mamba_acf_relation_coverage_screen_plan.json"


class MambaACFRelationCoverageScreenContractTests(unittest.TestCase):
    def test_plan_registers_three_paired_coverage_arms(self):
        plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            plan["status"], "exploratory_relation_coverage_screen_not_confirmatory"
        )
        self.assertEqual(
            [phase["name"] for phase in plan["phases"]],
            [name for name, *_ in analyzer.PHASES],
        )
        self.assertEqual(plan["registered_job_counts"]["total"], 18)
        self.assertEqual(plan["registered_job_counts"]["exploratory_total"], 18)
        expected = [("random", False), ("balanced", False), ("random", True)]
        for phase, (balance, active_norm) in zip(plan["phases"], expected):
            self.assertEqual(phase["datasets"], ["ETTh1", "traffic"])
            self.assertEqual(phase["horizons"], [96])
            self.assertEqual(phase["seeds"], [2021, 2022, 2023])
            self.assertEqual(phase["models"], ["MambaACFVariableAutoCon"])
            self.assertEqual(phase["autocon_lambda"], 0.03)
            self.assertTrue(phase["mamba_variable_mixer"])
            self.assertEqual(phase["mamba_variable_group_size"], 1)
            self.assertTrue(phase["mamba_use_acf_memory"])
            self.assertTrue(phase["mamba_acf_relation_loss"])
            self.assertEqual(phase["mamba_acf_relation_scope"], "global")
            self.assertEqual(phase["mamba_acf_relation_target_mode"], "hard_selected")
            self.assertEqual(phase["mamba_acf_relation_min_lag"], 96)
            self.assertEqual(phase["mamba_acf_relation_lag_balance"], balance)
            self.assertEqual(
                phase["mamba_acf_relation_active_anchor_normalization"], active_norm
            )

    def test_exposure_summary_checks_counts_and_reports_span(self):
        logs = {
            "acf_relation_lag_pair_counts": [
                {"96": 5, "120": 3}, {"96": 2, "120": 6}
            ],
            "acf_relation_lag_pair_exposure": [
                {"96": 0.625, "120": 0.375}, {"96": 0.25, "120": 0.75}
            ],
        }
        summary = analyzer._lag_exposure_summary(logs, Path("loss_logs.json"))
        self.assertEqual(summary["epochs"], 2)
        self.assertEqual(summary["total_pairs"], 16)
        self.assertEqual(summary["per_lag"]["96"]["pair_count"], 7)
        self.assertAlmostEqual(summary["per_lag"]["96"]["aggregate_exposure"], 7 / 16)
        self.assertAlmostEqual(summary["aggregate_balance_span"], 2 / 16)
        self.assertAlmostEqual(summary["mean_epoch_balance_span"], 0.375)

    def test_exposure_summary_rejects_inconsistent_exposure(self):
        logs = {
            "acf_relation_lag_pair_counts": [{"96": 1, "120": 3}],
            "acf_relation_lag_pair_exposure": [{"96": 0.5, "120": 0.5}],
        }
        with self.assertRaisesRegex(ValueError, "does not match counts"):
            analyzer._lag_exposure_summary(logs, Path("loss_logs.json"))

    def test_exposure_summary_rejects_unregistered_lag(self):
        logs = {
            "acf_relation_lag_pair_counts": [{"96": 1, "120": 1}],
            "acf_relation_lag_pair_exposure": [{"96": 0.5, "120": 0.5}],
        }
        with self.assertRaisesRegex(ValueError, "do not match effective lags"):
            analyzer._lag_exposure_summary(logs, Path("loss_logs.json"), [96])

    def test_config_requires_explicit_boolean_fields(self):
        with self.assertRaisesRegex(ValueError, "config mamba_low_rank_head is missing"):
            analyzer._assert_config({}, {"mamba_low_rank_head": False}, "job")
        with self.assertRaisesRegex(ValueError, "explicit boolean"):
            analyzer._assert_config(
                {"mamba_low_rank_head": "false"},
                {"mamba_low_rank_head": False},
                "job",
            )

    def test_contrast_uses_treatment_minus_reference(self):
        pairs = [
            {"reference_mse": 1.0, "balanced_mse": 0.9},
            {"reference_mse": 1.2, "balanced_mse": 1.0},
        ]
        result = analyzer._contrast("reference", "balanced", "mse", pairs)
        self.assertEqual(result["reference"], "reference")
        self.assertEqual(result["treatment"], "balanced")
        self.assertAlmostEqual(result["mean"], -0.15)
        self.assertEqual(result["wins_lower"], 2)


if __name__ == "__main__":
    unittest.main()
