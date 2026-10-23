"""Contract checks for the exploratory relation-target screen."""

import json
import unittest
from pathlib import Path

from scripts import analyze_mamba_acf_relation_target_screen as analyzer


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = REPO_ROOT / "docs/research/mamba_acf_relation_target_screen_plan.json"


class MambaACFRelationTargetScreenContractTests(unittest.TestCase):
    def test_plan_registers_hard_and_soft_paired_arms(self):
        plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            plan["status"], "exploratory_relation_target_screen_not_confirmatory"
        )
        self.assertEqual(
            [phase["name"] for phase in plan["phases"]],
            [name for name, *_ in analyzer.PHASES],
        )
        self.assertEqual(plan["registered_job_counts"]["total"], 12)
        self.assertEqual(plan["registered_job_counts"]["exploratory_total"], 12)
        self.assertEqual(plan["global"]["mamba_acf_relation_soft_min_abs"], 0.5)
        self.assertEqual(plan["global"]["mamba_acf_relation_diagnostic_min_abs"], 0.1)
        self.assertEqual(plan["phases"][0]["mamba_acf_relation_target_mode"], "hard_selected")
        self.assertTrue(plan["phases"][0]["mamba_acf_relation_lag_mask"])
        self.assertEqual(plan["phases"][1]["mamba_acf_relation_target_mode"], "soft_multi_peak")
        self.assertFalse(plan["phases"][1]["mamba_acf_relation_lag_mask"])
        for phase in plan["phases"]:
            self.assertEqual(phase["datasets"], ["ETTh1", "traffic"])
            self.assertEqual(phase["horizons"], [96])
            self.assertEqual(phase["seeds"], [2021, 2022, 2023])
            self.assertEqual(phase["mamba_acf_relation_diagnostic_min_abs"], 0.1)

    def test_lag_metadata_allows_soft_superset(self):
        analyzer._validate_lag_metadata([96, 120], [72, 96, 120], False, "soft")

    def test_lag_metadata_requires_equality_when_masked(self):
        with self.assertRaisesRegex(ValueError, "selected/effective lag mismatch"):
            analyzer._validate_lag_metadata([96, 120], [72, 96, 120], True, "hard")

    def test_lag_metadata_rejects_missing_effective_lag(self):
        with self.assertRaisesRegex(ValueError, "not contained"):
            analyzer._validate_lag_metadata([96, 120], [72, 96], False, "soft")


if __name__ == "__main__":
    unittest.main()
