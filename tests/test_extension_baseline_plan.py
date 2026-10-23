"""Contracts for the opt-in dataset/baseline extension plan."""

import unittest
from pathlib import Path

from scripts.compute_naive_baselines import ALL_DATASETS, DATASETS, HIGH_DIM_DATASETS
from scripts.run_training_plan import (
    _job_specs,
    _load_json,
    _validate_launch_authorization,
    _validate_plan,
)


ROOT = Path(__file__).resolve().parents[1]


class ExtensionBaselinePlanTests(unittest.TestCase):
    def test_plan_is_valid_and_expands_expected_matrix(self):
        plan_path = ROOT / "docs/research/extension_baseline_plan.json"
        plan = _load_json(plan_path)
        self.assertEqual(_validate_plan(plan), [])
        phase = plan["phases"][0]
        jobs = _job_specs(plan, phase, {})
        # 2 datasets * 6 models * 3 horizons * 5 seeds.
        self.assertEqual(len(jobs), 180)
        self.assertIn("NLinear", {job["model"] for job in jobs})
        dimensions = {
            (job["dataset"], job["config"]["enc_in"], job["config"]["c_out"])
            for job in jobs
        }
        self.assertEqual(
            dimensions,
            {
                ("electricity", 321, 321),
                ("traffic", 862, 862),
            },
        )
        self.assertFalse(plan["execution"]["authorized"])
        self.assertFalse(plan["execution"]["training_started"])
        with self.assertRaisesRegex(ValueError, "plan is not authorized"):
            _validate_launch_authorization(plan, plan_path)
        binding = _validate_launch_authorization(plan, plan_path, enforce=False)
        self.assertFalse(binding["authorized"])

    def test_high_dimensional_naive_baselines_are_opt_in(self):
        self.assertTrue(set(HIGH_DIM_DATASETS).issubset(ALL_DATASETS))
        self.assertTrue(set(DATASETS).isdisjoint(HIGH_DIM_DATASETS))
        # The default registry remains the bounded low-dimensional set; the
        # CLI's --include_highdim flag is the explicit expansion point.
        self.assertNotIn("traffic", DATASETS)
        self.assertNotIn("electricity", DATASETS)


if __name__ == "__main__":
    unittest.main()
