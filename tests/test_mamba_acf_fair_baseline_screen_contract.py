"""Contracts for the exploratory fair-baseline plan and readout."""

import json
import unittest
from pathlib import Path

import numpy as np

from scripts import analyze_mamba_acf_fair_baseline_screen as analyzer
from scripts.run_training_plan import REPO_ROOT, _command, _job_specs, _validate_plan


PLAN_PATH = REPO_ROOT / "docs/research/mamba_acf_fair_baseline_screen_plan.json"


def _plan():
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def _acf_config(relation=False, calibrated=False):
    plan = _plan()
    config = dict(plan["global"])
    dataset_channels = 7
    config.update(
        {
            "model": "MambaACFAutoCon" if relation else "MambaACF",
            "AutoCon": relation,
            "AutoCon_lambda": 0.03 if relation else 0.05,
            "seq_len": 96,
            "label_len": 48,
            "pred_len": 96,
            "enc_in": dataset_channels,
            "dec_in": dataset_channels,
            "c_out": dataset_channels,
            "train_epochs": 8,
            "mamba_use_acf_memory": True,
            "mamba_dynamic_acf": False,
            "mamba_acf_effective_lags": [72, 96, 120],
            "mamba_acf_effective_lag_source": "train_peaks_available",
            "mamba_acf_effective_lag_fallback": False,
            "mamba_acf_relation_effective_lags": [96, 120],
            "mamba_acf_relation_lag_source": "train_peak_subset",
            "mamba_acf_relation_lag_fallback": False,
            "mamba_acf_prior_artifact": "acf_prior.npz",
            "mamba_acf_prior_sha256": "0" * 64,
            "mamba_acf_prior_shape": [dataset_channels, 3],
            "mamba_acf_prior_lags": [72, 96, 120],
            "mamba_acf_horizon_calibration": calibrated,
            "mamba_acf_horizon_calibration_rank": 2,
            "mamba_acf_relation_loss": relation,
            "mamba_acf_relation_scope": "global",
            "mamba_acf_relation_lag_mask": relation,
            "mamba_acf_relation_min_lag": 96 if relation else 0,
            "AutoCon_acf_source": "raw",
        }
    )
    return config


class FairBaselineScreenContractTests(unittest.TestCase):
    def test_plan_and_registered_counts(self):
        plan = _plan()
        self.assertEqual(_validate_plan(plan), [])
        self.assertEqual(plan["registered_job_counts"]["total"], 42)
        self.assertEqual(plan["registered_job_counts"]["plain_baselines_h96"], 24)
        self.assertEqual(
            [phase["name"] for phase in plan["phases"]],
            ["plain_baselines_h96", "acf_memory_h96", "acf_memory_relation_h96", "acf_memory_relation_calibrated_h96"],
        )

    def test_runner_commands_keep_calibration_in_its_own_phase(self):
        plan = _plan()
        roots = {name: REPO_ROOT / ("tmp_fair_" + name) for name in ("results", "checkpoints", "test", "logs")}
        roots["result_log"] = REPO_ROOT / "tmp_fair_results.txt"
        for phase in plan["phases"]:
            jobs = _job_specs(plan, phase, {}, allow_unresolved=True)
            expected = 24 if phase["name"] == "plain_baselines_h96" else 6
            self.assertEqual(len(jobs), expected)
            _, command = _command(jobs[0], roots)
            self.assertEqual("--mamba_acf_horizon_calibration" in command, phase["name"].endswith("calibrated_h96"))
            relation_index = command.index("--mamba_acf_relation_min_lag")
            expected_min_lag = "96" if "relation" in phase["name"] else "0"
            self.assertEqual(command[relation_index + 1], expected_min_lag)

    def test_config_boundaries(self):
        baseline = dict(_plan()["global"])
        baseline.update({"model": "Mamba2", "AutoCon": False, "seq_len": 96, "label_len": 48, "pred_len": 96, "enc_in": 7, "dec_in": 7, "c_out": 7, "mamba_acf_relation_min_lag": 0})
        analyzer._validate_baseline_config(baseline, "Mamba2", "ETTh1", "baseline")
        analyzer._validate_acf_config(_acf_config(False, False), "MambaACF", "ETTh1", "memory", False, False)
        analyzer._validate_acf_config(_acf_config(True, True), "MambaACFAutoCon", "ETTh1", "calibrated", True, True)
        invalid = _acf_config(True, False)
        invalid["mamba_acf_relation_min_lag"] = 0
        with self.assertRaisesRegex(ValueError, "relation_min_lag"):
            analyzer._validate_acf_config(invalid, "MambaACFAutoCon", "ETTh1", "bad", True, False)

    def test_rank_two_checkpoint_contract(self):
        state = {"acf_horizon_calibration.coefficients": np.zeros((3, 2), dtype=np.float32)}
        info = analyzer._calibration_contract(state, Path("calibration.pth"), True, 3)
        self.assertEqual(info["parameter_count"], 6)
        self.assertEqual(info["coefficient_shape"], [3, 2])
        with self.assertRaisesRegex(ValueError, "shape"):
            analyzer._calibration_contract(
                {"acf_horizon_calibration.coefficients": np.zeros((2, 2), dtype=np.float32)},
                Path("bad.pth"), True, 3,
            )


if __name__ == "__main__":
    unittest.main()
