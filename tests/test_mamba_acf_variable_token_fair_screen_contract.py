"""Contracts for the variable-token ACF null-control screen."""

import json
import hashlib
import unittest
from pathlib import Path

import numpy as np

from scripts import analyze_mamba_acf_variable_token_fair_screen as analyzer
from scripts.run_training_plan import REPO_ROOT, _command, _job_specs, _validate_plan


PLAN_PATH = REPO_ROOT / "docs/research/mamba_acf_variable_token_fair_screen_plan.json"


def _plan():
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


class VariableTokenFairScreenContractTests(unittest.TestCase):
    def test_plan_is_valid_and_execution_state_is_explicit(self):
        plan = _plan()
        self.assertEqual(_validate_plan(plan), [])
        self.assertIsInstance(plan["execution"]["authorized"], bool)
        self.assertIsInstance(plan["execution"]["training_started"], bool)
        self.assertEqual(plan["registered_job_counts"]["total"], 30)
        self.assertEqual(
            [phase["name"] for phase in plan["phases"]],
            [name for name, _ in analyzer.PHASES],
        )

    def test_alias_commands_preserve_null_and_calibration_boundaries(self):
        plan = _plan()
        roots = {
            "results": REPO_ROOT / "tmp_variable_results",
            "checkpoints": REPO_ROOT / "tmp_variable_checkpoints",
            "test": REPO_ROOT / "tmp_variable_test",
            "logs": REPO_ROOT / "tmp_variable_logs",
            "result_log": REPO_ROOT / "tmp_variable_results.txt",
        }
        for phase in plan["phases"]:
            jobs = _job_specs(plan, phase, {}, allow_unresolved=True)
            self.assertEqual(len(jobs), 6)
            job_id, command = _command(jobs[0], roots)
            self.assertEqual(command[command.index("--model") + 1], analyzer.ALIASES[dict(analyzer.PHASES)[phase["name"]]][1])
            self.assertIn("--mamba_variable_mixer", command)
            record = {
                "job_id": job_id,
                "command": command,
                "command_sha256": hashlib.sha256(
                    json.dumps(command, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
                ).hexdigest(),
            }
            kind = dict(analyzer.PHASES)[phase["name"]]
            contract = {
                "alias": analyzer.ALIASES[kind][0],
                "executable": analyzer.ALIASES[kind][1],
                "relation": analyzer.ALIASES[kind][2],
                "memory": analyzer.ALIASES[kind][3],
                "lambda": analyzer.ALIASES[kind][4],
                "calibrated": kind == "calibrated",
            }
            analyzer._validate_command(record, contract, job_id)
            if kind == "relation_null":
                self.assertEqual(command[command.index("--AutoCon_lambda") + 1], "0.0")
            if kind == "calibrated":
                self.assertIn("--mamba_acf_horizon_calibration", command)

    def test_primary_contrast_is_null_to_relation_and_stopping_rule_is_explicit(self):
        rows = []
        signature = ("shared",)
        for _, kind in analyzer.PHASES:
            for dataset in analyzer.EXPECTED_DATASETS:
                for seed in analyzer.EXPECTED_SEEDS:
                    mse = 1.0
                    if kind == "relation":
                        # Four favorable coordinates, with at least one per dataset.
                        mse = 0.9 if (dataset, seed) in {("ETTh1", 2021), ("ETTh1", 2022), ("traffic", 2021), ("traffic", 2022)} else 1.1
                    elif kind == "calibrated":
                        mse = 0.95
                    rows.append({
                        "arm": kind,
                        "dataset": dataset,
                        "seed": seed,
                        "mse": mse,
                        "mae": mse,
                        "job_id": f"{kind}_{dataset}_{seed}",
                        "state_numel": 10,
                        "sidecar_sha256": "a" * 64,
                        "effective_lags": [24, 168, 336],
                        "relation_lags": [168, 336] if kind in {"relation_null", "relation", "calibrated"} else [],
                        "trunk_signature": signature,
                        "executable_model": "MambaACFAutoCon" if kind in {"relation_null", "relation", "calibrated"} else "MambaACF",
                        "runtime": {"acf_effective_memory_rms": {"mean": None}},
                        "relation_diagnostics": {},
                        "calibration": {},
                        "calibration_state": {"parameter_count": 6 if kind == "calibrated" else 0},
                    })
        result = analyzer.build_readout({"_path": str(PLAN_PATH)}, rows)
        self.assertIn("relation_lambda003_minus_lambda0", result["contrasts"])
        self.assertEqual(result["contrasts"]["relation_lambda003_minus_lambda0"]["reference_arm"], "relation_null")
        self.assertEqual(result["contrasts"]["relation_lambda003_minus_lambda0"]["treatment_arm"], "relation")
        self.assertTrue(result["stopping_decision"]["meets_descriptive_screen"])

    def test_rank_two_calibration_state_contract(self):
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
