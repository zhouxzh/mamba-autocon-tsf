"""Contracts for the exploratory horizon-amplitude calibration screen."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from scripts import analyze_mamba_acf_horizon_calibration_screen as analyzer
from scripts.run_training_plan import (
    REPO_ROOT,
    _command,
    _job_specs,
    _validate_plan,
)


PLAN_PATH = REPO_ROOT / "docs/research/mamba_acf_horizon_calibration_screen_plan.json"


def _plan() -> dict:
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def _synthetic_config(enabled: bool) -> dict:
    """Construct a persisted-config-shaped fixture without training."""

    plan = _plan()
    config = dict(plan["global"])
    config.update(
        {
            "model": "MambaACFAutoCon",
            "AutoCon": True,
            "AutoCon_lambda": 0.03,
            "AutoCon_acf_source": "raw",
            "AutoCon_warmup_epochs": 0,
            "AutoCon_multiscales": [96],
            "seq_len": 96,
            "label_len": 48,
            "pred_len": 96,
            "enc_in": 7,
            "dec_in": 7,
            "c_out": 7,
            "train_epochs": 8,
            "patience": 3,
            "mamba_acf_effective_lags": [72, 96, 120],
            "mamba_acf_effective_lag_source": "train_peaks_available",
            "mamba_acf_effective_lag_fallback": False,
            "mamba_acf_relation_effective_lags": [96, 120],
            "mamba_acf_relation_lag_source": "train_peak_subset",
            "mamba_acf_relation_lag_fallback": False,
            "mamba_acf_prior_artifact": "acf_prior.npz",
            "mamba_acf_prior_sha256": "0" * 64,
            "mamba_acf_prior_shape": [7, 3],
            "mamba_acf_prior_lags": [72, 96, 120],
            "mamba_acf_prior_full_shape": [7, 721],
            "mamba_acf_horizon_calibration": enabled,
            "mamba_acf_horizon_calibration_rank": 2,
        }
    )
    # These names are written by run.py even though the plan uses their
    # shorter global aliases.
    config.setdefault("mamba_patch_order", "time_major")
    config.setdefault("mamba_patch_mixer_depth", 1)
    config.setdefault("mamba_local_patch_len", 16)
    config.setdefault("mamba_local_patch_stride", 8)
    config.setdefault("mamba_local_width", 32)
    config.setdefault("mamba_local_rank", 16)
    config.setdefault("mamba_local_depth", 1)
    config.setdefault("mamba_variable_mixer_depth", 1)
    return config


def _diagnostics(enabled: bool) -> dict:
    relation = {
        key: {"n": 1, "mean": 0.5, "min": 0.5, "max": 0.5, "last": 0.5}
        for key in analyzer.RELATION_LOG_KEYS
    }
    runtime = {
        key: {"n": 1, "mean": 0.5, "min": 0.5, "max": 0.5, "last": 0.5}
        for key in analyzer.RUNTIME_LOG_KEYS
    }
    calibration = {
        key: {
            "n": 1,
            "finite_n": 1 if enabled else 0,
            "null_n": 0 if enabled else 1,
            "mean": 1.0 if enabled else None,
            "min": 1.0 if enabled else None,
            "max": 1.0 if enabled else None,
            "last": 1.0 if enabled else None,
        }
        for key in analyzer.CALIBRATION_LOG_KEYS
    }
    return {
        "relation_diagnostics": relation,
        "runtime_diagnostics": runtime,
        "calibration_diagnostics": calibration,
    }


class MambaACFHorizonCalibrationScreenContractTests(unittest.TestCase):
    def test_plan_is_runner_valid_and_registers_twelve_paired_jobs(self):
        plan = _plan()
        self.assertEqual(
            plan["status"],
            "exploratory_horizon_amplitude_calibration_screen_not_confirmatory",
        )
        self.assertEqual(_validate_plan(plan), [])
        self.assertEqual(
            [phase["name"] for phase in plan["phases"]],
            [name for name, _, _ in analyzer.EXPECTED_PHASES],
        )
        self.assertEqual(plan["registered_job_counts"]["total"], 12)
        self.assertEqual(plan["registered_job_counts"]["exploratory_total"], 12)
        self.assertEqual(plan["registered_job_counts"]["training_total"], 12)
        self.assertTrue(plan["require_acf_sidecar"])
        self.assertNotEqual(
            plan["phases"][0]["output_root"], plan["phases"][1]["output_root"]
        )
        for phase in plan["phases"]:
            self.assertEqual(phase["datasets"], ["ETTh1", "traffic"])
            self.assertEqual(phase["seeds"], [2021, 2022, 2023])
            self.assertEqual(phase["horizons"], [96])
            self.assertEqual(phase["models"], ["MambaACFAutoCon"])
            self.assertTrue(phase["mamba_use_acf_memory"])
            self.assertFalse(phase["mamba_dynamic_acf"])
            self.assertTrue(phase["mamba_acf_relation_loss"])
            self.assertEqual(phase["mamba_acf_relation_scope"], "global")
            self.assertEqual(phase["mamba_acf_relation_target_mode"], "hard_selected")
            self.assertEqual(phase["mamba_acf_horizon_calibration_rank"], 2)
            self.assertEqual(phase["epochs"], 8)

    def test_runner_command_contains_only_the_registered_treatment_flag(self):
        plan = _plan()
        roots = {
            "results": REPO_ROOT / "tmp_calibration_results",
            "checkpoints": REPO_ROOT / "tmp_calibration_checkpoints",
            "test": REPO_ROOT / "tmp_calibration_test_results",
            "logs": REPO_ROOT / "tmp_calibration_logs",
            "result_log": REPO_ROOT / "tmp_calibration_results.txt",
        }
        for phase, expected_flag in zip(plan["phases"], (False, True)):
            jobs = _job_specs(plan, phase, {}, allow_unresolved=True)
            self.assertEqual(len(jobs), 6)
            _, command = _command(jobs[0], roots)
            self.assertEqual(
                "--mamba_acf_horizon_calibration" in command, expected_flag
            )
            rank_index = command.index("--mamba_acf_horizon_calibration_rank")
            self.assertEqual(command[rank_index + 1], "2")

    def test_config_contract_requires_explicit_one_factor_fields(self):
        off = analyzer._check_config(_synthetic_config(False), False, "off")
        on = analyzer._check_config(_synthetic_config(True), True, "on")
        self.assertEqual(off["effective_lags"], [72, 96, 120])
        self.assertEqual(on["prior_shape"], [7, 3])
        invalid = _synthetic_config(False)
        invalid["mamba_acf_relation_scope"] = "both"
        with self.assertRaisesRegex(ValueError, "relation_scope"):
            analyzer._check_config(invalid, False, "invalid")
        invalid_bool = _synthetic_config(False)
        invalid_bool["mamba_acf_horizon_calibration"] = "false"
        with self.assertRaisesRegex(ValueError, "explicit boolean"):
            analyzer._check_config(invalid_bool, False, "invalid_bool")

    def test_checkpoint_state_contract_is_nested_and_rank_two(self):
        off = analyzer._calibration_contract({}, Path("off.pth"), False, 3)
        self.assertFalse(off["enabled"])
        self.assertEqual(off["parameter_count"], 0)
        on_state = {
            "acf_horizon_calibration.coefficients": torch.zeros(3, 2)
        }
        on = analyzer._calibration_contract(on_state, Path("on.pth"), True, 3)
        self.assertTrue(on["enabled"])
        self.assertEqual(on["coefficient_shape"], [3, 2])
        self.assertEqual(on["parameter_count"], 6)
        self.assertAlmostEqual(on["amplitude"]["mean"], 1.0)
        with self.assertRaisesRegex(ValueError, "shape"):
            analyzer._calibration_contract(
                {"acf_horizon_calibration.coefficients": torch.zeros(2, 2)},
                Path("bad.pth"),
                True,
                3,
            )
        with self.assertRaisesRegex(ValueError, "calibration-off"):
            analyzer._calibration_contract(on_state, Path("bad_off.pth"), False, 3)

    def test_paired_readout_rejects_sidecar_or_common_config_mismatch(self):
        signature = {"shared": "value"}
        rows = []
        for arm, enabled in (("off", False), ("on", True)):
            for dataset in analyzer.EXPECTED_DATASETS:
                for seed in analyzer.EXPECTED_SEEDS:
                    row = {
                        "arm": arm,
                        "dataset": dataset,
                        "seed": seed,
                        "mse": 1.0 if not enabled else 0.9,
                        "mae": 0.5 if not enabled else 0.45,
                        "elapsed_sec": 10.0,
                        "sidecar_sha256": "a" * 64,
                        "effective_lags": [72, 96, 120],
                        "config_signature": signature,
                        "state": {"enabled": enabled},
                        "job_id": f"{arm}-{dataset}-{seed}",
                        "autocon_loss": {"mean": 1.0},
                        **_diagnostics(enabled),
                    }
                    rows.append(row)
        result = analyzer.build_readout({"_path": str(PLAN_PATH)}, rows)
        self.assertEqual(len(result["pairs"]), 6)
        self.assertAlmostEqual(
            result["contrasts"]["on_minus_off"]["mse"]["mean"], -0.1
        )
        tampered = [dict(row) for row in rows]
        tampered[-1] = dict(tampered[-1])
        tampered[-1]["sidecar_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "sidecars differ"):
            analyzer.build_readout({"_path": str(PLAN_PATH)}, tampered)


if __name__ == "__main__":
    unittest.main()
