import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.audit_benchmark_matrix import (
    REPO_ROOT,
    REQUIRED_BASELINE_MODELS,
    _metric_status,
    audit_phase,
    build_audit,
    canonical_model,
    discover_results,
    expand_phase,
    _execution_consistency,
    load_plans,
    model_inventory,
)


class BenchmarkMatrixAuditTests(unittest.TestCase):
    def test_canonical_model_preserves_autocon_suffix(self):
        self.assertEqual(canonical_model({"model": "Mamba2", "AutoCon": True}), "Mamba2AutoCon")
        self.assertEqual(canonical_model("Mamba2AutoCon", False), "Mamba2AutoCon")
        self.assertEqual(canonical_model("Mamba2", False), "Mamba2")

    def test_canonical_model_recovers_runner_variable_arm_alias(self):
        self.assertEqual(canonical_model({
            "model": "MambaACF", "AutoCon": False,
            "mamba_variable_mixer": True,
            "plan_job_id": "variable_mixer_screen_ETTh1_MambaACFVariable_h96_s2021",
        }), "MambaACFVariable")
        self.assertEqual(canonical_model({
            "model": "MambaACF", "AutoCon": False,
            "mamba_variable_mixer": True,
            "plan_job_id": "ordinary_MambaACF_h96_s2021",
        }), "MambaACF")
        self.assertEqual(canonical_model({
            "model": "MambaACFAutoCon", "AutoCon": True,
            "mamba_variable_mixer": True,
            "plan_job_id": "variable_relation_ETTh1_MambaACFVariableAutoCon_h96_s2021",
        }), "MambaACFVariableAutoCon")

    def test_expand_phase_keeps_selected_lambda_unresolved(self):
        plan = {"global": {"features": "M"}}
        phase = {
            "name": "selection",
            "datasets": ["ETTh1"],
            "models": ["Mamba2", "Mamba2AutoCon"],
            "horizons": [96, 336],
            "seeds": [11, 12],
            "autocon_lambda": "selected",
        }
        rows = expand_phase(plan, phase)
        self.assertEqual(len(rows), 8)
        self.assertEqual({row["autocon_lambda"] for row in rows if row["model"] == "Mamba2"}, {None})
        self.assertEqual({row["autocon_lambda"] for row in rows if row["model"] == "Mamba2AutoCon"}, {"selected"})

    def test_discover_results_accepts_metrics_or_valid_stream_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            complete = root / "complete"
            complete.mkdir()
            (complete / "config.json").write_text(json.dumps({
                "data": "ETTh1", "model": "Mamba2", "AutoCon": True,
                "features": "M", "pred_len": 96, "run_seed": 7,
                "AutoCon_lambda": 0.03,
            }), encoding="utf-8")
            np.save(complete / "metrics.npy", np.asarray([1.0, 2.0, 3.0]))

            streamed = root / "streamed"
            streamed.mkdir()
            (streamed / "config.json").write_text(json.dumps({
                "data": "ETTh1", "model": "Mamba1", "AutoCon": False,
                "features": "M", "pred_len": 96, "seed": 8,
            }), encoding="utf-8")
            (streamed / "stream_metrics.json").write_text(json.dumps({
                "scaled": {"mse": 0.4},
            }), encoding="utf-8")
            rows = discover_results(root)
            self.assertEqual(len(rows), 2)
            by_model = {row["model"]: row for row in rows}
            self.assertTrue(by_model["Mamba2AutoCon"]["metrics_present"])
            self.assertTrue(by_model["Mamba1"]["stream_valid"])

    def test_metric_status_rejects_corrupt_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory)
            np.save(result / "metrics.npy", np.asarray([np.nan, 1.0, 1.0]))
            status = _metric_status(result)
            self.assertFalse(status["metrics_present"])
            self.assertTrue(status["metrics_npy"] is False)

    def test_metric_status_distinguishes_validation_only_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory)
            (result / "training_complete.json").write_text(
                json.dumps({"status": "trained_without_test"}), encoding="utf-8"
            )
            status = _metric_status(result)
            self.assertTrue(status["training_complete"])
            self.assertTrue(status["validation_complete"])
            self.assertFalse(status["metrics_present"])

    def test_audit_phase_reports_missing_baselines_and_extras(self):
        plan = {"name": "synthetic_plan", "status": "draft", "global": {"features": "M"}}
        phase = {
            "name": "synthetic_phase", "output_root": "artifacts/test_benchmark_matrix",
            "datasets": ["ETTh1"], "models": ["DLinear", "Mamba2AutoCon"],
            "horizons": [96], "seeds": [1],
        }
        root = REPO_ROOT / "artifacts/test_benchmark_matrix"
        observed = [
            {
                "dataset": "ETTh1", "model": "DLinear", "features": "M", "pred_len": 96,
                "seed": 1, "autocon_lambda": None, "metrics_present": True,
                "config_path": str(root / "dlinear" / "config.json"),
            },
            {
                "dataset": "ETTh1", "model": "Mamba2", "features": "M", "pred_len": 96,
                "seed": 1, "autocon": True, "autocon_lambda": 0.03, "metrics_present": True,
                "config_path": str(root / "mamba" / "config.json"),
            },
            {
                "dataset": "ETTh1", "model": "TimesNet", "features": "M", "pred_len": 96,
                "seed": 1, "autocon_lambda": None, "metrics_present": True,
                "config_path": str(root / "extra" / "config.json"),
            },
        ]
        report = audit_phase(plan, phase, observed)
        self.assertEqual(report["expected_jobs"], 2)
        self.assertEqual(report["matched_jobs"], 2)
        self.assertEqual(report["missing_jobs"], 0)
        self.assertEqual(report["extra_metric_rows"], 1)
        self.assertEqual(report["coverage"], 1.0)
        self.assertEqual(report["missing_required_baselines"], [
            model for model in REQUIRED_BASELINE_MODELS if model not in phase["models"]
        ])
        self.assertTrue(report["baseline_check_applicable"])

    def test_selection_phase_does_not_call_candidate_arms_missing_baselines(self):
        plan = {"name": "synthetic_plan", "global": {"features": "M"}}
        phase = {
            "name": "selection_M", "purpose": "validation-only candidate selection",
            "output_root": "artifacts/test_benchmark_matrix_selection",
            "datasets": ["ETTh1"], "models": ["Mamba2AutoCon"],
            "horizons": [96], "seeds": [1], "autocon_lambda": [0.03],
        }
        report = audit_phase(plan, phase, [])
        self.assertFalse(report["baseline_check_applicable"])
        self.assertEqual(report["missing_required_baselines"], [])

    def test_load_plans_rejects_json_without_phases(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad_plan.json"
            path.write_text(json.dumps({"version": 1}), encoding="utf-8")
            plans, errors = load_plans(path.parent, [path])
            self.assertEqual(plans, [])
            self.assertEqual(len(errors), 1)
            self.assertIn("missing phases", errors[0]["error"])

    def test_load_plans_discovers_dated_plan_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "plain_plan.json").write_text(
                json.dumps({"phases": []}), encoding="utf-8"
            )
            (root / "dated_plan_20260903.json").write_text(
                json.dumps({"phases": []}), encoding="utf-8"
            )
            (root / "classical_pilot_20260907.json").write_text(
                json.dumps({"phases": []}), encoding="utf-8"
            )
            # A launch sidecar is not a training plan and must remain excluded.
            (root / "screen_launch_sidecar_20260903.json").write_text(
                json.dumps({"phases": []}), encoding="utf-8"
            )

            plans, errors = load_plans(root)

            self.assertEqual(errors, [])
            self.assertEqual(
                [plan["name"] for plan in plans],
                ["classical_pilot_20260907", "dated_plan_20260903", "plain_plan"],
            )

    def test_build_audit_contains_registered_dataset_and_model_inventory(self):
        payload = build_audit([], Path("/tmp/nonexistent-benchmark-artifacts"))
        self.assertEqual(payload["schema_version"], 1)
        self.assertGreaterEqual(len(payload["datasets"]), 8)
        model_names = {row["name"] for row in model_inventory()}
        self.assertIn("MambaACFAutoCon", model_names)
        inventory = {row["name"]: row for row in model_inventory()}
        self.assertTrue(inventory["MambaACFVariable"]["source_exists"])
        self.assertEqual(inventory["MambaACFVariable"]["source_module"], "models/MambaACF.py")
        self.assertTrue(inventory["MambaACFVariableAutoCon"]["source_exists"])
        self.assertTrue(inventory["MambaTM1"]["source_exists"])
        self.assertEqual(inventory["MambaTM1"]["source_module"], "models/MambaTM1AutoCon.py")
        self.assertTrue(inventory["MambaTM2AutoCon"]["source_exists"])
        self.assertEqual(inventory["MambaTM2AutoCon"]["source_module"], "models/MambaTM2AutoCon.py")
        self.assertIn("iTransformer", model_names)
        self.assertIn("Autoformer", model_names)
        self.assertIn("FEDformer", model_names)
        self.assertIn("ETSformer", model_names)
        self.assertIn("Transformer", model_names)
        self.assertIn("Crossformer", model_names)
        self.assertEqual(payload["phases"], [])

    def test_execution_consistency_surfaces_immutable_snapshot_mismatch(self):
        plan = {
            "name": "historical_screen",
            "execution": {"authorized": True, "training_started": False},
            "registered_job_counts": {"training_started": False},
        }
        phase = {
            "plan": "historical_screen",
            "matched_jobs": 2,
            "validation_complete_rows": 0,
        }
        row = _execution_consistency([plan], [phase])[0]
        self.assertTrue(row["actual_artifacts_indicate_training"])
        self.assertFalse(row["execution_training_started"])
        self.assertIn(
            "artifacts_vs_execution_training_started_mismatch",
            row["warnings"],
        )
        self.assertNotIn(
            "execution_vs_registered_training_started_mismatch",
            row["warnings"],
        )


if __name__ == "__main__":
    unittest.main()
