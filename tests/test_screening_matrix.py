"""Contracts for the read-only screening matrix analysis utility."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.analyze_screening_matrix import (
    build_payload,
    discover_metric_rows,
    discover_resource_rows,
    paired_deltas,
    read_scaled_mse,
    summarize_seed_rows,
    summarize_models,
    write_outputs,
)


def _write_run(root: Path, name: str, config: dict, *, stream_mse=None, npy_mse=None) -> Path:
    result = root / name
    result.mkdir(parents=True, exist_ok=True)
    (result / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if stream_mse is not None:
        (result / "stream_metrics.json").write_text(
            json.dumps({"scaled": {"mse": stream_mse}}), encoding="utf-8"
        )
    if npy_mse is not None:
        np.save(result / "metrics.npy", np.asarray([0.1, npy_mse, 0.3]))
    return result


class ScreeningMatrixTests(unittest.TestCase):
    def _config(self, model, seed, *, epochs=20, autocon=False):
        return {
            "data": "toy",
            "model": model,
            "AutoCon": autocon,
            "features": "M",
            "seq_len": 12,
            "pred_len": 4,
            "run_seed": seed,
            "train_epochs": epochs,
            "des": "confirmatory",
        }

    def test_stream_metrics_is_preferred_over_legacy_npy(self):
        with tempfile.TemporaryDirectory() as directory:
            result = _write_run(
                Path(directory), "run", self._config("PatchTST", 1),
                stream_mse=0.25, npy_mse=0.75,
            )
            self.assertEqual(read_scaled_mse(result), (0.25, "stream_metrics.json", None))

    def test_metrics_npy_is_supported_when_stream_artifact_is_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            result = _write_run(
                Path(directory), "run", self._config("DLinear", 1), npy_mse=0.5
            )
            self.assertEqual(read_scaled_mse(result), (0.5, "metrics.npy", None))

    def test_valid_legacy_metric_falls_back_when_stream_is_corrupt(self):
        with tempfile.TemporaryDirectory() as directory:
            result = _write_run(
                Path(directory), "run", self._config("DLinear", 1), stream_mse=0.5, npy_mse=0.75
            )
            (result / "stream_metrics.json").write_text("{not-json", encoding="utf-8")
            self.assertEqual(read_scaled_mse(result), (0.75, "metrics.npy", None))

    def test_duplicate_coordinate_is_marked_conflicting_and_not_used_for_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_run(root, "base", self._config("PatchTST", 1), stream_mse=1.0)
            _write_run(root, "candidate_a", self._config("Mamba2", 1), stream_mse=0.8)
            _write_run(root, "candidate_b", self._config("Mamba2", 1), stream_mse=0.9)
            raw, issues = discover_metric_rows([root])
            self.assertEqual(issues, [])
            seed_rows = summarize_seed_rows(raw)
            candidate = next(row for row in seed_rows if row["model"] == "Mamba2")
            self.assertTrue(candidate["duplicate_conflict"])
            self.assertIsNone(candidate["scaled_mse"])
            delta = paired_deltas(seed_rows, ["PatchTST"])
            self.assertEqual(delta[0]["status"], "ambiguous_candidate")
            self.assertIsNone(delta[0]["paired_delta"])

    def test_even_identical_duplicate_coordinates_are_not_treated_as_a_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_run(root, "base", self._config("PatchTST", 1), stream_mse=1.0)
            _write_run(root, "candidate_a", self._config("Mamba2", 1), stream_mse=0.8)
            _write_run(root, "candidate_b", self._config("Mamba2", 1), stream_mse=0.8)
            raw, _ = discover_metric_rows([root])
            seed_rows = summarize_seed_rows(raw)
            delta = paired_deltas(seed_rows, ["PatchTST"])
            self.assertEqual(delta[0]["status"], "ambiguous_candidate")
            self.assertIsNone(delta[0]["paired_delta"])

    def test_paired_delta_is_candidate_minus_baseline_and_autocon_name_is_canonical(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_run(root, "base", self._config("PatchTST", 7, epochs=20), stream_mse=1.0)
            _write_run(root, "candidate", self._config("MambaACF", 7, epochs=20, autocon=True), stream_mse=0.75)
            raw, _ = discover_metric_rows([root])
            seed_rows = summarize_seed_rows(raw)
            rows = paired_deltas(seed_rows, ["PatchTST"])
            self.assertEqual(rows[0]["candidate_model"], "MambaACFAutoCon")
            self.assertEqual(rows[0]["status"], "paired")
            self.assertAlmostEqual(rows[0]["paired_delta"], -0.25)
            self.assertAlmostEqual(rows[0]["relative_delta"], -0.25)

    def test_model_summary_marks_one_seed_and_short_budget_exploratory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_run(root, "base", self._config("PatchTST", 1, epochs=3), stream_mse=1.0)
            raw, _ = discover_metric_rows([root], short_budget_epochs=3)
            seed_rows = summarize_seed_rows(raw)
            summary = summarize_models(seed_rows)
            self.assertEqual(len(summary), 1)
            self.assertTrue(summary[0]["one_seed"])
            self.assertTrue(summary[0]["short_budget"])
            self.assertTrue(summary[0]["exploratory"])

    def test_multi_seed_full_budget_summary_is_not_marked_exploratory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed, mse in ((1, 1.0), (2, 0.9), (3, 1.1)):
                _write_run(root, f"base_{seed}", self._config("PatchTST", seed, epochs=20), stream_mse=mse)
            raw, _ = discover_metric_rows([root], short_budget_epochs=3)
            seed_rows = summarize_seed_rows(raw)
            summary = summarize_models(seed_rows)
            self.assertEqual(summary[0]["n_seeds"], 3)
            self.assertFalse(summary[0]["short_budget"])
            self.assertFalse(summary[0]["exploratory"])

    def test_resource_v2_rows_and_protocol_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            resource = root / "model_resources.json"
            resource.write_text(json.dumps({
                "benchmark_protocol": {"device": "cuda:0", "benchmark_batch_size": 2, "timed_forwards": 20},
                "resource_records": [{
                    "model": "Mamba2", "dataset": "toy", "features": "M", "seq_len": 12, "pred_len": 4,
                    "parameters": {"total": 100, "trainable": 90},
                    "gpu_inference": {
                        "latency_ms_mean": 0.5, "latency_ms_median": 0.4, "latency_ms_p95": 0.7,
                        "steady_state_peak_allocated_mb": 12.0,
                        "synthetic_input_sequences_per_sec": 20.0,
                        "device": "cuda:0", "benchmark_batch_size": 2, "timed_forwards": 20,
                    },
                }],
            }), encoding="utf-8")
            rows, issues = discover_resource_rows([], [root])
            self.assertEqual(issues, [])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["peak_memory_mb"], 12.0)
            payload = build_payload([], [], [root], ["PatchTST"])
            self.assertEqual(payload["counts"]["resource_rows"], 1)
            self.assertTrue(payload["resource_summary"][0]["measurement_only"])
            self.assertTrue(payload["resource_summary"][0]["protocol_consistent"])

    def test_output_refuses_input_root_and_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_root = root / "results"
            result_root.mkdir()
            payload = build_payload([], [], [], ["PatchTST"])
            with self.assertRaises(ValueError):
                write_outputs(payload, result_root, [result_root])
            output = root / "analysis"
            write_outputs(payload, output, [result_root])
            with self.assertRaises(FileExistsError):
                write_outputs(payload, output, [result_root])
            write_outputs(payload, output, [result_root], overwrite=True)


if __name__ == "__main__":
    unittest.main()
