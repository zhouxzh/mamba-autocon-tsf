import json
import tempfile
import unittest
from pathlib import Path

from scripts.attest_selection import (
    _canonical_hash,
    _require_replay_equal,
    _validate,
    verify_selection_attestation,
)


class SelectionAttestationTests(unittest.TestCase):
    def test_canonical_hash_is_order_independent(self):
        self.assertEqual(
            _canonical_hash({"Mamba2AutoCon": 1.0, "Mamba1AutoCon": 1.0}),
            _canonical_hash({"Mamba1AutoCon": 1.0, "Mamba2AutoCon": 1.0}),
        )

    def test_invalid_validation_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection_path = root / "selection.json"
            validation_path = root / "validation.json"
            plan_path = root / "plan.json"
            manifest_path = root / "manifest.jsonl"
            selection_path.write_text(json.dumps({"phase": "selection"}), encoding="utf-8")
            validation_path.write_text(json.dumps({"valid": False}), encoding="utf-8")
            plan_path.write_text(json.dumps({"phases": []}), encoding="utf-8")
            manifest_path.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                _validate(selection_path, validation_path, plan_path, manifest_path)

    def test_replay_binding_rejects_changed_summary(self):
        recorded = {
            "algorithm": "independent_validation_trajectory_recompute",
            "mapping_matches": True,
            "rows_match": True,
            "manifest_rows": 1,
            "expected_cells": 1,
            "candidate_max_abs_difference": 0.0,
            "candidate_summary": {
                "Mamba1AutoCon": [{"lambda": 0.1, "mean_relative_val_loss": 1.0, "n_jobs": 1}],
            },
            "recomputed_rows_sha256": "abc",
        }
        changed = dict(recorded, recomputed_rows_sha256="def")
        with self.assertRaises(ValueError):
            _require_replay_equal(recorded, changed)

    def test_replay_binding_rejects_row_mismatch(self):
        replay = {
            "algorithm": "independent_validation_trajectory_recompute",
            "mapping_matches": True,
            "rows_match": False,
            "manifest_rows": 1,
            "expected_cells": 1,
            "candidate_max_abs_difference": 0.0,
            "candidate_summary": {
                "Mamba1AutoCon": [{"lambda": 0.1, "mean_relative_val_loss": 1.0, "n_jobs": 1}],
            },
            "recomputed_rows_sha256": "abc",
        }
        with self.assertRaises(ValueError):
            _require_replay_equal(replay, replay)

    def test_replay_binding_rejects_candidate_summary_change(self):
        replay = {
            "algorithm": "independent_validation_trajectory_recompute",
            "mapping_matches": True,
            "rows_match": True,
            "manifest_rows": 1,
            "expected_cells": 1,
            "candidate_max_abs_difference": 0.0,
            "candidate_summary": {
                "Mamba1AutoCon": [{"lambda": 0.1, "mean_relative_val_loss": 1.0, "n_jobs": 1}],
            },
            "recomputed_rows_sha256": "abc",
        }
        changed = dict(replay, candidate_summary={
            "Mamba1AutoCon": [{"lambda": 1.0, "mean_relative_val_loss": 1.0, "n_jobs": 1}],
        })
        with self.assertRaises(ValueError):
            _require_replay_equal(changed, replay)

    def test_registered_sidecar_path_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection = root / "selected_hyperparameters.json"
            plan = root / "plan.json"
            selection.write_text(json.dumps({"phase": "selection"}), encoding="utf-8")
            plan.write_text(json.dumps({"phases": []}), encoding="utf-8")
            phase = {"name": "confirmatory", "selection_phase": "selection", "features": "S"}
            with self.assertRaises(ValueError):
                verify_selection_attestation(selection, plan, phase, root / "other.json")


if __name__ == "__main__":
    unittest.main()
