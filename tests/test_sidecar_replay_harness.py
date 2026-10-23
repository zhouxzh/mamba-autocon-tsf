import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import scripts.replay_mamba_acf_sidecars as replay
from scripts.replay_mamba_acf_sidecars import (
    _canonical_lags,
    _command_sha256,
    _freeze_record,
    _metric_matches,
    _metrics,
    _ordered_lags,
    _phase_replay_contract,
    _replay_tool_identity,
    _replace_flag,
    _replay_config_identity,
    _run_job,
    _sidecar_replay_contract_satisfied,
    _source_config_identity,
    _validate_source_artifacts,
)


class SidecarReplayHarnessTests(unittest.TestCase):
    @staticmethod
    def _sha(path):
        digest = hashlib.sha256()
        digest.update(path.read_bytes())
        return digest.hexdigest()

    def _source_fixture(self, root, dataset_name="tiny"):
        phase_root = root / "phase"
        result_dir = phase_root / "results" / "setting"
        checkpoint_dir = phase_root / "checkpoints" / "setting"
        result_dir.mkdir(parents=True)
        checkpoint_dir.mkdir(parents=True)
        (root / "dataset").mkdir()
        dataset_path = root / "dataset" / "tiny.csv"
        dataset_path.write_text("date,value\n0,1\n", encoding="utf-8")
        (result_dir / "metrics.npy").write_bytes(b"metrics")
        (result_dir / "stream_metrics.json").write_text(
            json.dumps({"scaled": {name: 1.0 for name in replay.METRIC_NAMES}}),
            encoding="utf-8",
        )
        (result_dir / "acf_prior.npz").write_bytes(b"sidecar")
        (checkpoint_dir / "checkpoint.pth").write_bytes(b"checkpoint")
        config = {
            "setting": "setting",
            "plan_job_id": "job-1",
            "data": dataset_name,
            "model": "MambaACF",
            "seed": 7,
            "run_seed": 7,
            "root_path": "dataset",
            "data_path": "tiny.csv",
            "mamba_acf_prior_sha256": self._sha(result_dir / "acf_prior.npz"),
        }
        (result_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        command = [
            "python", "-u", "run.py",
            "--checkpoints", str(phase_root / "checkpoints"),
            "--results_dir", str(phase_root / "results"),
            "--test_results_dir", str(phase_root / "test_results"),
            "--result_log", str(phase_root / "results.txt"),
        ]
        record = {
            "phase": "phase",
            "job_id": "job-1",
            "dataset": dataset_name,
            "model": "MambaACF",
            "seed": 7,
            "status": "ok",
            "command": command,
            "command_sha256": _command_sha256(command),
            "expected_artifact": str(result_dir / "metrics.npy"),
        }
        hashes = {
            "config.json": self._sha(result_dir / "config.json"),
            "stream_metrics.json": self._sha(result_dir / "stream_metrics.json"),
            "acf_prior.npz": self._sha(result_dir / "acf_prior.npz"),
            "checkpoint.pth": self._sha(checkpoint_dir / "checkpoint.pth"),
        }
        record["artifact_hashes"] = hashes
        contract = {
            "phase": "phase",
            "provenance_id": "prov-1",
            "provenance_sha256": "prov-sha",
            "validation_sha256": "validation-sha",
            "manifest_sha256": "manifest-sha",
            "dataset_hashes": {dataset_name: self._sha(dataset_path)},
            "artifact_hashes_by_job": {"job-1": hashes},
        }
        return phase_root, record, contract, dataset_path

    def test_replace_flag_updates_only_the_registered_value(self):
        command = ["python", "run.py", "--results_dir", "old", "--is_training", "0"]
        _replace_flag(command, "--results_dir", "new")
        self.assertEqual(command, ["python", "run.py", "--results_dir", "new", "--is_training", "0"])

    def test_replay_tool_identity_hashes_the_executing_script(self):
        identity = _replay_tool_identity()
        tool_path = Path(replay.__file__).resolve()
        self.assertEqual(identity["path"], str(tool_path))
        self.assertEqual(identity["sha256"], self._sha(tool_path))

    def test_replace_flag_rejects_missing_value(self):
        with self.assertRaises(ValueError):
            _replace_flag(["python", "run.py", "--results_dir"], "--results_dir", "new")

    def test_metrics_reads_finite_scaled_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stream_metrics.json"
            path.write_text(
                json.dumps({
                    "scaled": {
                        "mae": 1.0,
                        "mse": 2.0,
                        "rmse": 3.0,
                        "mape": 4.0,
                        "mspe": 5.0,
                    }
                }),
                encoding="utf-8",
            )
            self.assertEqual(
                _metrics(path),
                {"mae": 1.0, "mse": 2.0, "rmse": 3.0, "mape": 4.0, "mspe": 5.0},
            )

    def test_lag_contract_keeps_order_for_audit_but_canonicalizes_sets(self):
        config = {"mamba_acf_effective_lags": [96, 72, 120]}
        self.assertEqual(_ordered_lags(config), [96, 72, 120])
        self.assertEqual(_canonical_lags(_ordered_lags(config)), [72, 96, 120])
        self.assertNotEqual(_ordered_lags(config), [72, 96, 120])

    def test_metric_tolerance_accepts_roundoff_and_rejects_drift(self):
        expected = {name: 1.0 for name in ("mae", "mse", "rmse", "mape", "mspe")}
        roundoff = {name: value + 1e-8 for name, value in expected.items()}
        drift = dict(roundoff)
        drift["mse"] = 1.01
        self.assertTrue(_metric_matches(roundoff, expected))
        self.assertFalse(_metric_matches(drift, expected))

    def test_sidecar_contract_requires_loader_only_for_live_acf_branch(self):
        off = {
            "sidecar_hash_equal": True,
            "source_acf_prior_required": False,
            "sidecar_load_message": False,
            "prior_fit_message": False,
        }
        required = {**off, "source_acf_prior_required": True}
        self.assertTrue(_sidecar_replay_contract_satisfied(off))
        self.assertFalse(
            _sidecar_replay_contract_satisfied(
                {**off, "sidecar_load_message": True}
            )
        )
        self.assertFalse(_sidecar_replay_contract_satisfied(required))
        self.assertTrue(
            _sidecar_replay_contract_satisfied(
                {**required, "sidecar_load_message": True}
            )
        )
        self.assertFalse(
            _sidecar_replay_contract_satisfied(
                {**off, "sidecar_hash_equal": False}
            )
        )
        self.assertFalse(
            _sidecar_replay_contract_satisfied(
                {**off, "prior_fit_message": True}
            )
        )

    def test_source_artifact_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            phase_root, record, contract, _ = self._source_fixture(Path(directory))
            result_dir = Path(record["expected_artifact"]).parent
            checkpoint_dir = phase_root / "checkpoints" / "setting"
            _validate_source_artifacts(record, result_dir, checkpoint_dir, contract)
            (result_dir / "acf_prior.npz").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "source artifact hash mismatch"):
                _validate_source_artifacts(record, result_dir, checkpoint_dir, contract)

    def test_source_config_identity_rejects_manifest_detachment(self):
        record = {"job_id": "job-1", "dataset": "tiny", "model": "MambaACF", "seed": 7}
        config = {
            "setting": "setting", "plan_job_id": "different", "data": "tiny",
            "model": "MambaACF", "seed": 7, "run_seed": 7,
            "mamba_acf_prior_sha256": "a" * 64,
        }
        with self.assertRaisesRegex(ValueError, "plan_job_id"):
            _source_config_identity(record, config, "setting", "a" * 64)

    def test_replay_config_identity_ignores_paths_but_detects_runtime_drift(self):
        source = {
            "setting": "setting",
            "plan_job_id": "job-1",
            "data": "tiny",
            "model": "MambaACF",
            "seed": 7,
            "run_seed": 7,
            "mamba_acf_prior_sha256": "a" * 64,
            "seq_len": 8,
            "label_len": 4,
            "pred_len": 2,
            "enc_in": 1,
            "c_out": 1,
            "d_model": 8,
            "d_ff": 16,
            "e_layers": 1,
            "mamba_d_state": 4,
            "mamba_d_conv": 2,
            "mamba_expand": 2,
            "mamba_headdim": 4,
            "mamba_scales": [1],
            "mamba_acf_lags": [2],
            "root_path": "source",
            "results_dir": "source/results",
            "is_training": 1,
        }
        replay_config = dict(source)
        replay_config.update({
            "root_path": "temporary",
            "results_dir": "temporary/results",
            "is_training": 0,
            "setting_override": "setting",
        })
        equal, mismatches = _replay_config_identity(
            source,
            replay_config,
            source_sidecar_sha256="a" * 64,
            source_prior_required=False,
        )
        self.assertTrue(equal, mismatches)
        replay_config["mamba_d_state"] = 8
        equal, mismatches = _replay_config_identity(
            source,
            replay_config,
            source_sidecar_sha256="a" * 64,
            source_prior_required=False,
        )
        self.assertFalse(equal)
        self.assertIn("mamba_runtime_contract.mamba_d_state", mismatches)

    def test_freeze_and_run_reject_dataset_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            phase_root, record, contract, dataset_path = self._source_fixture(root)
            with patch.object(replay, "REPO_ROOT", root):
                frozen = _freeze_record(phase_root, record, contract)
                dataset_path.write_text("date,value\n0,999\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "dataset hash mismatch"):
                    _run_job(
                        record, phase_root, root / "replay", 1,
                        phase_contract=contract, frozen_record=frozen,
                    )

    def test_phase_replay_contract_rejects_invalid_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            phase_root = Path(directory) / "phase"
            (phase_root / "logs").mkdir(parents=True)
            (phase_root / "provenance" / "source_snapshot").mkdir(parents=True)
            manifest = phase_root / "logs" / "job_manifest.jsonl"
            manifest.write_text("", encoding="utf-8")
            snapshot_file = phase_root / "provenance" / "source_snapshot" / "run.py"
            snapshot_file.write_text("# frozen\n", encoding="utf-8")
            source_files = {"run.py": self._sha(snapshot_file)}
            tree_hash = hashlib.sha256(
                json.dumps(source_files, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            plan = Path(directory) / "plan.json"
            plan.write_text("{}", encoding="utf-8")
            provenance = {
                "schema": 1, "phase": "phase", "provenance_id": "prov-1",
                "plan": str(plan), "plan_sha256": self._sha(plan),
                "dataset_hashes": {"tiny": "b" * 64},
                "source_snapshot": str((phase_root / "provenance" / "source_snapshot").resolve()),
                "source_files": source_files, "source_tree_sha256": tree_hash,
            }
            provenance_path = phase_root / "provenance.json"
            provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
            validation = {
                "valid": False, "phase": "phase",
                "manifest_sha256": self._sha(manifest),
                "provenance_id": "prov-1", "artifact_hashes_by_job": {},
            }
            (phase_root / "phase_validation.json").write_text(
                json.dumps(validation), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "phase validation is not valid"):
                _phase_replay_contract(phase_root)


if __name__ == "__main__":
    unittest.main()
