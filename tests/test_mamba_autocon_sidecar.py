"""Regression tests for MambaACF AutoCon sidecar initialization."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import exp.exp_long_term_forecasting_with_AutoCon as autocon_experiment
from exp.exp_long_term_forecasting_with_AutoCon import (
    Exp_Long_Term_Forecast_with_AutoCon,
)
from exp.exp_long_term_forecasting_with_mamba import _MambaExperimentMixin
from layers.losses import ACFRelationLoss
from utils.acf_relation_contract import resolve_acf_source


def _args(artifact_path):
    return SimpleNamespace(
        mamba_acf_prior_artifact_path=str(artifact_path),
        mamba_acf_prior_expected_sha256="",
        mamba_acf_relation_loss=True,
        mamba_acf_relation_lag_mask=True,
        mamba_acf_relation_min_lag=0,
        mamba_acf_relation_target_mode="hard_selected",
        mamba_acf_relation_lag_tolerance=0,
        mamba_acf_relation_scope="global",
        mamba_acf_relation_channel_reduce="mean",
        mamba_acf_relation_soft_min_abs=0.1,
        mamba_acf_relation_selected_lag_boost=0.0,
        AutoCon_temperature=0.2,
        AutoCon_relation_temperature=0.25,
        AutoCon_sign_weight=0.25,
        AutoCon_sign_margin=0.1,
    )


class _SidecarModel:
    channels = 2
    acf_lags = (3, 7)

    def __init__(self, full_values=None):
        self.loaded = []
        self._full_values = (
            np.asarray(full_values, dtype=np.float32)
            if full_values is not None else None
        )

    def load_acf_prior(self, path):
        self.loaded.append(str(path))
        return True

    def set_acf_prior(self, *_args, **_kwargs):
        return None

    def export_acf_prior_state(self):
        if self._full_values is None:
            return {
                "acf_values": np.asarray([[0.4, -0.2], [0.3, -0.1]], dtype=np.float32),
                "lags": np.asarray(self.acf_lags, dtype=np.int64),
                "acf_full_values": None,
            }
        return {
            "acf_values": self._full_values[:, list(self.acf_lags)],
            "lags": np.asarray(self.acf_lags, dtype=np.int64),
            "acf_full_values": self._full_values,
        }


class MambaAutoConSidecarTests(unittest.TestCase):
    def test_relation_component_lifecycle_persists_active_and_inactive_epochs(self):
        """The four fields cannot be dropped between batch and JSON schemas."""

        specs = autocon_experiment.ACF_RELATION_COMPONENT_SPECS
        self.assertEqual(len(specs), 4)
        logs = {}
        autocon_experiment._initialize_relation_component_logs(logs)
        self.assertEqual({log_key for _, log_key, _ in specs}, set(logs))

        # An epoch without AutoCon still emits one explicit null per component
        # rather than shortening a series or leaving a stale prior value.
        inactive_train_log = {}
        autocon_experiment._initialize_relation_component_train_log(
            inactive_train_log
        )
        autocon_experiment._reduce_relation_component_epoch(inactive_train_log)
        autocon_experiment._append_relation_component_epoch(logs, inactive_train_log)

        objective = ACFRelationLoss(
            torch.tensor([[1.0, 0.9, -0.5, 0.2]]),
            relation_scope="both",
            selected_lags=[1, 2],
            sign_weight=0.25,
        )
        objective(torch.randn(4, 5, 3), torch.tensor([0, 1, 2, 3]))
        active_train_log = {}
        autocon_experiment._initialize_relation_component_train_log(active_train_log)
        autocon_experiment._record_relation_component_batch(
            active_train_log, objective
        )
        autocon_experiment._reduce_relation_component_epoch(active_train_log)
        autocon_experiment._append_relation_component_epoch(logs, active_train_log)

        for train_key, log_key, _ in specs:
            self.assertIsNone(logs[log_key][0])
            self.assertEqual(logs[log_key][1], active_train_log[train_key])
            self.assertTrue(np.isfinite(logs[log_key][1]))
            self.assertEqual(len(logs[log_key]), 2)
        json.dumps(logs)

    def test_relation_diagnostic_snapshot_returns_python_scalars(self):
        objective = ACFRelationLoss(
            torch.tensor([[1.0, 0.9, -0.5, 0.2]]),
            relation_scope="global",
            selected_lags=[1, 2],
        )
        objective(torch.randn(4, 5, 3), torch.tensor([0, 1, 2, 3]))
        values = autocon_experiment._read_detached_scalar_fields(
            objective,
            autocon_experiment.ACF_RELATION_DIAGNOSTIC_SPECS,
        )
        self.assertEqual(
            set(values),
            {key for key, _ in autocon_experiment.ACF_RELATION_DIAGNOSTIC_SPECS},
        )
        self.assertTrue(all(isinstance(value, float) for value in values.values()))

    def test_init_autocon_prefers_lowercase_source_when_aliases_conflict(self):
        experiment = Exp_Long_Term_Forecast_with_AutoCon.__new__(
            Exp_Long_Term_Forecast_with_AutoCon
        )
        args = SimpleNamespace(
            seq_len=4,
            batch_size=2,
            model="AutoConNet",
            autocon_acf_source="raw",
            AutoCon_acf_source="residual",
        )
        experiment.args = args
        experiment.model = object()
        original = np.arange(12, dtype=np.float32).reshape(6, 2)
        experiment._get_data = lambda flag: (
            SimpleNamespace(data_x=original.copy()), None
        )
        captured = []

        class _Smoother:
            def __call__(self, values):
                return values + 10.0, values + 20.0

        def _fake_acf(values, **kwargs):
            captured.append(np.asarray(values).copy())
            return np.ones(int(kwargs["nlags"]) + 1, dtype=np.float64)

        with patch.object(
            autocon_experiment, "series_decomp", return_value=_Smoother()
        ), patch.object(autocon_experiment, "acf", side_effect=_fake_acf):
            experiment.init_AutoCon(args)

        self.assertEqual(len(captured), 2)
        np.testing.assert_array_equal(captured[0], original[:, 0])

    def test_shared_source_resolver_prefers_lowercase_for_trainer_namespaces(self):
        args = SimpleNamespace(
            autocon_acf_source="raw",
            AutoCon_acf_source="residual",
        )
        self.assertEqual(resolve_acf_source(args), "raw")
        self.assertEqual(
            resolve_acf_source(SimpleNamespace(AutoCon_acf_source="trend")),
            "trend",
        )

    def _experiment(self, model):
        experiment = Exp_Long_Term_Forecast_with_AutoCon.__new__(
            Exp_Long_Term_Forecast_with_AutoCon
        )
        experiment.model = model

        def fail_if_data_requested(*_args, **_kwargs):
            raise AssertionError("sidecar initialization must not read train data")

        experiment._get_data = fail_if_data_requested
        return experiment

    def test_init_autocon_loads_sidecar_before_relation_loss(self):
        full_values = np.zeros((2, 12), dtype=np.float32)
        full_values[:, 0] = 1.0
        full_values[0, 3] = 0.4
        full_values[0, 7] = -0.2
        full_values[1, 3] = 0.3
        full_values[1, 7] = -0.1
        model = _SidecarModel(full_values)

        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "acf_prior.npz"
            artifact.write_bytes(b"sidecar-placeholder")
            args = _args(artifact)
            experiment = self._experiment(model)
            loss = experiment.init_AutoCon(args)

        self.assertEqual(model.loaded, [str(artifact)])
        self.assertTrue(experiment._acf_prior_attached)
        self.assertIsInstance(loss, ACFRelationLoss)
        torch.testing.assert_close(loss.acf_values, torch.from_numpy(full_values))
        self.assertEqual(args.mamba_acf_effective_lags, [3, 7])
        self.assertEqual(args.mamba_acf_prior_max_lag, 11)

    def test_sidecar_hash_is_checked_before_loading(self):
        model = _SidecarModel(np.ones((2, 8), dtype=np.float32))
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "acf_prior.npz"
            artifact.write_bytes(b"sidecar-placeholder")
            args = _args(artifact)
            args.mamba_acf_prior_expected_sha256 = hashlib.sha256(
                b"different"
            ).hexdigest()
            experiment = self._experiment(model)
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                experiment.init_AutoCon(args)
        self.assertEqual(model.loaded, [])

    def test_empty_sidecar_path_is_treated_as_unconfigured(self):
        model = _SidecarModel(np.ones((2, 8), dtype=np.float32))
        experiment = self._experiment(model)
        args = _args("")
        args.mamba_acf_prior_artifact_path = "  "
        self.assertIsNone(
            experiment._load_mamba_sidecar_relation_loss(args, model)
        )
        self.assertEqual(model.loaded, [])

    def test_compact_only_sidecar_is_rejected_for_relation_objective(self):
        model = _SidecarModel(full_values=None)
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "acf_prior.npz"
            artifact.write_bytes(b"sidecar-placeholder")
            experiment = self._experiment(model)
            with self.assertRaisesRegex(RuntimeError, "complete train-only ACF"):
                experiment.init_AutoCon(_args(artifact))

    def test_relation_min_lag_beyond_sidecar_range_is_rejected_without_mask(self):
        full_values = np.zeros((2, 12), dtype=np.float32)
        full_values[:, 0] = 1.0
        model = _SidecarModel(full_values)
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "acf_prior.npz"
            artifact.write_bytes(b"sidecar-placeholder")
            args = _args(artifact)
            args.mamba_acf_relation_lag_mask = False
            args.mamba_acf_relation_min_lag = 12
            experiment = self._experiment(model)
            with self.assertRaisesRegex(ValueError, "exceeds the measured ACF range"):
                experiment.init_AutoCon(args)

    def test_plain_mamba_sidecar_preserves_complete_prior_max_lag(self):
        full_values = np.zeros((2, 12), dtype=np.float32)
        full_values[:, 0] = 1.0
        model = _SidecarModel(full_values)
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "acf_prior.npz"
            artifact.write_bytes(b"sidecar-placeholder")
            experiment = type("MambaExperiment", (), {})()
            experiment.model = model
            experiment.args = SimpleNamespace(
                mamba_acf_prior_artifact_path=str(artifact),
                mamba_acf_prior_expected_sha256="",
            )
            _MambaExperimentMixin._attach_train_acf_prior(experiment)
        self.assertEqual(experiment.args.mamba_acf_prior_max_lag, 11)
        self.assertTrue(experiment._acf_prior_attached)

    def test_plain_acf_off_evaluation_does_not_refit_train_prior(self):
        """An inference-only ACF-off replay must not require the train CSV."""

        model = _SidecarModel(np.ones((2, 8), dtype=np.float32))
        experiment = type("MambaExperiment", (), {})()
        experiment.model = model
        experiment.args = SimpleNamespace(
            model="MambaACF",
            is_training=False,
            AutoCon=False,
            mamba_use_acf_memory=False,
            mamba_acf_relation_loss=False,
            mamba_acf_anchor_copy=False,
            mamba_acf_channel_gate=False,
            mamba_acf_channel_conditioning=False,
            mamba_acf_scale_conditioning=False,
            mamba_acf_repr_fusion=False,
            mamba_acf_channel_order="input",
        )

        def fail_if_data_requested(*_args, **_kwargs):
            raise AssertionError("ACF-off evaluation must not read train data")

        experiment._get_data = fail_if_data_requested
        _MambaExperimentMixin._attach_train_acf_prior(experiment)

        self.assertEqual(model.loaded, [])
        self.assertFalse(hasattr(model, "acf"))

    def test_plain_acf_off_evaluation_ignores_explicit_sidecar(self):
        model = _SidecarModel(np.ones((2, 8), dtype=np.float32))
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "acf_prior.npz"
            artifact.write_bytes(b"sidecar-placeholder")
            experiment = type("MambaExperiment", (), {})()
            experiment.model = model
            experiment.args = SimpleNamespace(
                model="MambaACF",
                is_training=0,
                mamba_acf_prior_artifact_path=str(artifact),
                mamba_acf_prior_expected_sha256="",
                mamba_use_acf_memory=False,
                mamba_dynamic_acf=False,
                mamba_acf_relation_loss=False,
            )
            experiment._get_data = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("ACF-off evaluation must not read train data")
            )
            _MambaExperimentMixin._attach_train_acf_prior(experiment)
        self.assertEqual(model.loaded, [])
        self.assertFalse(getattr(experiment, "_acf_prior_attached", False))


if __name__ == "__main__":
    unittest.main()
