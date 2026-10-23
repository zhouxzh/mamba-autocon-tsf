"""Contracts for relation-target semantics and temporal batch sampling."""

import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from torch.utils.data import RandomSampler

from data_provider import data_factory
from data_provider.samplers import ACFLagPairBatchSampler
from run import persist_run_config


class _MemoryDataset:
    """Small dataset stub accepted by ``data_provider`` without CSV I/O."""

    def __init__(self, **_kwargs):
        self.size = 17

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return index, None, None, None, None


def _args(**overrides):
    values = {
        "data": "_contract",
        "embed": "timeF",
        "task_name": "long_term_forecast",
        "model": "MambaACFAutoCon",
        "root_path": ".",
        "data_path": "unused.csv",
        "seq_len": 8,
        "label_len": 4,
        "pred_len": 4,
        "features": "M",
        "target": "OT",
        "freq": "h",
        "seasonal_patterns": "Monthly",
        "train_ratio": 0.6,
        "batch_size": 4,
        "num_workers": 0,
        "drop_last_train": False,
        "mamba_acf_relation_loss": True,
        "mamba_acf_relation_batching": "lag_pairs",
        "mamba_acf_relation_target_mode": "hard_selected",
        "mamba_acf_relation_lag_mask": False,
        "mamba_acf_relation_selected_lag_boost": 0.0,
        "mamba_acf_relation_effective_lags": [3, 7],
        "run_seed": 2021,
        "plan_job_id": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class ACFRelationBatchingContractTests(unittest.TestCase):
    def test_hard_selected_requires_lag_mask(self):
        decision = data_factory.resolve_acf_relation_batching(_args())
        self.assertFalse(decision["use_lag_pairs"])
        self.assertEqual(decision["effective"], "shuffle")
        self.assertEqual(decision["reason"], "all_candidate_targets")
        self.assertFalse(decision["selected_lag_support"])

        masked = data_factory.resolve_acf_relation_batching(
            _args(mamba_acf_relation_lag_mask=True)
        )
        self.assertTrue(masked["use_lag_pairs"])
        self.assertEqual(masked["effective"], "lag_pairs")
        self.assertEqual(masked["reason"], "hard_selected_lag_mask")
        self.assertTrue(masked["selected_lag_support"])

    def test_soft_targets_need_an_explicit_selected_lag_boost(self):
        soft = data_factory.resolve_acf_relation_batching(
            _args(mamba_acf_relation_target_mode="soft_multi_peak")
        )
        self.assertFalse(soft["use_lag_pairs"])
        self.assertEqual(soft["reason"], "all_candidate_targets")

        boosted = data_factory.resolve_acf_relation_batching(
            _args(
                mamba_acf_relation_target_mode="soft_multi_peak",
                mamba_acf_relation_selected_lag_boost=0.5,
            )
        )
        self.assertTrue(boosted["use_lag_pairs"])
        self.assertEqual(boosted["reason"], "soft_selected_lag_boost")

        full_boosted = data_factory.resolve_acf_relation_batching(
            _args(
                mamba_acf_relation_target_mode="soft_full",
                mamba_acf_relation_selected_lag_boost=1.0,
            )
        )
        self.assertTrue(full_boosted["use_lag_pairs"])

    def test_random_request_and_inactive_objective_never_enable_pairs(self):
        random_request = data_factory.resolve_acf_relation_batching(
            _args(mamba_acf_relation_lag_mask=True, mamba_acf_relation_batching="random")
        )
        self.assertFalse(random_request["use_lag_pairs"])
        self.assertEqual(random_request["reason"], "requested_random_batching")
        # Semantic support is still present; the caller explicitly selected a
        # different sampler policy.
        self.assertTrue(random_request["selected_lag_support"])

        disabled = data_factory.resolve_acf_relation_batching(
            _args(mamba_acf_relation_loss=False, mamba_acf_relation_lag_mask=True)
        )
        self.assertFalse(disabled["use_lag_pairs"])
        self.assertEqual(disabled["reason"], "relation_loss_disabled")

    def test_local_scope_does_not_enable_global_lag_pairs(self):
        decision = data_factory.resolve_acf_relation_batching(
            _args(
                mamba_acf_relation_lag_mask=True,
                mamba_acf_relation_scope="local",
            )
        )
        self.assertFalse(decision["use_lag_pairs"])
        self.assertEqual(decision["effective"], "shuffle")
        self.assertEqual(decision["reason"], "local_scope_no_global_pairs")
        self.assertTrue(decision["selected_lag_support"])

    def test_string_false_flags_are_parsed_as_false(self):
        decision = data_factory.resolve_acf_relation_batching(
            _args(
                mamba_acf_relation_loss="false",
                mamba_acf_relation_lag_mask="true",
            )
        )
        self.assertFalse(decision["use_lag_pairs"])
        self.assertEqual(decision["reason"], "relation_loss_disabled")

    def test_data_provider_uses_shuffle_for_all_candidate_targets(self):
        args = _args()
        with patch.dict(data_factory.data_dict, {"_contract": _MemoryDataset}, clear=False):
            _dataset, loader = data_factory.data_provider(args, "train")
        self.assertIsInstance(loader.sampler, RandomSampler)
        self.assertNotIsInstance(loader.batch_sampler, ACFLagPairBatchSampler)
        self.assertEqual(args.mamba_acf_relation_batching_requested, "lag_pairs")
        self.assertEqual(args.mamba_acf_relation_batching_effective, "shuffle")
        self.assertEqual(args.mamba_acf_relation_batching_reason, "all_candidate_targets")
        self.assertFalse(args.mamba_acf_relation_selected_lag_support)

    def test_data_provider_uses_pairs_when_selected_lags_are_supervised(self):
        args = _args(mamba_acf_relation_lag_mask=True)
        with patch.dict(data_factory.data_dict, {"_contract": _MemoryDataset}, clear=False):
            _dataset, loader = data_factory.data_provider(args, "train")
        self.assertIsInstance(loader.batch_sampler, ACFLagPairBatchSampler)
        self.assertEqual(args.mamba_acf_relation_batching_effective, "lag_pairs")
        self.assertEqual(args.mamba_acf_relation_batching_reason, "hard_selected_lag_mask")
        self.assertTrue(args.mamba_acf_relation_selected_lag_support)

    def test_config_persistence_attests_the_resolved_sampler_decision(self):
        args = _args()
        with tempfile.TemporaryDirectory() as directory:
            persist_run_config(directory, args, 2021, "contract")
            payload = json.loads((Path(directory) / "config.json").read_text())
        self.assertEqual(payload["mamba_acf_relation_batching_requested"], "lag_pairs")
        self.assertEqual(payload["mamba_acf_relation_batching_effective"], "shuffle")
        self.assertEqual(payload["mamba_acf_relation_batching_reason"], "all_candidate_targets")
        self.assertFalse(payload["mamba_acf_relation_selected_lag_support"])


if __name__ == "__main__":
    unittest.main()
