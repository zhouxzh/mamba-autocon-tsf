"""Regression tests for persisted MambaACF relation sampler semantics."""

from __future__ import annotations

import unittest

from scripts.validate_training_phase import _validate_relation_contract


def _base_config(**overrides):
    config = {
        "model": "MambaACFAutoCon",
        "mamba_acf_relation_loss": True,
        "mamba_acf_relation_batching": "lag_pairs",
        "mamba_acf_relation_target_mode": "soft_multi_peak",
        "mamba_acf_relation_lag_mask": False,
        "mamba_acf_relation_selected_lag_boost": 0.0,
        "mamba_acf_relation_scope": "global",
        "mamba_acf_relation_effective_lags": [96],
        "mamba_acf_relation_selected_lags": [96],
        "mamba_acf_train_peak_lags": [96],
        "mamba_acf_prior_max_lag": 720,
        "mamba_acf_relation_min_lag": 96,
        "mamba_acf_lag_mode": "train_peaks_available",
        "mamba_acf_relation_lag_source": "train_peak_subset",
        "mamba_acf_relation_lag_fallback": False,
        "mamba_acf_relation_batching_requested": "lag_pairs",
        "mamba_acf_relation_batching_effective": "shuffle",
        "mamba_acf_relation_batching_reason": "all_candidate_targets",
        "mamba_acf_relation_selected_lag_support": False,
    }
    config.update(overrides)
    return config


def _base_logs(**overrides):
    logs = {
        "val_loss": [1.0, 0.9],
        "acf_relation_effective_lags": [96],
        "acf_relation_selected_lags": [96],
        "acf_relation_train_peak_lags": [96],
        "acf_relation_lag_source": "train_peak_subset",
        "acf_relation_lag_fallback": False,
        "acf_relation_global_valid_fraction": [0.2, 0.2],
        "acf_relation_global_active_anchor_fraction": [0.8, 0.8],
        "acf_relation_global_candidate_fraction": [0.3, 0.3],
        "acf_relation_global_candidate_active_anchor_fraction": [0.9, 0.9],
        "acf_relation_global_positive_count_mean": [1.2, 1.1],
        "acf_relation_global_high_acf_excluded_fraction": [0.0, 0.1],
        "acf_relation_batching_requested": "lag_pairs",
        "acf_relation_batching_effective": "shuffle",
        "acf_relation_batching_reason": "all_candidate_targets",
        "acf_relation_selected_lag_support": False,
        "acf_relation_lag_pair_counts": [],
        "acf_relation_lag_pair_exposure": [],
    }
    logs.update(overrides)
    return logs


class RelationValidatorContractTests(unittest.TestCase):
    def test_soft_all_candidate_targets_require_shuffle_without_pairs(self):
        config = _base_config()
        logs = _base_logs()
        self.assertEqual(_validate_relation_contract(config, logs, strict=True), [])

        logs.update({
            "acf_relation_batching_effective": "lag_pairs",
            "acf_relation_batching_reason": "hard_selected_lag_mask",
            "acf_relation_selected_lag_support": True,
            "acf_relation_lag_pair_counts": [{"96": 3}, {"96": 2}],
            "acf_relation_lag_pair_exposure": [{"96": 1.0}, {"96": 1.0}],
        })
        errors = _validate_relation_contract(config, logs, strict=True)
        self.assertIn("acf_relation_batching_effective_mismatch", errors)
        self.assertIn("acf_relation_lag_pair_counts_nonzero_for_shuffle", errors)

    def test_selected_lag_targets_require_pair_telemetry(self):
        config = _base_config(
            mamba_acf_relation_target_mode="hard_selected",
            mamba_acf_relation_lag_mask=True,
            mamba_acf_relation_batching_effective="lag_pairs",
            mamba_acf_relation_batching_reason="hard_selected_lag_mask",
            mamba_acf_relation_selected_lag_support=True,
        )
        logs = _base_logs(
            acf_relation_batching_effective="lag_pairs",
            acf_relation_batching_reason="hard_selected_lag_mask",
            acf_relation_selected_lag_support=True,
            acf_relation_lag_pair_counts=[{"96": 3}, {"96": 2}],
            acf_relation_lag_pair_exposure=[{"96": 1.0}, {"96": 1.0}],
        )
        self.assertEqual(_validate_relation_contract(config, logs, strict=True), [])

        logs["acf_relation_lag_pair_counts"] = []
        logs["acf_relation_lag_pair_exposure"] = []
        errors = _validate_relation_contract(config, logs, strict=True)
        self.assertIn(
            "acf_relation_lag_pair_counts_missing_or_empty_for_lag_pairs",
            errors,
        )

    def test_disabled_relation_accepts_blank_metadata_but_not_pair_claims(self):
        config = _base_config(mamba_acf_relation_loss=False)
        for field in (
            "mamba_acf_relation_batching_requested",
            "mamba_acf_relation_batching_effective",
            "mamba_acf_relation_batching_reason",
            "mamba_acf_relation_selected_lag_support",
        ):
            config.pop(field)
        logs = {
            "acf_relation_lag_pair_counts": [],
            "acf_relation_lag_pair_exposure": [],
        }
        self.assertEqual(_validate_relation_contract(config, logs, strict=True), [])

        logs["acf_relation_lag_pair_counts"] = [{"96": 1}]
        errors = _validate_relation_contract(config, logs, strict=True)
        self.assertIn(
            "acf_relation_lag_pair_counts_nonzero_for_shuffle",
            errors,
        )


if __name__ == "__main__":
    unittest.main()
