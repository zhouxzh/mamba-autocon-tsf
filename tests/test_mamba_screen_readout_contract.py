"""Contract tests for the exploratory MambaACF screen readout."""

import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts/analyze_mamba_variable_memory_relation_screen.py"
SPEC = importlib.util.spec_from_file_location("mamba_screen_readout", MODULE_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import contract
    raise RuntimeError("could not load the MambaACF screen analyzer")
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


def _payload(status):
    coverage = {
        "objective_version": 3,
        "acf_relation_global_valid_fraction": {"min": 0.1},
        "acf_relation_global_candidate_fraction": {"min": 0.2},
        "acf_relation_global_candidate_active_anchor_fraction": {"min": 0.3},
        "acf_relation_global_positive_count_mean": {"min": 0.4},
        "effective_lags": [96],
    }
    row = {
        "kind": "memory_relation",
        "dataset": "ETTh1",
        "seed": 2021,
        "coverage": coverage,
    }
    pair = {
        "dataset": "ETTh1",
        "seed": 2021,
        "plain_mse": 1.0,
        "memory_mse": 0.9,
        "relation_mse": 0.8,
        "delta_memory_minus_plain_mse": -0.1,
        "delta_relation_minus_memory_mse": -0.1,
        "plain_mae": 1.0,
        "memory_mae": 0.9,
        "relation_mae": 0.8,
        "delta_memory_minus_plain_mae": -0.1,
        "delta_relation_minus_memory_mae": -0.1,
    }
    summary = {
        "dataset": "ETTh1",
        "label": "screen",
        "mse": {
            "n": 1,
            "reference_mean": 1.0,
            "treatment_mean": 0.9,
            "delta_mean": -0.1,
            "delta_sd": 0.0,
            "wins_lower": 1,
            "relative_change_pct": -10.0,
            "paired_t": 0.0,
            "paired_t_two_sided_p": None,
        },
        "mae": {
            "n": 1,
            "reference_mean": 1.0,
            "treatment_mean": 0.9,
            "delta_mean": -0.1,
            "delta_sd": 0.0,
            "wins_lower": 1,
            "relative_change_pct": -10.0,
            "paired_t": 0.0,
            "paired_t_two_sided_p": None,
        },
    }
    return {
        "plan": {"path": "plan.json", "sha256": "plan-hash", "status": status},
        "validation": [
            {
                "phase": "phase",
                "observed_jobs": 1,
                "expected_jobs": 1,
                "manifest_sha256": "manifest-hash",
                "provenance": {
                    "provenance_id": "provenance-id",
                    "source_tree_sha256": "source-hash",
                },
            }
        ],
        "screen_root": "screen-root",
        "job_count": 1,
        "per_seed": [pair],
        "contrasts": [summary],
        "rows": [row],
    }


class MambaScreenReadoutContractTests(unittest.TestCase):
    def test_old_and_current_exploratory_statuses_are_supported(self):
        self.assertIn(
            "exploratory_variable_memory_relation_screen_not_confirmatory",
            ANALYZER.SUPPORTED_PLAN_STATUSES,
        )
        self.assertIn(
            "exploratory_current_code_sidecar_replay_not_confirmatory",
            ANALYZER.SUPPORTED_PLAN_STATUSES,
        )
        self.assertNotIn("approved_for_launch", ANALYZER.SUPPORTED_PLAN_STATUSES)

    def test_sidecar_replay_annotation_is_status_gated(self):
        old_readout = ANALYZER._markdown(
            _payload("exploratory_variable_memory_relation_screen_not_confirmatory")
        )
        replay_readout = ANALYZER._markdown(
            _payload("exploratory_current_code_sidecar_replay_not_confirmatory")
        )
        self.assertNotIn("Replay contract: current-code", old_readout)
        self.assertIn("Artifact contract: the provenance-snapshot", replay_readout)
        self.assertIn("not an independent confirmatory reproduction", replay_readout)
        self.assertIn("no result here is confirmatory evidence", old_readout)
        self.assertIn("no result here is confirmatory evidence", replay_readout)

    def test_readout_surfaces_live_source_drift(self):
        payload = _payload(
            "exploratory_current_code_sidecar_replay_not_confirmatory"
        )
        payload["validation"][0]["provenance"]["live_source_mismatches"] = [
            "run.py",
            "exp/exp_long_term_forecasting_with_mamba.py",
        ]
        readout = ANALYZER._markdown(payload)
        self.assertIn("live working tree differs in 2 captured source file(s)", readout)
        self.assertIn("provenance-snapshot training result", readout)


if __name__ == "__main__":
    unittest.main()
