"""Contract tests for the exploratory MambaACF low-rank-head screen."""

import json
import unittest
from pathlib import Path

import torch

from scripts import analyze_mamba_acf_low_rank_head_screen as analyzer


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = REPO_ROOT / "docs/research/mamba_acf_low_rank_head_screen_plan.json"


class MambaACFLowRankHeadScreenContractTests(unittest.TestCase):
    def test_plan_registers_only_the_two_paired_arms(self):
        plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            plan["status"], "exploratory_acf_low_rank_head_screen_not_confirmatory"
        )
        self.assertEqual(
            [phase["name"] for phase in plan["phases"]],
            ["low_rank_head_full_raw_h96", "low_rank_head_r16_raw_h96"],
        )
        self.assertEqual(plan["registered_job_counts"]["exploratory_total"], 12)
        self.assertEqual(plan["registered_job_counts"]["total"], 12)
        self.assertFalse(plan["phases"][0]["mamba_low_rank_head"])
        self.assertTrue(plan["phases"][1]["mamba_low_rank_head"])
        self.assertEqual(plan["phases"][1]["mamba_temporal_rank"], 16)
        for phase in plan["phases"]:
            self.assertEqual(phase["datasets"], ["ETTh1", "traffic"])
            self.assertEqual(phase["seeds"], [2021, 2022, 2023])
            self.assertEqual(phase["horizons"], [96])
            self.assertEqual(phase["mamba_variable_group_size"], 1)
            self.assertTrue(phase["mamba_use_acf_memory"])
            self.assertTrue(phase["mamba_acf_relation_loss"])

    def test_checkpoint_contract_distinguishes_full_and_factorized_heads(self):
        full_state = {
            "temporal_head.weight": torch.zeros(96, 96),
            "linear_residual.weight": torch.zeros(96, 96),
            "temporal_head.bias": torch.zeros(96),
            "linear_residual.bias": torch.zeros(96),
        }
        full = analyzer._head_contract(full_state, Path("full.pth"), False)
        self.assertFalse(full["factorized"])
        self.assertIsNone(full["rank"])
        self.assertEqual(full["parameter_count"], 2 * (96 * 96 + 96))

        low_rank_state = {
            "temporal_head.input_projection.weight": torch.zeros(16, 96),
            "temporal_head.output_projection.weight": torch.zeros(96, 16),
            "temporal_head.output_projection.bias": torch.zeros(96),
            "linear_residual.input_projection.weight": torch.zeros(16, 96),
            "linear_residual.output_projection.weight": torch.zeros(96, 16),
            "linear_residual.output_projection.bias": torch.zeros(96),
        }
        low_rank = analyzer._head_contract(
            low_rank_state, Path("rank16.pth"), True
        )
        self.assertTrue(low_rank["factorized"])
        self.assertEqual(low_rank["rank"], 16)
        self.assertEqual(low_rank["parameter_count"], 2 * (16 * 96 + 96 * 16 + 96))

    def test_paired_contrast_uses_rank16_minus_full(self):
        pairs = [
            {
                "full_mse": 1.0,
                "rank16_mse": 0.9,
                "full_mae": 0.5,
                "rank16_mae": 0.49,
                "full_elapsed_sec": 10.0,
                "rank16_elapsed_sec": 8.0,
            },
            {
                "full_mse": 1.1,
                "rank16_mse": 1.0,
                "full_mae": 0.6,
                "rank16_mae": 0.59,
                "full_elapsed_sec": 11.0,
                "rank16_elapsed_sec": 9.0,
            },
        ]
        result = analyzer._contrast("elapsed_sec", pairs)
        self.assertEqual(result["treatment"], "rank16")
        self.assertEqual(result["reference"], "full")
        self.assertAlmostEqual(result["mean"], -2.0)
        self.assertEqual(result["wins_lower"], 2)

    def test_config_requires_explicit_boolean_fields(self):
        config = {
            "model": "MambaACFAutoCon",
            "features": "M",
            "pred_len": 96,
            "AutoCon": True,
            "AutoCon_lambda": 0.03,
            "AutoCon_acf_source": "raw",
            "mamba_variable_mixer": True,
            "mamba_variable_group_size": 1,
            "mamba_use_acf_memory": True,
            "mamba_acf_relation_loss": True,
            "mamba_acf_relation_lag_mask": True,
            "mamba_acf_relation_scope": "global",
            "mamba_acf_relation_min_lag": 96,
            "mamba_acf_relation_target_mode": "hard_selected",
            "mamba_token_stride": 1,
            "mamba_scales": [1],
            "mamba_channel_gate": False,
            "mamba_acf_channel_gate": False,
            "mamba_low_rank_head": False,
            "mamba_temporal_rank": 16,
        }
        missing = dict(config)
        missing.pop("mamba_low_rank_head")
        with self.assertRaisesRegex(ValueError, "config mamba_low_rank_head is missing"):
            analyzer._check_config(missing, False, "job")
        with self.assertRaisesRegex(ValueError, "explicit boolean"):
            invalid = dict(config)
            invalid["mamba_low_rank_head"] = "false"
            analyzer._check_config(invalid, False, "job")


if __name__ == "__main__":
    unittest.main()
