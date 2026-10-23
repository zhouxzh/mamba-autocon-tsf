"""Regression tests for model aliases used by result/report collectors."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.analyze_screening_matrix import (
    _resource_record_rows,
    canonical_model as screening_canonical_model,
    discover_metric_rows,
    summarize_seed_rows,
)
from scripts.build_final_report import canonical_model as report_canonical_model
from scripts.collect_extension_results import _model_name
from scripts.collect_results import (
    canonical_model as paper_canonical_model,
    load_trained,
)
from scripts.summarize_channel_metrics import _model_name as channel_model_name
from scripts.benchmark_model_resources import _model_module, _normalise_config
from utils.model_identity import canonical_model


class ModelIdentityTests(unittest.TestCase):
    def test_variable_alias_requires_runner_provenance(self):
        variable = {
            "model": "MambaACF",
            "AutoCon": False,
            "mamba_variable_mixer": True,
            "plan_job_id": "screen_ETTh1_MambaACFVariable_h96_s1",
        }
        self.assertEqual(canonical_model(variable), "MambaACFVariable")
        self.assertEqual(canonical_model({**variable, "plan_job_id": "pilot_MambaACF_h96_s1"}), "MambaACF")
        self.assertEqual(canonical_model({**variable, "mamba_variable_mixer": False}), "MambaACF")
        self.assertEqual(
            canonical_model({**variable, "plan_job_id": "screen_ETTh1_MambaACFVariableAutoCon_h96_s1"}),
            "MambaACF",
        )

    def test_variable_autocon_alias_requires_matching_executable_and_provenance(self):
        variable = {
            "model": "MambaACFAutoCon",
            "AutoCon": True,
            "mamba_variable_mixer": True,
            "plan_job_id": "screen_ETTh1_MambaACFVariableAutoCon_h96_s1",
        }
        self.assertEqual(canonical_model(variable), "MambaACFVariableAutoCon")
        self.assertEqual(canonical_model({**variable, "AutoCon": False}), "MambaACFAutoCon")
        self.assertEqual(canonical_model({**variable, "model": "MambaACF"}), "MambaACFAutoCon")

    def test_lite_autocon_alias_requires_matching_provenance(self):
        lite = {
            "model": "MambaACFAutoCon",
            "AutoCon": True,
            "mamba_scales": [1],
            "mamba_low_rank_head": True,
            "plan_job_id": "screen_ETTh1_MambaACFLiteAutoCon_h96_s1",
        }
        self.assertEqual(canonical_model(lite), "MambaACFLiteAutoCon")
        self.assertEqual(
            canonical_model({**lite, "plan_job_id": "screen_ETTh1_MambaACFAutoCon_h96_s1"}),
            "MambaACFAutoCon",
        )

    def test_all_collectors_use_the_same_alias_rule(self):
        config = {
            "model": "MambaACF",
            "AutoCon": False,
            "mamba_variable_mixer": True,
            "plan_job_id": "screen_traffic_MambaACFVariable_h96_s2",
        }
        for resolver in (
            screening_canonical_model,
            paper_canonical_model,
            report_canonical_model,
            _model_name,
            channel_model_name,
        ):
            self.assertEqual(resolver(config), "MambaACFVariable")

    def test_resource_aliases_keep_public_name_but_use_mamba_acf_module(self):
        common = {
            "seq_len": 96,
            "label_len": 48,
            "pred_len": 96,
            "enc_in": 7,
            "dec_in": 7,
            "c_out": 7,
            "mamba_variable_mixer": True,
        }
        plain = _normalise_config({**common, "model": "MambaACFVariable"})
        self.assertEqual(plain["model"], "MambaACFVariable")
        self.assertEqual(_model_module(plain), "MambaACF")

        relation = _normalise_config({
            **common,
            "model": "MambaACFVariableAutoCon",
            "AutoCon": True,
            "mamba_acf_relation_loss": True,
        })
        self.assertEqual(relation["model"], "MambaACFVariableAutoCon")
        self.assertEqual(_model_module(relation), "MambaACF")

        lite = _normalise_config({
            **common,
            "model": "MambaACFLiteAutoCon",
            "AutoCon": True,
            "mamba_scales": [1],
            "mamba_low_rank_head": True,
            "mamba_temporal_rank": 8,
            "mamba_acf_relation_loss": True,
        })
        self.assertEqual(lite["model"], "MambaACFLiteAutoCon")
        self.assertEqual(lite["mamba_scales"], [1])
        self.assertTrue(lite["mamba_causal_embedding"])
        self.assertTrue(lite["mamba_low_rank_head"])
        self.assertEqual(lite["mamba_temporal_rank"], 8)
        self.assertFalse(lite["mamba_variable_mixer"])
        self.assertFalse(lite["mamba_patch_mixer"])
        self.assertFalse(lite["mamba_local_mixer"])
        self.assertFalse(lite["mamba_use_acf_memory"])
        self.assertTrue(lite["mamba_acf_relation_loss"])
        self.assertEqual(_model_module(lite), "MambaACF")

        plan_lite = _normalise_config({
            **common,
            "model": "MambaACFAutoCon",
            "AutoCon": True,
            "plan_job_id": "screen_ETTh1_MambaACFLiteAutoCon_h96_s1",
            "mamba_scales": [2],
            "mamba_causal_embedding": False,
            "mamba_low_rank_head": False,
            "mamba_temporal_rank": 16,
            "mamba_acf_relation_loss": False,
        })
        self.assertEqual(plan_lite["model"], "MambaACFAutoCon")
        self.assertEqual(plan_lite["mamba_scales"], [1])
        self.assertTrue(plan_lite["mamba_causal_embedding"])
        self.assertTrue(plan_lite["mamba_low_rank_head"])
        self.assertEqual(plan_lite["mamba_temporal_rank"], 8)
        self.assertFalse(plan_lite["mamba_variable_mixer"])
        self.assertFalse(plan_lite["mamba_patch_mixer"])
        self.assertFalse(plan_lite["mamba_local_mixer"])
        self.assertFalse(plan_lite["mamba_use_acf_memory"])
        self.assertTrue(plan_lite["mamba_acf_relation_loss"])
        self.assertEqual(_model_module(plan_lite), "MambaACF")

    def test_result_collector_reads_canonical_acf_source_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            result_dir = Path(directory) / "run"
            result_dir.mkdir()
            (result_dir / "config.json").write_text(json.dumps({
                "data": "toy",
                "model": "MambaACFAutoCon",
                "AutoCon": True,
                "autocon_acf_source": " RAW ",
                "AutoCon_acf_source": "trend",
                "features": "M",
                "seq_len": 12,
                "pred_len": 4,
                "run_seed": 7,
                "enc_in": 2,
                "dec_in": 2,
                "c_out": 2,
                "AutoCon_lambda": 0.03,
            }), encoding="utf-8")
            np.save(result_dir / "metrics.npy", np.arange(5, dtype=np.float32))
            rows = load_trained(Path(directory))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["acf_source"], "raw")

    def test_screening_keeps_plain_and_variable_arms_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plain = root / "plain"
            variable = root / "variable"
            plain.mkdir()
            variable.mkdir()
            base = {
                "data": "toy",
                "features": "M",
                "seq_len": 12,
                "pred_len": 4,
                "run_seed": 7,
                "train_epochs": 20,
            }
            (plain / "config.json").write_text(json.dumps({
                **base,
                "model": "MambaACF",
                "AutoCon": False,
                "mamba_variable_mixer": False,
                "plan_job_id": "plain_toy_MambaACF_h4_s7",
            }), encoding="utf-8")
            (variable / "config.json").write_text(json.dumps({
                **base,
                "model": "MambaACF",
                "AutoCon": False,
                "mamba_variable_mixer": True,
                "plan_job_id": "variable_toy_MambaACFVariable_h4_s7",
            }), encoding="utf-8")
            (plain / "stream_metrics.json").write_text(json.dumps({"scaled": {"mse": 1.0}}), encoding="utf-8")
            (variable / "stream_metrics.json").write_text(json.dumps({"scaled": {"mse": 0.8}}), encoding="utf-8")
            rows, issues = discover_metric_rows([root])
            self.assertEqual(issues, [])
            seed_rows = summarize_seed_rows(rows)
            self.assertEqual({row["model"] for row in seed_rows}, {"MambaACF", "MambaACFVariable"})
            self.assertFalse(any(row["duplicate_conflict"] for row in seed_rows))

    def test_resource_source_job_id_restores_variable_alias(self):
        payload = {
            "benchmark_protocol": {"device": "cuda:0", "benchmark_batch_size": 1, "timed_forwards": 2},
            "resource_records": [{
                "model": "MambaACF",
                "dataset": "toy",
                "features": "M",
                "seq_len": 12,
                "pred_len": 4,
                "source": {"job_id": "variable_toy_MambaACFVariable_h4_s7"},
                "architecture": {
                    "model": "MambaACF",
                    "AutoCon": False,
                    "mamba_variable_mixer": True,
                },
                "parameters": {"total": 10, "trainable": 10},
                "gpu_inference": {
                    "latency_ms_mean": 1.0,
                    "latency_ms_median": 1.0,
                    "latency_ms_p95": 1.0,
                    "steady_state_peak_allocated_mb": 1.0,
                },
            }],
        }
        rows, issue = _resource_record_rows(payload, Path("resource.json"))
        self.assertIsNone(issue)
        self.assertEqual(rows[0]["model"], "MambaACFVariable")


if __name__ == "__main__":
    unittest.main()
