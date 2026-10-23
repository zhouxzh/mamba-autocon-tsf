import unittest

from scripts.benchmark_model_resources import (
    AUTOCON_IDENTIFIERS,
    MODEL_MODULES,
    REPO_ROOT,
    _activate_resource_acf_channel_order,
    _explicit_request,
    _flatten_records,
    _forward_outputs,
    _forward_prediction,
    _measure_request,
    _manifest_elapsed_summary,
    _normalise_config,
    _parser,
    _percentile,
    _representation_shape,
    _resource_signature,
    _setup_source_root,
    _task_prediction,
)


class ResourceBenchmarkTests(unittest.TestCase):
    def test_resource_registry_exposes_dcrnn(self):
        self.assertEqual(MODEL_MODULES["DCRNN"], "DCRNN")

    def test_resource_registry_exposes_timemachine_variants(self):
        self.assertEqual(MODEL_MODULES["MambaTM1"], "MambaTM1AutoCon")
        self.assertEqual(MODEL_MODULES["MambaTM2"], "MambaTM2AutoCon")
        self.assertIn("MambaTM1AutoCon", AUTOCON_IDENTIFIERS)
        self.assertIn("MambaTM2AutoCon", AUTOCON_IDENTIFIERS)

    def test_resource_registry_exposes_smamba_variants(self):
        self.assertEqual(MODEL_MODULES["MambaSM"], "MambaSMambaAutoCon")
        self.assertIn("MambaSMAutoCon", AUTOCON_IDENTIFIERS)

    def test_resource_registry_exposes_mambats_v6_variants(self):
        self.assertEqual(MODEL_MODULES["MambaTSv6"], "MambaTSv6")
        self.assertEqual(MODEL_MODULES["MambaTSv6AutoCon"], "MambaTSv6")
        self.assertIn("MambaTSv6AutoCon", AUTOCON_IDENTIFIERS)

    def test_resource_registry_exposes_mambats_v7_variants(self):
        self.assertEqual(MODEL_MODULES["MambaTSv7"], "MambaTSv7")
        self.assertEqual(MODEL_MODULES["MambaTSv7AutoCon"], "MambaTSv7")
        self.assertIn("MambaTSv7AutoCon", AUTOCON_IDENTIFIERS)

    def test_resource_registry_exposes_mambats_v8_variants(self):
        self.assertEqual(MODEL_MODULES["MambaTSv8"], "MambaTSv8")
        self.assertEqual(MODEL_MODULES["MambaTSv8AutoCon"], "MambaTSv8")
        self.assertIn("MambaTSv8AutoCon", AUTOCON_IDENTIFIERS)

    def test_resource_registry_exposes_mambats_v9_variants(self):
        self.assertEqual(MODEL_MODULES["MambaTSv9"], "MambaTSv9")
        self.assertEqual(MODEL_MODULES["MambaTSv9AutoCon"], "MambaTSv9")
        self.assertIn("MambaTSv9AutoCon", AUTOCON_IDENTIFIERS)

    def test_resource_registry_exposes_mambats_v10_variants(self):
        self.assertEqual(MODEL_MODULES["MambaTSv10"], "MambaTSv10")
        self.assertEqual(MODEL_MODULES["MambaTSv10AutoCon"], "MambaTSv10")
        self.assertIn("MambaTSv10AutoCon", AUTOCON_IDENTIFIERS)

    def test_resource_registry_exposes_mambats_v11_variants(self):
        self.assertEqual(MODEL_MODULES["MambaTSv11"], "MambaTSv11")
        self.assertEqual(MODEL_MODULES["MambaTSv11AutoCon"], "MambaTSv11")
        self.assertIn("MambaTSv11AutoCon", AUTOCON_IDENTIFIERS)

    def test_resource_registry_exposes_mambats_v12_variants(self):
        self.assertEqual(MODEL_MODULES["MambaTSv12"], "MambaTSv12")
        self.assertEqual(MODEL_MODULES["MambaTSv12AutoCon"], "MambaTSv12")
        self.assertIn("MambaTSv12AutoCon", AUTOCON_IDENTIFIERS)

    def test_resource_registry_exposes_mambats_v13_variants(self):
        self.assertEqual(MODEL_MODULES["MambaTSv13"], "MambaTSv13")
        self.assertEqual(MODEL_MODULES["MambaTSv13AutoCon"], "MambaTSv13")
        self.assertIn("MambaTSv13AutoCon", AUTOCON_IDENTIFIERS)

    def test_resource_registry_exposes_mambats_v14_variants(self):
        self.assertEqual(MODEL_MODULES["MambaTSv14"], "MambaTSv14")
        self.assertEqual(MODEL_MODULES["MambaTSv14AutoCon"], "MambaTSv14")
        self.assertIn("MambaTSv14AutoCon", AUTOCON_IDENTIFIERS)

    def test_resource_registry_exposes_mambats_v15_variants(self):
        self.assertEqual(MODEL_MODULES["MambaTSv15"], "MambaTSv15")
        self.assertEqual(MODEL_MODULES["MambaTSv15AutoCon"], "MambaTSv15")
        self.assertIn("MambaTSv15AutoCon", AUTOCON_IDENTIFIERS)

    def test_resource_registry_exposes_mambats_v16_variants(self):
        self.assertEqual(MODEL_MODULES["MambaTSv16"], "MambaTSv16")
        self.assertEqual(MODEL_MODULES["MambaTSv16AutoCon"], "MambaTSv16")
        self.assertIn("MambaTSv16AutoCon", AUTOCON_IDENTIFIERS)

    def test_explicit_resource_request_selects_v15_without_manual_variant(self):
        parser = _parser()
        args = parser.parse_args([
            "--model", "MambaTSv15AutoCon", "--dataset", "pems07",
            "--features", "M", "--freq", "t", "--seq-len", "96", "--label-len", "48",
            "--pred-len", "12", "--enc-in", "883", "--dec-in", "883", "--c-out", "883",
            "--output-dir", "artifacts/exploratory/test_resource_benchmark_v15",
        ])
        request = _explicit_request(args, REPO_ROOT)
        self.assertEqual(request["config"]["mamba_ts_variant"], "v15")

    def test_explicit_resource_request_selects_v14_without_manual_variant(self):
        parser = _parser()
        args = parser.parse_args([
            "--model", "MambaTSv14AutoCon", "--dataset", "pems07",
            "--features", "M", "--freq", "t", "--seq-len", "96",
            "--label-len", "48", "--pred-len", "12", "--enc-in", "883",
            "--dec-in", "883", "--c-out", "883",
            "--output-dir", "artifacts/exploratory/test_resource_benchmark_v14",
        ])
        request = _explicit_request(args, REPO_ROOT)
        self.assertEqual(request["config"]["mamba_ts_variant"], "v14")

    def test_explicit_resource_request_selects_v8(self):
        parser = _parser()
        args = parser.parse_args([
            "--model", "MambaTSv8AutoCon", "--dataset", "pems03",
            "--features", "M", "--freq", "t", "--seq-len", "96",
            "--label-len", "48", "--pred-len", "96", "--enc-in", "358",
            "--dec-in", "358", "--c-out", "358", "--mamba-ts-spectral-top-k", "3",
            "--output-dir", "artifacts/exploratory/test_resource_benchmark_v8",
        ])
        request = _explicit_request(args, REPO_ROOT)
        self.assertEqual(request["config"]["mamba_ts_variant"], "v8")
        self.assertEqual(request["config"]["mamba_ts_spectral_top_k"], 3)

    def test_timemachine_signature_includes_architecture_controls(self):
        common = {
            "model": "MambaTM2",
            "seq_len": 96, "label_len": 48, "pred_len": 96,
            "enc_in": 7, "dec_in": 7, "c_out": 7,
            "mamba_tm_n1": 64, "mamba_tm_n2": 128,
            "mamba_tm_repr_width": 64,
            "mamba_tm_channel_independent": False,
            "mamba_tm_residual": True,
        }
        changed_width = dict(common, mamba_tm_n2=96)
        changed_mode = dict(common, mamba_tm_channel_independent=True)
        self.assertNotEqual(_resource_signature(common), _resource_signature(changed_width))
        self.assertNotEqual(_resource_signature(common), _resource_signature(changed_mode))

    def _record(self, model="Mamba2AutoCon", lambda_value=0.3, seed=101, elapsed=2.0):
        return {
            "job_id": "job-{}-{}-{}".format(model, lambda_value, seed),
            "status": "ok",
            "model": model,
            "dataset": "ETTh1",
            "features": "S",
            "enc_in": 1,
            "dec_in": 1,
            "c_out": 1,
            "seed": seed,
            "autocon_lambda": lambda_value,
            "elapsed_sec": elapsed,
            "command": [
                "--seq_len", "96", "--pred_len", "96", "--batch_size", "32",
                "--train_epochs", "50", "--lradj", "cosine", "--d_model", "32",
                "--d_ff", "32", "--e_layers", "1", "--mamba_d_state", "16",
                "--mamba_d_conv", "4", "--mamba_expand", "2", "--mamba_headdim", "16",
                "--learning_rate", "0.001", "--min_learning_rate", "1e-5",
            ],
        }

    def test_manifest_groups_seeds_but_keeps_lambda_variants_separate(self):
        rows = [
            self._record(lambda_value=0.3, seed=101, elapsed=2.0),
            self._record(lambda_value=0.3, seed=102, elapsed=4.0),
            self._record(lambda_value=1.0, seed=101, elapsed=6.0),
        ]
        summary = _manifest_elapsed_summary(rows)
        self.assertEqual(len(summary), 2)
        self.assertEqual(sorted(row["job_count"] for row in summary), [1, 2])
        self.assertEqual(sorted(row["unique_seed_count"] for row in summary), [1, 2])

    def test_manifest_keeps_new_mamba_architectures_separate(self):
        first = self._record()
        second = self._record(seed=102)
        first["command"] += [
            "--mamba_scales", "1", "2", "--mamba_variable_mixer_depth", "1",
            "--mamba_use_acf_memory", "false",
        ]
        second["command"] += [
            "--mamba_scales", "1", "2", "--mamba_variable_mixer",
            "--mamba_variable_mixer_depth", "2", "--mamba_use_acf_memory", "false",
        ]
        summary = _manifest_elapsed_summary([first, second])
        self.assertEqual(len(summary), 2)
        self.assertEqual(sorted(row["mamba_variable_mixer_depth"] for row in summary), [1, 2])
        self.assertEqual(sorted(row["mamba_variable_mixer"] for row in summary), [False, True])
        self.assertEqual({row["mamba_use_acf_memory"] for row in summary}, {False})

    def test_manifest_keeps_token_stride_variants_separate(self):
        first = self._record(model="MambaACF1AutoCon", seed=101)
        second = self._record(model="MambaACF1AutoCon", seed=102)
        second["command"] += ["--mamba_token_stride", "8"]
        summary = _manifest_elapsed_summary([first, second])
        self.assertEqual(len(summary), 2)
        self.assertEqual(sorted(row["mamba_token_stride"] for row in summary), [1, 8])

    def test_manifest_keeps_relation_target_variants_separate(self):
        first = self._record(model="MambaACFAutoCon", seed=101)
        second = self._record(model="MambaACFAutoCon", seed=102)
        first["command"] += [
            "--mamba_acf_relation_target_mode", "hard_selected",
            "--mamba_acf_relation_soft_min_abs", "0.1",
            "--mamba_acf_relation_selected_lag_boost", "0.0",
        ]
        second["command"] += [
            "--mamba_acf_relation_target_mode", "soft_multi_peak",
            "--mamba_acf_relation_soft_min_abs", "0.25",
            "--mamba_acf_relation_selected_lag_boost", "0.5",
        ]
        summary = _manifest_elapsed_summary([first, second])
        self.assertEqual(len(summary), 2)
        self.assertEqual(
            {row["mamba_acf_relation_target_mode"] for row in summary},
            {"hard_selected", "soft_multi_peak"},
        )

    def test_manifest_keeps_acf_scale_conditioning_variants_separate(self):
        first = self._record(model="MambaACFAutoCon", seed=101)
        second = self._record(model="MambaACFAutoCon", seed=102)
        first["command"] += ["--mamba_scales", "1", "2"]
        second["command"] += [
            "--mamba_scales", "1", "2", "--mamba_acf_scale_conditioning"
        ]
        summary = _manifest_elapsed_summary([first, second])
        self.assertEqual(len(summary), 2)
        self.assertEqual(
            {row["mamba_acf_scale_conditioning"] for row in summary},
            {False, True},
        )

    def test_manifest_keeps_reliability_interpolation_variants_separate(self):
        first = self._record(model="MambaACF", seed=101)
        second = self._record(model="MambaACF", seed=102)
        first["command"] += [
            "--mamba_use_acf_memory", "true", "--AutoCon_acf_source", "raw",
            "--mamba_acf_reliability_interpolation", "false",
        ]
        second["command"] += [
            "--mamba_use_acf_memory", "true", "--AutoCon_acf_source", "raw",
            "--mamba_acf_reliability_interpolation", "true",
        ]
        summary = _manifest_elapsed_summary([first, second])
        self.assertEqual(len(summary), 2)
        self.assertEqual(
            {row["mamba_acf_reliability_interpolation"] for row in summary},
            {False, True},
        )

    def test_manifest_keeps_local_agreement_variants_and_support_separate(self):
        first = self._record(model="MambaACF", seed=101)
        second = self._record(model="MambaACF", seed=102)
        first["command"] += [
            "--mamba_acf_lags", "24", "--mamba_acf_local_min_pairs", "16",
        ]
        second["command"] += [
            "--mamba_acf_lags", "24", "--mamba_acf_local_agreement",
            "--mamba_acf_local_min_pairs", "32", "--mamba_use_acf_memory",
            "--AutoCon_acf_source", "raw",
        ]
        summary = _manifest_elapsed_summary([first, second])
        self.assertEqual(len(summary), 2)
        self.assertEqual(
            {(row["mamba_acf_local_agreement"], row["mamba_acf_local_min_pairs"])
             for row in summary},
            {(False, 16), (True, 32)},
        )

    def test_manifest_keeps_acf_representation_fusion_variants_separate(self):
        first = self._record(model="MambaACFAutoCon", seed=101)
        second = self._record(model="MambaACFAutoCon", seed=102)
        second["command"] += ["--mamba_acf_repr_fusion"]
        summary = _manifest_elapsed_summary([first, second])
        self.assertEqual(len(summary), 2)
        self.assertEqual(
            {row["mamba_acf_repr_fusion"] for row in summary}, {False, True}
        )

    def test_config_normalisation_preserves_mamba_autocon_contract(self):
        config = _normalise_config({
            "model": "Mamba2AutoCon",
            "AutoCon": True,
            "seq_len": 96,
            "label_len": 48,
            "pred_len": 96,
            "enc_in": 1,
            "dec_in": 1,
            "c_out": 1,
        })
        self.assertTrue(config["AutoCon"])
        self.assertEqual(config["mamba_headdim"], 16)
        self.assertEqual(config["AutoCon_multiscales"], [96])
        self.assertEqual(config["mamba_token_stride"], 1)
        self.assertEqual(config["mamba_acf_relation_target_mode"], "hard_selected")
        self.assertEqual(config["mamba_acf_relation_soft_min_abs"], 0.1)
        self.assertEqual(config["mamba_acf_relation_selected_lag_boost"], 0.0)
        self.assertFalse(config["mamba_acf_reliability_interpolation"])

    def test_resource_config_accepts_only_a_graph_bound_v3_autocon(self):
        config = _normalise_config({
            "model": "MambaTSv3AutoCon", "AutoCon": True,
            "seq_len": 96, "label_len": 48, "pred_len": 12,
            "enc_in": 5, "dec_in": 5, "c_out": 5,
            "mamba_ts_graph_spatial": True,
            "mamba_ts_graph_path": "/tmp/graph-contract.npz",
        })
        self.assertTrue(config["mamba_ts_graph_spatial"])
        with self.assertRaisesRegex(ValueError, "graph_path"):
            _normalise_config({
                "model": "MambaTSv3AutoCon", "AutoCon": True,
                "seq_len": 96, "label_len": 48, "pred_len": 12,
                "enc_in": 5, "dec_in": 5, "c_out": 5,
                "mamba_ts_graph_spatial": True,
            })
        baseline = _normalise_config({
            "model": "GraphDLinear", "AutoCon": False,
            "seq_len": 96, "label_len": 48, "pred_len": 12,
            "enc_in": 5, "dec_in": 5, "c_out": 5,
            "mamba_ts_graph_spatial": True,
            "mamba_ts_graph_path": "/tmp/graph-contract.npz",
        })
        self.assertEqual(baseline["model"], "GraphDLinear")
        wave = _normalise_config({
            "model": "GraphWaveNet", "AutoCon": False,
            "seq_len": 96, "label_len": 48, "pred_len": 12,
            "enc_in": 5, "dec_in": 5, "c_out": 5,
            "mamba_ts_graph_spatial": True,
            "mamba_ts_graph_path": "/tmp/graph-contract.npz",
        })
        self.assertEqual(wave["model"], "GraphWaveNet")

    def test_config_normalisation_resolves_canonical_acf_source_alias(self):
        common = {
            "model": "MambaACFAutoCon",
            "AutoCon": True,
            "seq_len": 96,
            "label_len": 48,
            "pred_len": 96,
            "enc_in": 1,
            "dec_in": 1,
            "c_out": 1,
            "mamba_acf_relation_loss": True,
        }
        lower_only = _normalise_config({
            **common, "autocon_acf_source": " RAW "
        })
        self.assertEqual(lower_only["AutoCon_acf_source"], "raw")
        conflicting = _normalise_config({
            **common,
            "autocon_acf_source": "raw",
            "AutoCon_acf_source": "trend",
        })
        self.assertEqual(conflicting["AutoCon_acf_source"], "raw")

    def test_config_normalisation_enforces_strict_representation_causality(self):
        common = {
            "model": "MambaACFAutoCon",
            "AutoCon": True,
            "mamba_acf_relation_loss": True,
            "mamba_require_strict_representation_causality": True,
            "mamba_causal_embedding": True,
            "mamba_bidirectional": False,
            "mamba_token_stride": 1,
            "mamba_scales": [1],
            "mamba_channel_gate": False,
            "mamba_acf_scale_conditioning": False,
            "mamba_local_mixer": False,
            "mamba_acf_repr_fusion": False,
            "autocon_acf_source": "raw",
            "seq_len": 96,
            "label_len": 48,
            "pred_len": 96,
            "enc_in": 1,
            "dec_in": 1,
            "c_out": 1,
        }
        with self.assertRaisesRegex(ValueError, "window_full_normalization_forbidden"):
            _normalise_config(common)
        with self.assertRaisesRegex(ValueError, "mamba_scales_must_be_single_scale_one"):
            _normalise_config({**common, "mamba_scales": [1, 2]})
        with self.assertRaisesRegex(ValueError, "supported only by MambaACF"):
            _normalise_config({
                **common,
                "model": "Mamba2",
                "AutoCon": False,
                "mamba_acf_relation_loss": False,
            })

    def test_config_normalisation_canonicalizes_local_agreement_contract(self):
        config = _normalise_config({
            "model": "MambaACF",
            "mamba_use_acf_memory": True,
            "mamba_acf_local_agreement": "true",
            "mamba_acf_local_min_pairs": 32,
            "mamba_acf_lags": [24],
            "AutoCon_acf_source": "raw",
            "seq_len": 96, "label_len": 48, "pred_len": 96,
            "enc_in": 7, "dec_in": 7, "c_out": 7,
        })
        self.assertTrue(config["mamba_acf_local_agreement"])
        self.assertEqual(config["mamba_acf_local_min_pairs"], 32)
        with self.assertRaisesRegex(ValueError, "static ACF memory"):
            _normalise_config({
                "model": "MambaACF", "mamba_use_acf_memory": True,
                "mamba_dynamic_acf": True, "mamba_acf_local_agreement": True,
                "AutoCon_acf_source": "raw", "mamba_acf_lags": [24],
                "seq_len": 96, "label_len": 48, "pred_len": 96,
                "enc_in": 7, "dec_in": 7, "c_out": 7,
            })

    def test_config_normalisation_canonicalizes_channel_lag_bank(self):
        config = _normalise_config({
            "model": "MambaACF",
            "mamba_use_acf_memory": True,
            "mamba_acf_channel_lag_bank": True,
            "mamba_acf_channel_lag_top_k": 2,
            "mamba_acf_lags": [24, 168, 336],
            "AutoCon_acf_source": "raw",
            "seq_len": 96, "label_len": 48, "pred_len": 96,
            "enc_in": 7, "dec_in": 7, "c_out": 7,
        })
        self.assertTrue(config["mamba_acf_channel_lag_bank"])
        self.assertEqual(config["mamba_acf_channel_lag_top_k"], 2)
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            _normalise_config({
                "model": "MambaACF",
                "mamba_use_acf_memory": True,
                "mamba_acf_channel_lag_bank": True,
                "mamba_acf_channel_lag_top_k": 1.5,
                "mamba_acf_lags": [24, 168, 336],
                "AutoCon_acf_source": "raw",
                "seq_len": 96, "label_len": 48, "pred_len": 96,
                "enc_in": 7, "dec_in": 7, "c_out": 7,
            })
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            _normalise_config({
                "model": "MambaACF", "mamba_use_acf_memory": True,
                "mamba_acf_channel_lag_bank": True,
                "mamba_acf_channel_lag_top_k": 4,
                "mamba_acf_lags": [24, 168, 336],
                "AutoCon_acf_source": "raw",
                "seq_len": 96, "label_len": 48, "pred_len": 96,
                "enc_in": 7, "dec_in": 7, "c_out": 7,
            })

    def test_explicit_parser_exposes_channel_lag_bank(self):
        args = _parser().parse_args([
            "--model", "MambaACF", "--seq-len", "96", "--label-len", "48",
            "--pred-len", "96", "--enc-in", "7", "--dec-in", "7", "--c-out", "7",
            "--mamba-use-acf-memory", "--mamba-acf-channel-lag-bank",
            "--mamba-acf-channel-lag-top-k", "2", "--mamba-acf-lags", "24", "168", "336",
            "--autocon-acf-source", "raw", "--output-dir", "/tmp/resource-lag-bank-test",
        ])
        request = _explicit_request(args, REPO_ROOT)
        self.assertTrue(request["config"]["mamba_acf_channel_lag_bank"])
        self.assertEqual(request["config"]["mamba_acf_channel_lag_top_k"], 2)
        with self.assertRaisesRegex(ValueError, "minimum pair support"):
            _normalise_config({
                "model": "MambaACF", "mamba_use_acf_memory": True,
                "mamba_acf_local_agreement": True,
                "mamba_acf_local_min_pairs": 90, "mamba_acf_lags": [24],
                "AutoCon_acf_source": "raw",
                "seq_len": 96, "label_len": 48, "pred_len": 96,
                "enc_in": 7, "dec_in": 7, "c_out": 7,
            })

    def test_config_normalisation_canonicalizes_new_mamba_acf_fields(self):
        config = _normalise_config({
            "model": "MambaACFAutoCon",
            "AutoCon": True,
            "mamba_acf_relation_loss": True,
            "mamba_scales": [4, 2],
            "mamba_bidirectional": "false",
            "mamba_channel_gate": "true",
            "mamba_acf_channel_gate": "true",
            "mamba_variable_mixer": True,
            "mamba_variable_mixer_depth": 2,
            "mamba_variable_group_size": 4,
            "mamba_acf_channel_order": "strength",
            "mamba_acf_channel_conditioning": "true",
            "mamba_acf_lags": "24, 48",
            "seq_len": 96,
            "label_len": 48,
            "pred_len": 96,
            "enc_in": 7,
            "dec_in": 7,
            "c_out": 7,
        })
        self.assertEqual(config["mamba_scales"], [1, 2, 4])
        self.assertEqual(config["mamba_acf_lags"], [24, 48])
        self.assertFalse(config["mamba_bidirectional"])
        self.assertTrue(config["mamba_channel_gate"])
        self.assertTrue(config["mamba_acf_channel_gate"])
        self.assertTrue(config["mamba_variable_mixer"])
        self.assertEqual(config["mamba_variable_mixer_depth"], 2)
        self.assertEqual(config["mamba_variable_group_size"], 4)
        self.assertEqual(config["mamba_acf_channel_order"], "strength")
        self.assertTrue(config["mamba_acf_channel_conditioning"])
        self.assertEqual(config["mamba_acf_relation_batching"], "lag_pairs")
        self.assertEqual(config["mamba_acf_relation_lag_balance"], "random")

        config = _normalise_config({
            "model": "MambaACFAutoCon", "AutoCon": True,
            "mamba_acf_relation_loss": True,
            "mamba_acf_scale_conditioning": "true",
            "mamba_scales": [1, 2],
            "AutoCon_acf_source": "raw",
            "seq_len": 96, "label_len": 48, "pred_len": 96,
            "enc_in": 7, "dec_in": 7, "c_out": 7,
        })
        self.assertTrue(config["mamba_acf_scale_conditioning"])

        config = _normalise_config({
            "model": "MambaACFAutoCon", "AutoCon": True,
            "mamba_acf_relation_loss": True,
            "mamba_acf_relation_batching": "lag_pairs",
            "mamba_acf_relation_lag_balance": "balanced",
            "seq_len": 96, "label_len": 48, "pred_len": 96,
            "enc_in": 7, "dec_in": 7, "c_out": 7,
        })
        self.assertEqual(config["mamba_acf_relation_lag_balance"], "balanced")

        config = _normalise_config({
            "model": "MambaACFAutoCon", "AutoCon": True,
            "mamba_acf_relation_loss": True,
            "mamba_acf_relation_target_mode": "soft_multi_peak",
            "mamba_acf_relation_soft_min_abs": 0.25,
            "mamba_acf_relation_selected_lag_boost": 0.5,
            "seq_len": 96, "label_len": 48, "pred_len": 96,
            "enc_in": 7, "dec_in": 7, "c_out": 7,
        })
        self.assertEqual(config["mamba_acf_relation_target_mode"], "soft_multi_peak")
        self.assertAlmostEqual(config["mamba_acf_relation_soft_min_abs"], 0.25)
        self.assertAlmostEqual(config["mamba_acf_relation_selected_lag_boost"], 0.5)

        config = _normalise_config({
            "model": "MambaACF1AutoCon", "AutoCon": True,
            "mamba_acf_relation_loss": True, "mamba_token_stride": 8,
            "mamba_scales": [1, 2], "seq_len": 96, "label_len": 48,
            "pred_len": 96, "enc_in": 7, "dec_in": 7, "c_out": 7,
        })
        self.assertEqual(config["mamba_token_stride"], 8)

        config = _normalise_config({
            "model": "MambaACFAutoCon", "AutoCon": True,
            "mamba_acf_relation_loss": True,
            "mamba_acf_relation_min_lag": 96,
            "seq_len": 96, "label_len": 48, "pred_len": 96,
            "enc_in": 7, "dec_in": 7, "c_out": 7,
        })
        self.assertEqual(config["mamba_acf_relation_min_lag"], 96)
        self.assertFalse(config["mamba_acf_relation_lag_mask"])
        config = _normalise_config({
            "model": "NLinearVariableMambaAutoCon", "AutoCon": True,
            "mamba_acf_relation_loss": True,
            "mamba_acf_relation_min_lag": 96,
            "mamba_variable_repr_fusion": True,
            "seq_len": 96, "label_len": 48, "pred_len": 96,
            "enc_in": 7, "dec_in": 7, "c_out": 7,
        })
        self.assertEqual(config["mamba_acf_relation_min_lag"], 96)
        config = _normalise_config({
            "model": "MambaACF", "mamba_variable_mixer": True,
            "mamba_use_acf_memory": True,
            "mamba_acf_memory_gate_mode": "variable_token",
            "seq_len": 96, "label_len": 48, "pred_len": 96,
            "enc_in": 7, "dec_in": 7, "c_out": 7,
        })
        self.assertEqual(config["mamba_acf_memory_gate_mode"], "variable_token")
        with self.assertRaisesRegex(ValueError, "memory_gate_mode=variable_token"):
            _normalise_config({
                "model": "MambaACF", "mamba_acf_memory_gate_mode": "variable_token",
                "seq_len": 96, "label_len": 48, "pred_len": 96,
                "enc_in": 7, "dec_in": 7, "c_out": 7,
            })
        with self.assertRaisesRegex(ValueError, "relation_min_lag cannot exceed"):
            _normalise_config({
                "model": "MambaACFAutoCon", "AutoCon": True,
                "mamba_acf_relation_loss": True,
                "mamba_acf_relation_min_lag": 721,
                "mamba_acf_max_lag": 720,
                "seq_len": 96, "label_len": 48, "pred_len": 96,
                "enc_in": 7, "dec_in": 7, "c_out": 7,
            })

    def test_config_normalisation_rejects_incompatible_or_missing_acf_contract(self):
        common = {
            "seq_len": 96, "label_len": 48, "pred_len": 96,
            "enc_in": 7, "dec_in": 7, "c_out": 7,
        }
        with self.assertRaisesRegex(ValueError, "relation_loss"):
            _normalise_config({**common, "model": "MambaACFAutoCon", "AutoCon": True})
        with self.assertRaisesRegex(ValueError, "channel_gate.*channel_independent"):
            _normalise_config({
                **common, "model": "MambaACF", "mamba_channel_gate": True,
                "mamba_channel_independent": True,
            })
        with self.assertRaisesRegex(ValueError, "variable_permutation"):
            _normalise_config({**common, "model": "MambaACF", "mamba_variable_permutation": True})
        with self.assertRaisesRegex(ValueError, "acf_channel_conditioning.*variable_mixer"):
            _normalise_config({
                **common, "model": "MambaACF",
                "mamba_acf_channel_conditioning": True,
            })
        with self.assertRaisesRegex(ValueError, "acf_channel_order.*variable_mixer"):
            _normalise_config({
                **common, "model": "MambaACF",
                "mamba_acf_channel_order": "strength",
            })
        with self.assertRaisesRegex(ValueError, "must be input, strength, or similarity"):
            _normalise_config({
                **common, "model": "MambaACF",
                "mamba_acf_channel_order": "unknown",
            })
        with self.assertRaisesRegex(ValueError, "AutoCon_acf_source=raw"):
            _normalise_config({
                **common, "model": "MambaACFAutoCon", "AutoCon": True,
                "mamba_acf_relation_loss": True,
                "mamba_acf_scale_conditioning": True,
                "mamba_scales": [1, 2],
                "AutoCon_acf_source": "residual",
            })

    def test_explicit_parser_exposes_new_mamba_fields(self):
        args = _parser().parse_args([
            "--model", "MambaACFAutoCon", "--seq-len", "96", "--label-len", "48",
            "--pred-len", "96", "--enc-in", "7", "--dec-in", "7", "--c-out", "7",
            "--mamba-scales", "1", "4", "--mamba-bidirectional",
            "--mamba-channel-gate", "--mamba-variable-mixer", "--mamba-variable-mixer-depth", "2",
            "--mamba-acf-channel-gate",
            "--mamba-acf-channel-order", "strength",
            "--mamba-acf-channel-conditioning",
            "--mamba-acf-repr-fusion",
            "--mamba-variable-group-size", "4",
            "--mamba-patch-mixer", "--mamba-patch-mixer-depth", "2", "--mamba-patch-len", "8",
            "--mamba-patch-stride", "4", "--mamba-patch-order", "channel_major",
            "--mamba-patch-permutation",
            "--mamba-local-mixer", "--mamba-local-patch-len", "16",
            "--mamba-local-patch-stride", "8", "--mamba-local-width", "12",
            "--mamba-local-rank", "4", "--mamba-local-depth", "2",
            "--mamba-acf-relation-loss", "--output-dir", "/tmp/resource-contract-test",
        ])
        request = _explicit_request(args, REPO_ROOT)
        config = request["config"]
        self.assertEqual(config["mamba_scales"], [1, 4])
        self.assertTrue(config["mamba_bidirectional"])
        self.assertTrue(config["mamba_channel_gate"])
        self.assertTrue(config["mamba_acf_channel_gate"])
        self.assertTrue(config["mamba_variable_mixer"])
        self.assertEqual(config["mamba_variable_mixer_depth"], 2)
        self.assertEqual(config["mamba_variable_group_size"], 4)
        self.assertEqual(config["mamba_acf_channel_order"], "strength")
        self.assertTrue(config["mamba_acf_channel_conditioning"])
        self.assertTrue(config["mamba_acf_repr_fusion"])
        self.assertTrue(config["mamba_patch_mixer"])
        self.assertEqual(config["mamba_patch_mixer_depth"], 2)
        self.assertEqual(config["mamba_patch_len"], 8)
        self.assertEqual(config["mamba_patch_stride"], 4)
        self.assertEqual(config["mamba_patch_order"], "channel_major")
        self.assertTrue(config["mamba_patch_permutation"])
        self.assertTrue(config["mamba_local_mixer"])
        self.assertEqual(config["mamba_local_patch_len"], 16)
        self.assertEqual(config["mamba_local_patch_stride"], 8)
        self.assertEqual(config["mamba_local_width"], 12)
        self.assertEqual(config["mamba_local_rank"], 4)
        self.assertEqual(config["mamba_local_depth"], 2)
        self.assertTrue(config["mamba_acf_relation_loss"])

    def test_explicit_parser_exposes_timemachine_fields(self):
        args = _parser().parse_args([
            "--model", "MambaTM2AutoCon", "--seq-len", "96", "--label-len", "48",
            "--pred-len", "96", "--enc-in", "7", "--dec-in", "7", "--c-out", "7",
            "--mamba-tm-n1", "48", "--mamba-tm-n2", "96",
            "--mamba-tm-repr-width", "32", "--mamba-tm-channel-independent",
            "--mamba-tm-residual", "false", "--output-dir", "/tmp/resource-tm-test",
        ])
        config = _explicit_request(args, REPO_ROOT)["config"]
        self.assertEqual(config["mamba_tm_n1"], 48)
        self.assertEqual(config["mamba_tm_n2"], 96)
        self.assertEqual(config["mamba_tm_repr_width"], 32)
        self.assertTrue(config["mamba_tm_channel_independent"])
        self.assertFalse(config["mamba_tm_residual"])
        self.assertTrue(config["AutoCon"])

    def test_explicit_parser_exposes_strict_representation_causality(self):
        args = _parser().parse_args([
            "--model", "MambaACFAutoCon", "--seq-len", "96", "--label-len", "48",
            "--pred-len", "96", "--enc-in", "1", "--dec-in", "1", "--c-out", "1",
            "--mamba-acf-relation-loss", "--mamba-causal-embedding",
            "--mamba-scales", "1", "--mamba-require-strict-representation-causality",
            "--autocon-acf-source", "raw", "--output-dir", "/tmp/resource-strict-contract-test",
        ])
        with self.assertRaisesRegex(ValueError, "window_full_normalization_forbidden"):
            _explicit_request(args, REPO_ROOT)

    def test_config_normalisation_rejects_local_adapter_contract_errors(self):
        common = {
            "seq_len": 96, "label_len": 48, "pred_len": 96,
            "enc_in": 7, "dec_in": 7, "c_out": 7,
        }
        with self.assertRaisesRegex(ValueError, "only by MambaACF"):
            _normalise_config({
                **common, "model": "Mamba2", "mamba_local_mixer": True,
            })
        with self.assertRaisesRegex(ValueError, "channel_gate.*local_mixer"):
            _normalise_config({
                **common, "model": "MambaACF", "mamba_local_mixer": True,
                "mamba_channel_independent": True,
            })
        with self.assertRaisesRegex(ValueError, "mamba_local_patch_len"):
            _normalise_config({
                **common, "model": "MambaACF", "mamba_local_mixer": True,
                "mamba_local_patch_len": 128,
            })
        with self.assertRaisesRegex(ValueError, "only by MambaACF"):
            _normalise_config({
                **common, "model": "Mamba2", "mamba_token_stride": 8,
            })
        with self.assertRaisesRegex(ValueError, "base tokens"):
            _normalise_config({
                **common, "model": "MambaACF", "mamba_token_stride": 64,
                "mamba_scales": [1, 3],
            })
        with self.assertRaisesRegex(ValueError, "effective temporal tokens"):
            _normalise_config({
                **common, "model": "MambaACF1", "mamba_token_stride": 8,
                "mamba_low_rank_head": True, "mamba_temporal_rank": 12,
            })

    def test_resource_record_contains_channel_profile(self):
        args = _parser().parse_args([
            "--model", "MambaACF", "--dataset", "traffic", "--features", "M",
            "--seq-len", "96", "--label-len", "48", "--pred-len", "96",
            "--enc-in", "862", "--dec-in", "862", "--c-out", "862",
            "--d-model", "32", "--d-ff", "64", "--e-layers", "1",
            "--mamba-scales", "1", "2", "--output-dir", "/tmp/resource-profile-test",
            "--mamba-token-stride", "8",
        ])
        request = _explicit_request(args, REPO_ROOT)
        record = _measure_request(request, args)
        profile = record["mamba_channel_profile"]
        self.assertEqual(profile["channel_mode"], "shared_temporal")
        self.assertEqual(profile["input_channels"], 862)
        self.assertEqual(profile["temporal_token_stride"], 8)
        self.assertEqual(profile["temporal_token_count_before_multiscale"], 12)
        self.assertEqual(profile["main_scan_tokens_per_scale"], [12, 6])
        self.assertEqual(profile["shared_mamba_component_channel_rank_bound"], 32)
        self.assertTrue(profile["linear_residual_is_channel_preserving"])

    def test_resource_record_attests_model_source_file(self):
        args = _parser().parse_args([
            "--model", "MambaACF", "--dataset", "ETTh1", "--features", "M",
            "--seq-len", "96", "--label-len", "48", "--pred-len", "96",
            "--enc-in", "7", "--dec-in", "7", "--c-out", "7",
            "--output-dir", "/tmp/resource-source-attestation-test",
        ])
        record = _measure_request(_explicit_request(args, REPO_ROOT), args)
        source = record["source"]
        self.assertEqual(source["model_module"], "MambaACF")
        self.assertEqual(source["model_source_file"], "models/MambaACF.py")
        self.assertEqual(len(source["model_source_sha256"]), 64)
        self.assertEqual(len(source["benchmark_tool_sha256"]), 64)

    def test_source_setup_reloads_snapshot_utils_modules(self):
        import importlib
        import sys
        import tempfile
        from pathlib import Path

        original_path = list(sys.path)
        package_roots = ("models", "layers", "utils")
        original_modules = {
            name: module
            for name, module in sys.modules.items()
            if any(
                name == package or name.startswith(package + ".")
                for package in package_roots
            )
        }
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for package in package_roots:
                    package_path = root / package
                    package_path.mkdir()
                    (package_path / "__init__.py").write_text("", encoding="utf-8")
                expected_path = root / "utils" / "acf_relation_contract.py"
                expected_path.write_text('SOURCE = "snapshot"\n', encoding="utf-8")
                _setup_source_root(root)
                loaded = importlib.import_module("utils.acf_relation_contract")
                self.assertEqual(loaded.SOURCE, "snapshot")
                self.assertEqual(
                    Path(loaded.__file__).resolve(), expected_path.resolve()
                )
        finally:
            for name in list(sys.modules):
                if any(
                    name == package or name.startswith(package + ".")
                    for package in package_roots
                ):
                    del sys.modules[name]
            sys.modules.update(original_modules)
            sys.path[:] = original_path
            importlib.invalidate_caches()

    def test_resource_record_contains_memory_gate_mode_cost_profile(self):
        args = _parser().parse_args([
            "--model", "MambaACF", "--dataset", "traffic", "--features", "M",
            "--seq-len", "96", "--label-len", "48", "--pred-len", "96",
            "--enc-in", "4", "--dec-in", "4", "--c-out", "4",
            "--d-model", "16", "--d-ff", "32", "--e-layers", "1",
            "--mamba-variable-mixer", "--mamba-use-acf-memory",
            "--mamba-acf-memory-gate-mode", "variable_token",
            "--output-dir", "/tmp/resource-memory-gate-profile-test",
        ])
        record = _measure_request(_explicit_request(args, REPO_ROOT), args)
        profile = record["mamba_channel_profile"]
        self.assertEqual(profile["acf_memory_gate_mode"], "variable_token")
        self.assertEqual(profile["acf_memory_gate_activation_shape"], "B,H,C")
        self.assertEqual(profile["acf_memory_gate_output_elements_per_batch_item"], 384)
        self.assertEqual(profile["acf_memory_gate_input_tokens"], 4)

    def test_resource_record_contains_local_patch_profile(self):
        args = _parser().parse_args([
            "--model", "MambaACF", "--dataset", "traffic", "--features", "M",
            "--seq-len", "96", "--label-len", "48", "--pred-len", "96",
            "--enc-in", "862", "--dec-in", "862", "--c-out", "862",
            "--d-model", "32", "--d-ff", "64", "--e-layers", "1",
            "--mamba-local-mixer", "--mamba-local-width", "12",
            "--mamba-local-rank", "8", "--output-dir", "/tmp/resource-local-profile-test",
        ])
        record = _measure_request(_explicit_request(args, REPO_ROOT), args)
        profile = record["mamba_channel_profile"]
        self.assertTrue(profile["local_patch_enabled"])
        self.assertTrue(profile["local_patch_channel_preserving"])
        self.assertEqual(profile["local_patch_cross_channel_parameters"], 0)
        self.assertEqual(profile["local_patch_width"], 12)

    def test_strength_order_resource_measurement_activates_only_synthetic_prior(self):
        args = _parser().parse_args([
            "--model", "MambaACF", "--dataset", "traffic", "--features", "M",
            "--seq-len", "96", "--label-len", "48", "--pred-len", "96",
            "--enc-in", "4", "--dec-in", "4", "--c-out", "4",
            "--d-model", "16", "--d-ff", "32", "--e-layers", "1",
            "--mamba-variable-mixer", "--mamba-acf-channel-order", "strength",
            "--output-dir", "/tmp/resource-order-profile-test",
        ])
        record = _measure_request(_explicit_request(args, REPO_ROOT), args)
        self.assertEqual(
            record["acf_channel_order_prior_source"],
            "synthetic_deterministic_activation",
        )
        profile = record["mamba_channel_profile"]
        self.assertEqual(profile["variable_token_order"], "strength")
        self.assertTrue(profile["variable_token_order_prior_attached"])

    def test_similarity_order_resource_measurement_activates_only_synthetic_prior(self):
        args = _parser().parse_args([
            "--model", "MambaACF", "--dataset", "traffic", "--features", "M",
            "--seq-len", "96", "--label-len", "48", "--pred-len", "96",
            "--enc-in", "4", "--dec-in", "4", "--c-out", "4",
            "--d-model", "16", "--d-ff", "32", "--e-layers", "1",
            "--mamba-variable-mixer", "--mamba-acf-channel-order", "similarity",
            "--output-dir", "/tmp/resource-similarity-profile-test",
        ])
        record = _measure_request(_explicit_request(args, REPO_ROOT), args)
        self.assertEqual(
            record["acf_channel_order_prior_source"],
            "synthetic_deterministic_activation",
        )
        profile = record["mamba_channel_profile"]
        self.assertEqual(profile["variable_token_order"], "similarity")
        self.assertEqual(
            profile["variable_token_order_rule"],
            "signed_acf_profile_cosine_greedy",
        )
        self.assertTrue(profile["variable_token_order_prior_attached"])

    def test_resource_acf_prior_activation_uses_shared_contract(self):
        import torch

        class _Probe:
            def __init__(self):
                self.prior = None
                self.lags = None

            def set_acf_prior(self, prior, lags=None):
                self.prior = prior
                self.lags = lags

        cases = (
            (
                {
                    "model": "MambaACF",
                    "enc_in": 2,
                    "mamba_acf_lags": [2],
                    "mamba_acf_channel_conditioning": True,
                },
                "channel_conditioning",
            ),
            (
                {
                    "model": "MambaACFAutoCon",
                    "AutoCon": True,
                    "enc_in": 2,
                    "mamba_acf_lags": [2],
                    "mamba_acf_repr_fusion": True,
                },
                "representation_fusion",
            ),
        )
        for config, marker in cases:
            probe = _Probe()
            source = _activate_resource_acf_channel_order(probe, config, torch)
            self.assertEqual(tuple(probe.prior.shape), (2, 3))
            self.assertEqual(probe.lags, [2])
            self.assertIn(marker, source)

    def test_resource_csv_discloses_acf_representation_fusion(self):
        record = {
            "model": "MambaACFAutoCon",
            "dataset": "ETTh1",
            "features": "M",
            "seq_len": 96,
            "label_len": 48,
            "pred_len": 96,
            "enc_in": 1,
            "dec_in": 1,
            "c_out": 1,
            "resource_signature_sha256": "signature",
            "parameters": {
                "total": 1,
                "trainable": 1,
                "parameter_mb": 0.0,
                "buffer_mb": 0.0,
            },
            "effective_mamba_architecture": {},
            "mamba_channel_profile": {},
            "architecture": {"mamba_acf_repr_fusion": True},
            "gpu_inference": {"status": "not_run"},
            "source": {"kind": "explicit_cli"},
        }
        row = _flatten_records([record])[0]
        self.assertTrue(row["architecture_mamba_acf_repr_fusion"])

    def test_relation_only_resource_measurement_attaches_sidecar_without_forecast_cache(self):
        args = _parser().parse_args([
            "--model", "MambaACFAutoCon", "--dataset", "ETTh1", "--features", "M",
            "--seq-len", "96", "--label-len", "48", "--pred-len", "96",
            "--enc-in", "7", "--dec-in", "7", "--c-out", "7",
            "--d-model", "16", "--d-ff", "32", "--e-layers", "1",
            "--mamba-acf-relation-loss", "--autocon-acf-source", "raw",
            "--output-dir", "/tmp/resource-relation-only-profile-test",
        ])
        request = _explicit_request(args, REPO_ROOT)
        record = _measure_request(request, args)
        self.assertIn("synthetic_deterministic_acf_prior", record["acf_channel_order_prior_source"])
        profile = record["mamba_channel_profile"]
        self.assertEqual(profile["acf_anchor_copy_enabled"], False)
        # The compact/full train-only sidecar is present, while the lazy
        # forecast lookup tables remain empty for this relation-only config.
        self.assertGreater(record["parameters"]["buffer_mb"], 0.0)

    def test_percentile_interpolates_without_treating_values_as_replicates(self):
        self.assertEqual(_percentile([1.0, 2.0, 3.0], 50.0), 2.0)
        self.assertEqual(_percentile([1.0, 2.0, 3.0, 4.0], 95.0), 3.85)

    def test_ms_prediction_uses_the_trainer_target_slice(self):
        import torch

        prediction = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
        result = _task_prediction(prediction, {"features": "MS"}, torch)
        self.assertEqual(tuple(result.shape), (2, 4, 1))
        self.assertTrue(torch.equal(result[:, :, 0], prediction[:, :, -1]))
        self.assertTrue(torch.equal(_task_prediction(prediction, {"features": "M"}, torch), prediction))

    def test_forward_prediction_disables_auxiliary_representation(self):
        import torch

        class _Probe(torch.nn.Module):
            supports_return_repr_control = True

            def forward(self, *inputs, return_repr=None):
                self.return_repr = return_repr
                prediction = torch.zeros(2, 3, 1)
                return prediction, torch.ones(2, 4, 5)

        probe = _Probe().eval()
        result = _forward_prediction(
            probe,
            (torch.zeros(2, 8, 1), None, None, None),
            torch,
            {"features": "M"},
        )
        self.assertEqual(tuple(result.shape), (2, 3, 1))
        self.assertIs(probe.return_repr, False)

    def test_forward_outputs_can_request_auxiliary_representation(self):
        import torch

        class _Probe(torch.nn.Module):
            supports_return_repr_control = True

            def forward(self, *inputs, return_repr=None):
                self.return_repr = return_repr
                prediction = torch.zeros(2, 3, 1)
                representation = torch.ones(2, 8, 5) if return_repr else None
                return prediction, representation

        probe = _Probe().eval()
        prediction, representation = _forward_outputs(
            probe,
            (torch.zeros(2, 8, 1), None, None, None),
            torch,
            {"features": "M"},
            request_representation=True,
        )
        self.assertEqual(tuple(prediction.shape), (2, 3, 1))
        self.assertEqual(tuple(representation.shape), (2, 8, 5))
        self.assertIs(probe.return_repr, True)

    def test_resource_parser_exposes_representation_measurement_switch(self):
        args = _parser().parse_args([
            "--model", "MambaACFAutoCon", "--seq-len", "96",
            "--label-len", "48", "--pred-len", "96", "--enc-in", "1",
            "--dec-in", "1", "--c-out", "1", "--include-representation",
            "--output-dir", "/tmp/resource-representation-parser-test",
        ])
        self.assertTrue(args.include_representation)

    def test_representation_shape_contract_rejects_missing_required_output(self):
        import torch

        self.assertIsNone(_representation_shape(None, torch, required=False))
        with self.assertRaisesRegex(ValueError, "returned no auxiliary representation"):
            _representation_shape(None, torch, required=True)

    def test_representation_shape_contract_records_tensor_dimensions(self):
        import torch

        result = _representation_shape(torch.zeros(2, 8, 5), torch, required=True)
        self.assertEqual(result, [2, 8, 5])


if __name__ == "__main__":
    unittest.main()
