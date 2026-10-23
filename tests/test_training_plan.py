import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.run_training_plan import (
    MODEL_SPECS, _artifact_exists, _artifact_hashes, _command, _job_specs, _sha256,
    _registered_job_count_errors, _validate_data_fingerprints,
    _canonical_hash, _validate_confirmation_launch_binding,
    _validate_launch_authorization, _validate_plan,
    _validate_mamba_options, _validate_variable_gdd_mixer_contract,
    _cpu_thread_environment, _validate_numeric_plan_config,
)
from scripts.validate_training_phase import (
    _DEFAULT_MAMBA_SCALES,
    _acf_source_value,
    _validate_acf_prior_artifact,
    _validate_memory_lag_contract,
    _validate_record,
    _validate_strict_representation_causality_contract,
)
from utils.acf_guidance import compile_channel_lag_support


class TrainingPlanTests(unittest.TestCase):
    def test_cpu_thread_limit_generates_blas_environment(self):
        expected = {
            "AUTOCON_CPU_THREAD_LIMIT": "4",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "NUMEXPR_NUM_THREADS": "4",
        }
        self.assertEqual(_cpu_thread_environment({"cpu_thread_limit": 4}), expected)
        self.assertEqual(_cpu_thread_environment({}), {})
        self.assertTrue(_validate_numeric_plan_config(
            "limited", {"cpu_thread_limit": 0}
        ))

    def test_phase_validator_uses_the_runner_single_scale_default(self):
        """An omitted scale field must replay the command runner's [1] default."""

        self.assertEqual(_DEFAULT_MAMBA_SCALES, [1])
        job = _job_specs(
            {"global": {"features": "M", "seq_len": 96}},
            {
                "name": "mamba2_default", "datasets": ["ETTh1"],
                "models": ["Mamba2"], "features": "M",
                "horizons": [96], "seeds": [2021],
            },
            {},
        )[0]
        roots = {
            "checkpoints": Path("/tmp/mamba2-default-checkpoints"),
            "results": Path("/tmp/mamba2-default-results"),
            "test": Path("/tmp/mamba2-default-test"),
            "result_log": Path("/tmp/mamba2-default-results.txt"),
        }
        _, command = _command(job, roots)
        scale_index = command.index("--mamba_scales")
        self.assertEqual(command[scale_index + 1], str(_DEFAULT_MAMBA_SCALES[0]))

    def test_traffic_graph_command_uses_registered_hashed_contract(self):
        job = {
            "phase": "graph_contract", "dataset": "metr_la",
            "model": "MambaTSv3AutoCon", "pred_len": 12, "seed": 2021,
            "autocon_lambda": 0.01,
            "config": {
                "features": "M", "seq_len": 96, "label_len": 48,
                "enc_in": 207, "dec_in": 207, "c_out": 207,
                "mamba_ts_graph_spatial": True,
            },
        }
        roots = {
            "checkpoints": Path("/tmp/graph-contract-checkpoints"),
            "results": Path("/tmp/graph-contract-results"),
            "test": Path("/tmp/graph-contract-test"),
            "result_log": Path("/tmp/graph-contract-results.txt"),
        }
        _, command = _command(job, roots)
        self.assertIn("--mamba_ts_graph_spatial", command)
        graph_path = Path(command[command.index("--mamba_ts_graph_path") + 1])
        self.assertEqual(graph_path.name, "adjacency.npz")
        self.assertTrue(graph_path.is_file())
        self.assertEqual(
            command[command.index("--mamba_ts_graph_expected_sha256") + 1],
            "97cfc5c63ab90bc6517108881f1aa3c26c62fbc548584b4ec76b28b0e16eac89",
        )
        self.assertEqual(
            _validate_mamba_options(
                "graph_contract", {"mamba_ts_graph_spatial": True},
                ["MambaTSv3AutoCon"],
            ),
            [],
        )

    def test_graph_screen_commands_keep_topology_and_autocon_arms_distinct(self):
        plan_path = Path(
            "docs/research/metro_pemsbay_graph_spatial_mambatsv3_autocon_screen_plan_20260927.json"
        )
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        roots = {
            "checkpoints": Path("/tmp/graph-screen-checkpoints"),
            "results": Path("/tmp/graph-screen-results"),
            "test": Path("/tmp/graph-screen-test"),
            "result_log": Path("/tmp/graph-screen-results.txt"),
        }
        commands = {}
        for phase in plan["phases"]:
            job = _job_specs(plan, phase, {}, allow_unresolved=True)[0]
            commands[phase["name"]] = _command(job, roots)[1]
        control = commands["mambatsv3_autocon_graph_off_h12"]
        treatment = commands["mambatsv3_autocon_graph_on_h12"]
        graph_baseline = commands["graphdlinear_h12"]
        external = commands["external_context_h12"]
        self.assertIn("--AutoCon", control)
        self.assertIn("--AutoCon", treatment)
        self.assertNotIn("--mamba_ts_graph_spatial", control)
        self.assertIn("--mamba_ts_graph_spatial", treatment)
        self.assertIn("--mamba_ts_graph_path", treatment)
        self.assertIn("--mamba_ts_graph_spatial", graph_baseline)
        self.assertNotIn("--AutoCon", graph_baseline)
        self.assertNotIn("--mamba_ts_graph_spatial", external)
        self.assertEqual(
            _validate_mamba_options(
                "graph_baseline", {"mamba_ts_graph_spatial": True},
                ["GraphDLinear"],
            ),
            [],
        )
        self.assertEqual(
            _validate_mamba_options(
                "graph_wavenet", {"mamba_ts_graph_spatial": True},
                ["GraphWaveNet"],
            ),
            [],
        )
        self.assertEqual(
            _validate_mamba_options(
                "dcrnn_graph", {"mamba_ts_graph_spatial": True},
                ["DCRNN"],
            ),
            [],
        )
        self.assertIn("AGCRN", MODEL_SPECS)
    def test_nlinear_variable_mamba_accepts_its_rezero_residual_initialization(self):
        errors = _validate_mamba_options(
            "nlinear_variable",
            {
                "seq_len": 96,
                "mamba_ts_residual_init": 0.1,
            },
            ["NLinearVariableMambaAutoCon"],
        )
        self.assertEqual(errors, [])

    def test_meanstd_baseline_pilot_plan_is_valid_and_complete(self):
        plan_path = Path(
            "docs/research/mamba_meanstd_baseline_pilot_20260907.json"
        )
        with plan_path.open(encoding="utf-8") as handle:
            plan = json.load(handle)
        self.assertEqual(_validate_plan(plan), [])
        jobs = _job_specs(plan, plan["phases"][0], {}, allow_unresolved=True)
        self.assertEqual(len(jobs), 84)
        self.assertEqual(
            {job["model"] for job in jobs},
            {"DLinear", "NLinear", "PatchTST", "TimesNet", "iTransformer", "Mamba1", "Mamba2"},
        )
        self.assertEqual({job["pred_len"] for job in jobs}, {96})
        self.assertEqual(
            {job["config"]["autocon_wnorm"] for job in jobs}, {"MeanStd"}
        )

    def test_mamba_ts_v3_strong_baseline_screen_plan_is_valid_and_complete(self):
        plan_path = Path(
            "docs/research/mamba_pems_ts_v3_strong_baseline_screen_20260909.json"
        )
        with plan_path.open(encoding="utf-8") as handle:
            plan = json.load(handle)
        self.assertEqual(_validate_plan(plan), [])
        jobs = _job_specs(plan, plan["phases"][0], {}, allow_unresolved=True)
        self.assertEqual(len(jobs), 40)
        self.assertEqual(
            {job["dataset"] for job in jobs},
            {"pems03", "pems04", "pems07", "pems08"},
        )
        self.assertEqual(
            {job["model"] for job in jobs},
            {"DLinear", "PatchTST", "TimesNet", "Mamba2", "MambaTSv3"},
        )
        self.assertEqual({job["pred_len"] for job in jobs}, {12, 96})
        self.assertEqual({job["seed"] for job in jobs}, {2022})
        self.assertEqual(
            {job["config"]["autocon_wnorm"] for job in jobs}, {"MeanStd"}
        )

    def test_mamba_ts_v3_timesnet_fresh_seed_plan_is_valid_and_complete(self):
        plan_path = Path(
            "docs/research/mamba_pems_ts_v3_timesnet_fresh_seed_20260909.json"
        )
        with plan_path.open(encoding="utf-8") as handle:
            plan = json.load(handle)
        self.assertEqual(_validate_plan(plan), [])
        jobs = _job_specs(plan, plan["phases"][0], {}, allow_unresolved=True)
        self.assertEqual(len(jobs), 16)
        self.assertEqual({job["model"] for job in jobs}, {"TimesNet", "MambaTSv3"})
        self.assertEqual({job["pred_len"] for job in jobs}, {12, 96})
        self.assertEqual({job["seed"] for job in jobs}, {2023})

    def test_classical_extension_baselines_are_registered(self):
        self.assertEqual(
            {name for name in (
                "Autoformer", "FEDformer", "ETSformer", "Transformer", "Crossformer"
            ) if name in MODEL_SPECS},
            {"Autoformer", "FEDformer", "ETSformer", "Transformer", "Crossformer"},
        )

    def test_classical_core_extension_plan_is_valid_and_complete(self):
        plan_path = Path(
            "docs/research/mamba_classical_baseline_core_extension_pilot_20260907.json"
        )
        with plan_path.open(encoding="utf-8") as handle:
            plan = json.load(handle)
        self.assertEqual(_validate_plan(plan), [])
        jobs = _job_specs(plan, plan["phases"][0], {}, allow_unresolved=True)
        self.assertEqual(len(jobs), 60)
        self.assertEqual(
            {job["dataset"] for job in jobs},
            {"ETTh1", "ETTm1", "electricity", "traffic"},
        )
        self.assertEqual(
            {job["model"] for job in jobs},
            {"Autoformer", "FEDformer", "ETSformer", "Transformer", "Crossformer"},
        )
        self.assertEqual({job["pred_len"] for job in jobs}, {96})
        self.assertEqual(
            {job["config"]["autocon_wnorm"] for job in jobs}, {"MeanStd"}
        )

    def test_timemachine_plan_is_valid_and_expands_plain_and_autocon_arms(self):
        plan_path = Path(
            "docs/research/mamba_timemachine_baseline_screen_plan_20260907.json"
        )
        with plan_path.open(encoding="utf-8") as handle:
            plan = json.load(handle)
        self.assertEqual(_validate_plan(plan), [])
        jobs = _job_specs(plan, plan["phases"][0], {}, allow_unresolved=True)
        self.assertEqual(len(jobs), 165)
        self.assertIn("MambaTM1", MODEL_SPECS)
        self.assertIn("MambaTM2AutoCon", MODEL_SPECS)
        self.assertFalse(
            next(job for job in jobs if job["model"] == "MambaTM2")[
                "autocon_lambda"
            ]
        )
        self.assertEqual(
            next(job for job in jobs if job["model"] == "MambaTM2AutoCon")[
                "autocon_lambda"
            ],
            0.03,
        )

    def test_smamba_variants_are_registered_with_autocon_metadata(self):
        self.assertEqual(MODEL_SPECS["MambaSM"], {"model": "MambaSM", "autocon": False})
        self.assertEqual(
            MODEL_SPECS["MambaSMAutoCon"],
            {"model": "MambaSMAutoCon", "autocon": True},
        )

    def test_mambats_v6_variants_are_registered_with_autocon_metadata(self):
        self.assertEqual(MODEL_SPECS["MambaTSv6"], {"model": "MambaTSv6", "autocon": False})
        self.assertEqual(
            MODEL_SPECS["MambaTSv6AutoCon"],
            {"model": "MambaTSv6AutoCon", "autocon": True},
        )
        self.assertEqual(
            MODEL_SPECS["MambaTSv6WarmAutoCon"]["config_overrides"],
            {"mamba_ts_residual_init": 0.1},
        )

    def test_mambats_v7_variants_are_registered_with_autocon_metadata(self):
        self.assertEqual(MODEL_SPECS["MambaTSv7"], {"model": "MambaTSv7", "autocon": False})
        self.assertEqual(MODEL_SPECS["MambaTSv7AutoCon"], {"model": "MambaTSv7AutoCon", "autocon": True})

    def test_mambats_v9_variants_are_registered_with_autocon_metadata(self):
        self.assertEqual(MODEL_SPECS["MambaTSv9"], {"model": "MambaTSv9", "autocon": False})
        self.assertEqual(MODEL_SPECS["MambaTSv9AutoCon"], {"model": "MambaTSv9AutoCon", "autocon": True})

    def test_mambats_v10_variants_are_registered_with_autocon_metadata(self):
        self.assertEqual(MODEL_SPECS["MambaTSv10"], {"model": "MambaTSv10", "autocon": False})
        self.assertEqual(MODEL_SPECS["MambaTSv10AutoCon"], {"model": "MambaTSv10AutoCon", "autocon": True})

    def test_mambats_v11_variants_are_registered_with_autocon_metadata(self):
        self.assertEqual(MODEL_SPECS["MambaTSv11"], {"model": "MambaTSv11", "autocon": False})
        self.assertEqual(MODEL_SPECS["MambaTSv11AutoCon"], {"model": "MambaTSv11AutoCon", "autocon": True})

    def test_mambats_v12_variants_are_registered_with_autocon_metadata(self):
        self.assertEqual(MODEL_SPECS["MambaTSv12"], {"model": "MambaTSv12", "autocon": False})
        self.assertEqual(MODEL_SPECS["MambaTSv12AutoCon"], {"model": "MambaTSv12AutoCon", "autocon": True})

    def test_mambats_v13_variants_are_registered_with_autocon_metadata(self):
        self.assertEqual(MODEL_SPECS["MambaTSv13"], {"model": "MambaTSv13", "autocon": False})
        self.assertEqual(MODEL_SPECS["MambaTSv13AutoCon"], {"model": "MambaTSv13AutoCon", "autocon": True})
        errors = _validate_mamba_options(
            "v13", {"mamba_ts_repr_forecast_fusion": True}, ["MambaTSv13AutoCon"]
        )
        self.assertEqual(errors, [])

    def test_mambats_v14_variants_are_registered_with_autocon_metadata(self):
        self.assertEqual(MODEL_SPECS["MambaTSv14"], {"model": "MambaTSv14", "autocon": False})
        self.assertEqual(MODEL_SPECS["MambaTSv14AutoCon"], {"model": "MambaTSv14AutoCon", "autocon": True})
        errors = _validate_mamba_options(
            "v14", {"mamba_ts_repr_forecast_fusion": True}, ["MambaTSv14AutoCon"]
        )
        self.assertEqual(errors, [])

    def test_mambats_v15_variants_are_registered_with_autocon_metadata(self):
        self.assertEqual(MODEL_SPECS["MambaTSv15"], {"model": "MambaTSv15", "autocon": False})
        self.assertEqual(MODEL_SPECS["MambaTSv15AutoCon"], {"model": "MambaTSv15AutoCon", "autocon": True})
        self.assertEqual(
            _validate_mamba_options(
                "v15", {"mamba_ts_repr_forecast_fusion": True}, ["MambaTSv15AutoCon"]
            ),
            [],
        )

    def test_mambats_v16_variants_are_registered_with_autocon_metadata(self):
        self.assertEqual(MODEL_SPECS["MambaTSv16"], {"model": "MambaTSv16", "autocon": False})
        self.assertEqual(MODEL_SPECS["MambaTSv16AutoCon"], {"model": "MambaTSv16AutoCon", "autocon": True})
        self.assertEqual(
            _validate_mamba_options(
                "v16", {"mamba_ts_repr_forecast_fusion": True}, ["MambaTSv16AutoCon"]
            ),
            [],
        )

    def test_timemachine_command_serializes_its_architecture_and_autocon_flag(self):
        plan = {
            "global": {
                "features": "M", "seq_len": 16, "label_len": 8,
                "d_model": 16, "d_ff": 32, "e_layers": 1,
                "mamba_d_state": 4, "mamba_d_conv": 3,
                "mamba_expand": 2, "mamba_headdim": 4,
                "mamba_tm_n1": 8, "mamba_tm_n2": 12,
                "mamba_tm_repr_width": 8,
                "mamba_tm_channel_independent": False,
                "mamba_tm_residual": True,
                "autocon_wnorm": "MeanStd", "autocon_scale": 96,
            }
        }
        phase = {
            "name": "tm", "datasets": ["ETTh1"],
            "models": ["MambaTM2", "MambaTM2AutoCon"],
            "horizons": [8], "seeds": [2021], "features": "M",
            "autocon_lambda": 0.03,
        }
        jobs = _job_specs(plan, phase, {}, allow_unresolved=True)
        roots = {key: Path("/tmp") / ("tm-test-" + key) for key in (
            "checkpoints", "results", "test", "logs"
        )}
        roots["result_log"] = Path("/tmp/tm-test-results.txt")
        plain_id, plain_cmd = _command(jobs[0], roots, provenance_id="test")
        ac_id, ac_cmd = _command(jobs[1], roots, provenance_id="test")
        self.assertIn("--mamba_tm_n1", plain_cmd)
        self.assertIn("--mamba_tm_n2", plain_cmd)
        self.assertIn("--mamba_tm_repr_width", plain_cmd)
        self.assertNotIn("--AutoCon", plain_cmd)
        self.assertIn("--AutoCon", ac_cmd)
        self.assertNotEqual(plain_id, ac_id)

    def test_acf_source_alias_resolution_matches_runner_precedence(self):
        self.assertEqual(_acf_source_value({"autocon_acf_source": "RAW"}), "raw")
        self.assertEqual(_acf_source_value({"AutoCon_acf_source": "Trend"}), "trend")
        # A phase command is built from the lower-case plan key when both
        # spellings are present, so the phase validator must resolve it first.
        self.assertEqual(
            _acf_source_value(
                {"autocon_acf_source": "raw", "AutoCon_acf_source": "residual"}
            ),
            "raw",
        )
        self.assertEqual(_acf_source_value({}), "residual")

    def test_strict_acf_sidecar_accepts_canonical_lags_and_checks_full_table(self):
        with tempfile.TemporaryDirectory() as directory:
            result_dir = Path(directory)
            artifact = result_dir / "acf_prior.npz"
            full_values = np.zeros((2, 11), dtype=np.float32)
            # The compact columns are stored in canonical lag order even when
            # the resolved config records the selector's ranking order.
            full_values[:, 3] = [0.25, -0.5]
            full_values[:, 7] = [-0.75, 0.125]
            values = full_values[:, [3, 7]].copy()

            def write_artifact(current_full):
                np.savez_compressed(
                    artifact,
                    schema_version=np.asarray(1, dtype=np.int64),
                    acf_values=values,
                    acf_full_values=current_full,
                    lags=np.asarray([3, 7], dtype=np.int64),
                    seq_len=np.asarray(4, dtype=np.int64),
                    pred_len=np.asarray(2, dtype=np.int64),
                    channels=np.asarray(2, dtype=np.int64),
                    acf_source=np.asarray("raw"),
                    acf_temperature=np.asarray(4.0, dtype=np.float64),
                )

            write_artifact(full_values)
            config = {
                "model": "MambaACF",
                "mamba_use_acf_memory": True,
                "mamba_dynamic_acf": False,
                "mamba_acf_prior_artifact": "acf_prior.npz",
                "mamba_acf_prior_sha256": _sha256(artifact),
                "mamba_acf_effective_lags": [7, 3],
                "mamba_acf_prior_shape": [2, 2],
                "mamba_acf_prior_full_shape": [2, 11],
                "mamba_acf_prior_max_lag": 10,
                "seq_len": 4,
                "pred_len": 2,
                "enc_in": 2,
                "AutoCon_acf_source": "raw",
                "mamba_acf_temperature": 4.0,
            }
            job = {"model": "MambaACF"}
            self.assertEqual(
                _validate_acf_prior_artifact(config, result_dir, job, strict=True),
                [],
            )

            corrupted = full_values.copy()
            corrupted[0, 3] += 0.1
            write_artifact(corrupted)
            config["mamba_acf_prior_sha256"] = _sha256(artifact)
            errors = _validate_acf_prior_artifact(
                config, result_dir, job, strict=True
            )
            self.assertIn(
                "mamba_acf_prior_complete_table_compact_values_mismatch",
                errors,
            )

            write_artifact(full_values)
            with np.load(artifact, allow_pickle=False) as payload:
                fields = {name: payload[name] for name in payload.files}
            fields["lags"] = np.asarray([3.5, 7.0], dtype=np.float32)
            np.savez_compressed(artifact, **fields)
            config["mamba_acf_prior_sha256"] = _sha256(artifact)
            errors = _validate_acf_prior_artifact(
                config, result_dir, job, strict=True
            )
            self.assertIn("mamba_acf_prior_artifact_lags_dtype_invalid", errors)

    def test_channel_lag_bank_sidecar_recomputes_support_and_accepts_unsorted_lags(self):
        """Phase validation must verify derived support metadata, not trust it."""
        with tempfile.TemporaryDirectory() as directory:
            result_dir = Path(directory)
            artifact = result_dir / "acf_prior.npz"
            canonical_lags = [1, 3, 5]
            values = np.asarray(
                [[0.9, 0.1, 0.2], [0.1, 0.8, 0.3]], dtype=np.float32
            )
            full_values = np.zeros((2, 6), dtype=np.float32)
            full_values[:, canonical_lags] = values
            support, legal_fraction, _ = compile_channel_lag_support(
                values, canonical_lags, seq_len=8, horizon=3, top_k=2
            )

            def write_sidecar(sidecar_lags, sidecar_values):
                np.savez_compressed(
                    artifact,
                    schema_version=np.asarray(1, dtype=np.int64),
                    acf_values=np.asarray(sidecar_values, dtype=np.float32),
                    acf_full_values=full_values,
                    lags=np.asarray(sidecar_lags, dtype=np.int64),
                    seq_len=np.asarray(8, dtype=np.int64),
                    pred_len=np.asarray(3, dtype=np.int64),
                    channels=np.asarray(2, dtype=np.int64),
                    acf_source=np.asarray("raw"),
                    acf_temperature=np.asarray(4.0, dtype=np.float64),
                )

            write_sidecar(canonical_lags, values)
            config = {
                "model": "MambaACF",
                "mamba_use_acf_memory": True,
                "mamba_dynamic_acf": False,
                "mamba_acf_channel_lag_bank": True,
                "mamba_acf_channel_lag_top_k": 2,
                "mamba_acf_prior_artifact": "acf_prior.npz",
                "mamba_acf_prior_sha256": _sha256(artifact),
                "mamba_acf_effective_lags": canonical_lags,
                "mamba_acf_prior_shape": [2, 3],
                "mamba_acf_prior_full_shape": [2, 6],
                "mamba_acf_prior_max_lag": 5,
                "seq_len": 8,
                "pred_len": 3,
                "enc_in": 2,
                "AutoCon_acf_source": "raw",
                "mamba_acf_temperature": 4.0,
                "mamba_acf_channel_lag_bank_selection": (
                    "abs_acf_times_legal_step_fraction"
                ),
                "mamba_acf_channel_lag_bank_resolved_lags": canonical_lags,
                "mamba_acf_channel_lag_bank_legal_lags": [1, 3, 5],
                "mamba_acf_channel_lag_bank_legal_step_fractions": (
                    legal_fraction.tolist()
                ),
                "mamba_acf_channel_lag_bank_support_shape": [2, 3],
                "mamba_acf_channel_lag_bank_selected_slots": int(support.sum()),
                "mamba_acf_channel_lag_bank_total_slots": int(support.size),
                "mamba_acf_channel_lag_bank_density": float(
                    support.sum() / support.size
                ),
                "mamba_acf_channel_lag_bank_support_sha256": hashlib.sha256(
                    support.tobytes(order="C")
                ).hexdigest(),
            }
            job = {"model": "MambaACF"}
            self.assertEqual(
                _validate_acf_prior_artifact(config, result_dir, job, strict=True),
                [],
            )

            config["mamba_acf_channel_lag_bank_selection"] = "forged"
            self.assertIn(
                "config_mamba_acf_channel_lag_selection_invalid",
                _validate_acf_prior_artifact(config, result_dir, job, strict=True),
            )
            config["mamba_acf_channel_lag_bank_selection"] = (
                "abs_acf_times_legal_step_fraction"
            )
            config["mamba_acf_channel_lag_bank_resolved_lags"] = [True, 3, 5]
            self.assertIn(
                "config_mamba_acf_channel_lag_bank_resolved_lags_contains_nonpositive_or_nonfinite",
                _validate_acf_prior_artifact(config, result_dir, job, strict=True),
            )
            config["mamba_acf_channel_lag_bank_resolved_lags"] = canonical_lags

            config["mamba_acf_channel_lag_bank_support_sha256"] = "0" * 64
            self.assertIn(
                "config_mamba_acf_channel_lag_support_hash_mismatch",
                _validate_acf_prior_artifact(config, result_dir, job, strict=True),
            )
            config["mamba_acf_channel_lag_bank_support_sha256"] = "z" * 64
            self.assertIn(
                "config_mamba_acf_channel_lag_support_hash_invalid",
                _validate_acf_prior_artifact(config, result_dir, job, strict=True),
            )

            config["mamba_acf_channel_lag_bank_support_sha256"] = hashlib.sha256(
                support.tobytes(order="C")
            ).hexdigest()
            config["mamba_acf_channel_lag_top_k"] = 1.5
            self.assertIn(
                "config_mamba_acf_channel_lag_top_k_invalid",
                _validate_acf_prior_artifact(config, result_dir, job, strict=True),
            )

            config["mamba_acf_channel_lag_top_k"] = 2
            unsorted_lags = [5, 1, 3]
            unsorted_values = values[:, [2, 0, 1]]
            write_sidecar(unsorted_lags, unsorted_values)
            config["mamba_acf_prior_sha256"] = _sha256(artifact)
            self.assertEqual(
                _validate_acf_prior_artifact(config, result_dir, job, strict=True),
                [],
            )

            with np.load(artifact, allow_pickle=False) as payload:
                fields = {name: payload[name] for name in payload.files}
            fields["seq_len"] = np.asarray(8.5, dtype=np.float64)
            np.savez_compressed(artifact, **fields)
            config["mamba_acf_prior_sha256"] = _sha256(artifact)
            self.assertIn(
                "mamba_acf_prior_artifact_seq_len_invalid",
                _validate_acf_prior_artifact(config, result_dir, job, strict=True),
            )

    def test_strict_sidecar_is_required_for_memory_off_mamba_acf_ablation(self):
        """Strict ablations retain the train-only prior for replay/audit."""
        with tempfile.TemporaryDirectory() as directory:
            result_dir = Path(directory)
            config = {
                "model": "MambaACF",
                "mamba_use_acf_memory": False,
                "mamba_dynamic_acf": False,
                "mamba_acf_prior_artifact": "acf_prior.npz",
                "mamba_acf_prior_sha256": "0" * 64,
                "mamba_acf_effective_lags": [3, 7],
                "mamba_acf_prior_shape": [2, 2],
                "mamba_acf_prior_full_shape": [2, 11],
                "mamba_acf_prior_max_lag": 10,
                "seq_len": 4,
                "pred_len": 2,
                "enc_in": 2,
                "AutoCon_acf_source": "raw",
                "mamba_acf_temperature": 4.0,
            }
            job = {"model": "MambaACF"}
            self.assertIn(
                "mamba_acf_prior_artifact_missing",
                _validate_acf_prior_artifact(config, result_dir, job, strict=True),
            )
            full_values = np.zeros((2, 11), dtype=np.float32)
            full_values[:, 3] = [0.25, -0.5]
            full_values[:, 7] = [-0.75, 0.125]
            np.savez_compressed(
                result_dir / "acf_prior.npz",
                schema_version=np.asarray(1, dtype=np.int64),
                acf_values=full_values[:, [3, 7]],
                acf_full_values=full_values,
                lags=np.asarray([3, 7], dtype=np.int64),
                seq_len=np.asarray(4, dtype=np.int64),
                pred_len=np.asarray(2, dtype=np.int64),
                channels=np.asarray(2, dtype=np.int64),
                acf_source=np.asarray("raw"),
                acf_temperature=np.asarray(4.0, dtype=np.float64),
            )
            config["mamba_acf_prior_sha256"] = _sha256(
                result_dir / "acf_prior.npz"
            )
            self.assertEqual(
                _validate_acf_prior_artifact(config, result_dir, job, strict=True),
                [],
            )

    def test_confirmation_launch_requires_authorized_status_and_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "plan.json"
            plan_path.write_text("{}", encoding="utf-8")
            draft = {
                "confirmation_contract": {},
                "statistical_gate": {},
                "status": "draft_requires_prelaunch_review",
                "execution": {
                    "authorized": False,
                    "status": "draft_requires_prelaunch_review",
                    "required_before_launch": ["human_review"],
                },
            }
            with self.assertRaises(ValueError):
                _validate_launch_authorization(draft, plan_path)

            approved = {
                **draft,
                "status": "approved_for_launch",
                "execution": {"authorized": True, "status": "approved_for_launch"},
            }
            sidecar = root / "launch.json"
            sidecar.write_text(json.dumps({
                "schema_version": 1,
                "kind": "mamba_acf_confirmation_launch_authorization",
                "plan_path": str(plan_path),
                "plan_sha256": "bad",
                "plan_status": "approved_for_launch",
                "authorized": True,
                "approval": {
                    "approved": True,
                    "approved_by": "reviewer",
                    "approved_at_utc": "2026-09-01T00:00:00Z",
                },
                "prelaunch_gate_evidence_hashes": {"human_review": "0" * 64},
            }), encoding="utf-8")
            with self.assertRaises(ValueError):
                _validate_launch_authorization(approved, plan_path, sidecar)

            sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
            sidecar_payload["plan_sha256"] = _sha256(plan_path)
            sidecar.write_text(json.dumps(sidecar_payload), encoding="utf-8")
            binding = _validate_launch_authorization(approved, plan_path, sidecar)
            self.assertEqual(binding["plan_sha256"], sidecar_payload["plan_sha256"])
            self.assertEqual(binding["sidecar_path"], str(sidecar.resolve()))

            self.assertEqual(
                _validate_launch_authorization(approved, plan_path, enforce=False)["authorized"],
                True,
            )

    def test_confirmation_contract_schema_cannot_fall_back_to_exploratory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            base = {
                "version": 2,
                "confirmation_contract": {},
                "statistical_gate": {},
                "status": "draft_requires_prelaunch_review",
                "execution": {
                    "authorized": False,
                    "status": "draft_requires_prelaunch_review",
                },
            }
            path.write_text(json.dumps(base), encoding="utf-8")

            missing_execution = {key: value for key, value in base.items() if key != "execution"}
            with self.assertRaisesRegex(ValueError, "requires an execution object"):
                _validate_launch_authorization(missing_execution, path, enforce=False)
            self.assertTrue(any(
                "requires an execution object" in error
                for error in _validate_plan(missing_execution)
            ))

            partial = {key: value for key, value in base.items() if key != "statistical_gate"}
            with self.assertRaisesRegex(ValueError, "statistical_gate"):
                _validate_launch_authorization(partial, path, enforce=False)
            self.assertTrue(any(
                "statistical_gate" in error for error in _validate_plan(partial)
            ))

            missing_status = {**base, "execution": {"authorized": False, "status": "draft_requires_prelaunch_review"}}
            missing_status.pop("status")
            with self.assertRaisesRegex(ValueError, "plan status"):
                _validate_launch_authorization(missing_status, path, enforce=False)
            self.assertTrue(any(
                "status must be a non-empty" in error
                for error in _validate_plan(missing_status)
            ))

            whitespace_status = {
                **base,
                "status": " approved_for_launch ",
                "execution": {"authorized": True, "status": " approved_for_launch "},
            }
            with self.assertRaisesRegex(ValueError, "surrounding whitespace"):
                _validate_launch_authorization(whitespace_status, path, enforce=False)
            self.assertTrue(any(
                "surrounding whitespace" in error
                for error in _validate_plan(whitespace_status)
            ))

    def test_confirmation_report_binding_rejects_missing_or_tampered_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "plan.json"
            plan = {
                "confirmation_contract": {},
                "statistical_gate": {},
                "status": "approved_for_launch",
                "execution": {
                    "authorized": True,
                    "status": "approved_for_launch",
                    "required_before_launch": ["human_review"],
                },
            }
            plan_path.write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")
            sidecar_path = root / "launch.json"
            sidecar_path.write_text(json.dumps({
                "schema_version": 1,
                "kind": "mamba_acf_confirmation_launch_authorization",
                "plan_path": str(plan_path),
                "plan_sha256": _sha256(plan_path),
                "plan_status": "approved_for_launch",
                "authorized": True,
                "approval": {
                    "approved": True,
                    "approved_by": "reviewer",
                    "approved_at_utc": "2026-09-01T00:00:00Z",
                },
                "prelaunch_gate_evidence_hashes": {"human_review": "0" * 64},
            }, sort_keys=True), encoding="utf-8")
            binding = _validate_launch_authorization(plan, plan_path, sidecar_path)
            phase_root = root / "phase"
            archive_dir = phase_root / "provenance"
            archive_dir.mkdir(parents=True)
            archive = archive_dir / "launch_authorization.json"
            archive.write_bytes(sidecar_path.read_bytes())
            provenance = {
                "launch_authorization": binding,
                "launch_authorization_copy": str(archive),
                "launch_authorization_copy_sha256": _sha256(archive),
                "launch_authorization_binding_sha256": _canonical_hash(binding),
            }
            self.assertIsNotNone(
                _validate_confirmation_launch_binding(
                    plan, plan_path, provenance, phase_root
                )
            )

            archive.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sidecar hash"):
                _validate_confirmation_launch_binding(
                    plan, plan_path, provenance, phase_root
                )

            archive.unlink()
            with self.assertRaisesRegex(ValueError, "sidecar"):
                _validate_confirmation_launch_binding(
                    plan, plan_path, provenance, phase_root
                )

    def test_registered_job_counts_are_recomputed_for_phases_and_totals(self):
        phases = [
            {
                "name": "selection_phase",
                "datasets": ["ETTh1"],
                "models": ["DLinear"],
                "features": "M",
                "horizons": [96],
                "seeds": [1, 2],
            },
            {
                "name": "confirmatory_phase",
                "datasets": ["ETTh1"],
                "models": ["DLinear"],
                "features": "M",
                "horizons": [96],
                "seeds": [1],
            },
        ]
        plan = {
            "global": {"features": "M"},
            "phases": phases,
            "registered_job_counts": {
                "selection_phase": 2,
                "confirmatory_phase": 1,
                "selection_total": 2,
                "confirmation_total": 1,
                "training_total": 3,
            },
        }
        self.assertEqual(_registered_job_count_errors(plan, phases), [])
        plan["registered_job_counts"]["training_total"] = 4
        errors = _registered_job_count_errors(plan, phases)
        self.assertTrue(any("training_total" in error for error in errors))
        plan["registered_job_counts"].pop("confirmatory_phase")
        errors = _registered_job_count_errors(plan, phases)
        self.assertTrue(any("confirmatory_phase" in error for error in errors))
        plan["registered_job_counts"]["confirmatory_phase"] = 1
        plan["registered_job_counts"]["unexpected_total"] = 3
        errors = _registered_job_count_errors(plan, phases)
        self.assertTrue(any("unknown key" in error for error in errors))

    def test_frozen_confirmation_count_matrix_is_exact(self):
        plan_path = (
            Path(__file__).resolve().parents[1]
            / "docs/research/mamba_acf_confirmation_plan.json"
        )
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        self.assertEqual(_validate_plan(plan), [])
        self.assertEqual(plan["registered_job_counts"]["training_total"], 884)

        plan["registered_job_counts"]["training_total"] = 883
        errors = _validate_plan(plan)
        self.assertTrue(any("training_total" in error for error in errors))

        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan.pop("registered_job_counts")
        errors = _validate_plan(plan)
        self.assertTrue(any("requires registered_job_counts" in error for error in errors))

        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan["registered_job_counts"].pop("selection_total")
        errors = _validate_plan(plan)
        self.assertTrue(any("missing aggregate key selection_total" in error for error in errors))

    def test_accuracy_claim_contract_rejects_gate_drift(self):
        plan_path = (
            Path(__file__).resolve().parents[1]
            / "docs/research/mamba_acf_confirmation_plan.json"
        )
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        self.assertEqual(_validate_plan(plan), [])

        drifted = json.loads(plan_path.read_text(encoding="utf-8"))
        drifted["statistical_gate"]["primary_comparisons"].append(
            "confirmatory_autocon_memory_relation_M:MambaACFAutoCon::confirmatory_baselines_M:Mamba1"
        )
        errors = _validate_plan(drifted)
        self.assertTrue(any("descriptive baseline Mamba1 appears" in error for error in errors))

        missing = json.loads(plan_path.read_text(encoding="utf-8"))
        missing["statistical_gate"]["primary_comparisons"] = [
            item for item in missing["statistical_gate"]["primary_comparisons"]
            if ":Mamba2" not in item
        ]
        errors = _validate_plan(missing)
        self.assertTrue(any("Mamba2 is absent" in error for error in errors))

    def test_strict_memory_lag_contract_separates_peak_and_placebo(self):
        peak_job = {"phase": "peak", "model": "MambaACF"}
        peak_config = {
            "mamba_use_acf_memory": True,
            "mamba_causal_embedding": True,
            "mamba_bidirectional": False,
            "AutoCon_acf_source": "raw",
            "mamba_acf_effective_lags": [24, 168],
            "mamba_acf_train_peak_lags": [24, 168, 336],
            "mamba_acf_lag_mode": "train_peaks_available",
            "mamba_acf_effective_lag_source": "train_peaks_available",
            "mamba_acf_effective_lag_fallback": False,
            "mamba_acf_prior_max_lag": 720,
        }
        plan = {
            "acf_falsification_contract": {
                "fixed_placebo_arm": "placebo",
                "placebo_lags": [5, 11, 17],
            }
        }
        self.assertEqual(
            _validate_memory_lag_contract(peak_config, peak_job, plan, strict=True), []
        )
        peak_config["mamba_acf_effective_lag_fallback"] = True
        errors = _validate_memory_lag_contract(peak_config, peak_job, plan, strict=True)
        self.assertIn("config_mamba_acf_effective_lag_fallback_forbidden", errors)

        placebo_job = {"phase": "placebo", "model": "MambaACF"}
        placebo_config = {
            "mamba_use_acf_memory": True,
            "mamba_causal_embedding": True,
            "mamba_bidirectional": False,
            "AutoCon_acf_source": "raw",
            "mamba_acf_effective_lags": [5, 11, 17],
            "mamba_acf_lag_mode": "fixed",
            "mamba_acf_effective_lag_source": "fixed_configured",
            "mamba_acf_effective_lag_fallback": False,
            "mamba_acf_prior_max_lag": 720,
        }
        self.assertEqual(
            _validate_memory_lag_contract(placebo_config, placebo_job, plan, strict=True), []
        )

        residual_config = dict(peak_config)
        residual_config["AutoCon_acf_source"] = "residual"
        self.assertIn(
            "config_acf_source_raw_required_for_strict_causality",
            _validate_memory_lag_contract(residual_config, peak_job, plan, strict=True),
        )

    def test_exploratory_plan_remains_compatible_without_launch_sidecar(self):
        plan = {"status": "exploratory_screen_not_confirmatory"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(plan), encoding="utf-8")
            self.assertIsNone(_validate_launch_authorization(plan, path))

    def test_exploratory_execution_metadata_may_omit_status(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "docs/research/mamba_acf_variable_memory_relation_screen_plan.json"
        )
        plan = json.loads(path.read_text(encoding="utf-8"))
        binding = _validate_launch_authorization(plan, path, enforce=False)
        self.assertFalse(binding["authorized"])
        self.assertEqual(binding["plan_status"], plan["status"])
        self.assertEqual(binding["required_prelaunch_gates"], plan["execution"]["required_before_launch"])

    def test_zero_autocon_lambda_requires_explicit_exploratory_opt_in(self):
        phase = {
            "name": "null_control",
            "datasets": ["ETTh1"],
            "models": ["MambaACFAutoCon"],
            "seeds": [2021],
            "horizons": [96],
            "features": "S",
            "mamba_acf_relation_loss": True,
            "autocon_lambda": 0.0,
            "output_root": "artifacts/paper_final/null_control",
        }
        plan = {
            "version": 2,
            "status": "exploratory_null_control",
            "global": {"features": "S", "seq_len": 96},
            "phases": [phase],
        }
        errors = _validate_plan(plan)
        self.assertTrue(any("finite positive" in error for error in errors))

        phase["allow_zero_autocon_lambda"] = True
        self.assertEqual(_validate_plan(plan), [])
        jobs = _job_specs(plan, phase, {})
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["autocon_lambda"], 0.0)
        roots = {
            "checkpoints": Path("/tmp/null-control-checkpoints"),
            "results": Path("/tmp/null-control-results"),
            "test": Path("/tmp/null-control-test"),
            "result_log": Path("/tmp/null-control-results.txt"),
        }
        _, command = _command(jobs[0], roots)
        index = command.index("--AutoCon_lambda")
        self.assertEqual(command[index + 1], "0.0")

    def test_zero_autocon_lambda_opt_in_is_null_control_only(self):
        phase = {
            "name": "null_control",
            "datasets": ["ETTh1"],
            "models": ["MambaACFAutoCon"],
            "seeds": [2021],
            "horizons": [96],
            "features": "S",
            "mamba_acf_relation_loss": True,
            "autocon_lambda": 0.03,
            "allow_zero_autocon_lambda": True,
            "output_root": "artifacts/paper_final/null_control",
        }
        plan = {
            "version": 2,
            "status": "exploratory_null_control",
            "global": {"features": "S", "seq_len": 96},
            "phases": [phase],
        }
        errors = _validate_plan(plan)
        self.assertTrue(any("explicit zero lambda" in error for error in errors))

        phase["autocon_lambda"] = "selected"
        errors = _validate_plan(plan)
        self.assertTrue(any("cannot be used with selected lambdas" in error for error in errors))

        plan["confirmation_contract"] = {}
        phase["autocon_lambda"] = 0.0
        errors = _validate_plan(plan)
        self.assertTrue(any("exploratory-only" in error for error in errors))
        with self.assertRaisesRegex(ValueError, "exploratory-only"):
            _job_specs(plan, phase, {})

        plan.pop("confirmation_contract")
        plan["status"] = "approved_for_launch"
        errors = _validate_plan(plan)
        self.assertTrue(any("exploratory or screen plan status" in error for error in errors))
        with self.assertRaisesRegex(ValueError, "exploratory or screen plan status"):
            _job_specs(plan, phase, {})

    def test_data_fingerprint_contract_checks_registered_path_and_hash(self):
        valid = _validate_data_fingerprints({
            "ETTh1": {
                "relative_path": "dataset/ETT-small/ETTh1.csv",
                "channels_including_target": 7,
                "frequency": "h",
                "time_column": "date",
                "target": "OT",
                "sha256": "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066",
            }
        })
        self.assertEqual(valid, [])
        errors = _validate_data_fingerprints({
            "ETTh1": {
                "relative_path": "dataset/ETT-small/other.csv",
                "channels_including_target": 7,
                "frequency": "h",
                "time_column": "date",
                "target": "OT",
                "sha256": "0" * 64,
            }
        })
        self.assertTrue(any("relative_path" in error for error in errors))
        self.assertTrue(any("sha256 mismatch" in error for error in errors))

    def test_graph_screen_plan_rejects_tampered_graph_fingerprint(self):
        plan_path = Path(
            "docs/research/metro_pemsbay_graph_spatial_mambatsv3_autocon_screen_plan_20260927.json"
        )
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        self.assertEqual(_validate_plan(plan), [])
        plan["data_fingerprints"]["metr_la"]["directed_edge_count"] = 1721
        errors = _validate_plan(plan)
        self.assertTrue(any("directed_edge_count mismatch" in error for error in errors))

    def _load_scale_conditioning_plan(self):
        plan_path = (
            Path(__file__).resolve().parents[1]
            / "docs/research/mamba_acf_scale_conditioning_screen_plan_20260902.json"
        )
        return json.loads(plan_path.read_text(encoding="utf-8"))

    def test_scale_plan_requires_structured_pairing_and_accuracy_contracts(self):
        plan = self._load_scale_conditioning_plan()
        self.assertEqual(_validate_plan(plan), [])

        plan["seed_policy"]["pairing_unit"] = "same dataset, horizon, and seed"
        errors = _validate_plan(plan)
        self.assertTrue(any("seed_policy.pairing_unit must be an object" in error for error in errors))

        plan = self._load_scale_conditioning_plan()
        plan["comparison_contract"]["accuracy_gate"] = "no test selection"
        errors = _validate_plan(plan)
        self.assertTrue(any("accuracy_gate must be a structured object" in error for error in errors))

        plan = self._load_scale_conditioning_plan()
        plan["comparison_contract"]["accuracy_gate"]["test_metric_for_selection"] = True
        errors = _validate_plan(plan)
        self.assertTrue(any("test_metric_for_selection must be false" in error for error in errors))

    def test_scale_plan_rejects_fingerprint_frequency_tampering(self):
        plan = self._load_scale_conditioning_plan()
        plan["data_fingerprints"]["ETTh1"]["frequency"] = "t"
        errors = _validate_plan(plan)
        self.assertTrue(any("frequency mismatch" in error for error in errors))

        plan = self._load_scale_conditioning_plan()
        plan["data_fingerprints"]["ETTh1"].pop("frequency")
        errors = _validate_plan(plan)
        self.assertTrue(any("frequency is required" in error for error in errors))

    def test_scale_plan_numeric_and_pair_contracts_reject_tampering(self):
        plan = self._load_scale_conditioning_plan()
        plan["global"]["batch_size"] = 0
        errors = _validate_plan(plan)
        self.assertTrue(any("batch_size must be a positive integer" in error for error in errors))

        plan = self._load_scale_conditioning_plan()
        plan["phases"][2]["horizons"] = [0]
        errors = _validate_plan(plan)
        self.assertTrue(any("horizons[0] must be a positive integer" in error for error in errors))

        plan = self._load_scale_conditioning_plan()
        plan["phases"][2]["horizons"] = [192]
        errors = _validate_plan(plan)
        self.assertTrue(any("horizon declaration" in error for error in errors))

        plan = self._load_scale_conditioning_plan()
        plan["phases"][2]["epochs"] = 19
        errors = _validate_plan(plan)
        self.assertTrue(any("differ outside the allowed flag" in error for error in errors))

        plan = self._load_scale_conditioning_plan()
        plan["phases"][2]["datasets"] = ["ETTh1"]
        errors = _validate_plan(plan)
        self.assertTrue(any("paired scale phases differ in datasets" in error for error in errors))

        plan = self._load_scale_conditioning_plan()
        plan["phases"][2]["seeds"] = [2021, 2022]
        errors = _validate_plan(plan)
        self.assertTrue(any("paired scale phases differ in seeds" in error for error in errors))

        plan = self._load_scale_conditioning_plan()
        plan["paired_phase_contract"]["allowed_difference"] = [
            "mamba_acf_scale_conditioning", "epochs"
        ]
        errors = _validate_plan(plan)
        self.assertTrue(any("allowed_difference" in error for error in errors))

        plan = self._load_scale_conditioning_plan()
        plan["phases"][2]["autocon_acf_source"] = "residual"
        errors = _validate_plan(plan)
        self.assertTrue(any("requires autocon_acf_source=raw" in error for error in errors))

    def test_scale_plan_null_phase_overrides_inherit_global_values(self):
        plan = self._load_scale_conditioning_plan()
        for phase in plan["phases"]:
            phase["batch_size"] = None
            phase["patience"] = None
            phase["mamba_acf_channel_gate"] = None
        self.assertEqual(_validate_plan(plan), [])

    def _load_variable_repr_fusion_plan(self):
        plan_path = (
            Path(__file__).resolve().parents[1]
            / "docs/research/mamba_variable_repr_fusion_screen_plan_20260903.json"
        )
        return json.loads(plan_path.read_text(encoding="utf-8"))

    def test_variable_repr_fusion_plan_enforces_one_factor_pairing(self):
        plan = self._load_variable_repr_fusion_plan()
        self.assertEqual(_validate_plan(plan), [])

        plan = self._load_variable_repr_fusion_plan()
        plan["phases"][1]["epochs"] = 9
        errors = _validate_plan(plan)
        self.assertTrue(any("outside the allowed flag" in error for error in errors))

        plan = self._load_variable_repr_fusion_plan()
        plan["paired_variable_repr_fusion_contract"]["allowed_difference"] = [
            "mamba_variable_repr_fusion", "epochs"
        ]
        errors = _validate_plan(plan)
        self.assertTrue(any("allowed_difference" in error for error in errors))

    def test_plan_validator_covers_patch_mixer_contract(self):
        valid = _validate_mamba_options(
            "mamba_patch", {
                "seq_len": 96,
                "mamba_patch_mixer": True,
                "mamba_patch_mixer_depth": 2,
                "mamba_patch_len": 16,
                "mamba_patch_stride": 8,
                "mamba_patch_order": "channel_major",
                "mamba_patch_permutation": True,
            },
            ["MambaACF"],
        )
        self.assertEqual(valid, [])
        errors = _validate_mamba_options(
            "mamba_patch", {
                "seq_len": 96,
                "mamba_channel_independent": True,
                "mamba_patch_mixer": True,
                "mamba_patch_len": 128,
                "mamba_patch_permutation": True,
                "mamba_patch_order": "bad",
            },
            ["MambaACF"],
        )
        self.assertTrue(any("patch_len" in error for error in errors))
        self.assertTrue(any("channel_independent" in error for error in errors))
        self.assertTrue(any("patch_order" in error for error in errors))
        mixed_errors = _validate_mamba_options(
            "mixed_patch", {
                "seq_len": 96, "mamba_patch_mixer": True,
            }, ["MambaACF", "Mamba2"],
        )
        self.assertTrue(any("only MambaACF" in error for error in mixed_errors))

    def test_plan_validator_covers_causal_token_stride_contract(self):
        valid = _validate_mamba_options(
            "mamba_stride", {
                "seq_len": 96, "mamba_token_stride": 8,
                "mamba_scales": [1, 2],
            }, ["MambaACF1"],
        )
        self.assertEqual(valid, [])
        invalid_length = _validate_mamba_options(
            "mamba_stride_length", {
                "seq_len": 96, "mamba_token_stride": 97,
            }, ["MambaACF"],
        )
        self.assertTrue(any("cannot exceed seq_len" in error for error in invalid_length))
        invalid_scale = _validate_mamba_options(
            "mamba_stride_scale", {
                "seq_len": 96, "mamba_token_stride": 64,
                "mamba_scales": [1, 3],
            }, ["MambaACF"],
        )
        self.assertTrue(any("base tokens" in error for error in invalid_scale))
        mixed = _validate_mamba_options(
            "mamba_stride_mixed", {"seq_len": 96, "mamba_token_stride": 8},
            ["MambaACF", "Mamba2"],
        )
        self.assertTrue(any("only MambaACF" in error for error in mixed))

        job = {
            "phase": "mamba_stride", "dataset": "ETTh1", "model": "MambaACF1",
            "pred_len": 96, "seed": 2021, "autocon_lambda": None,
            "config": {"features": "M", "enc_in": 7, "dec_in": 7, "c_out": 7,
                        "mamba_token_stride": 8},
        }
        roots = {
            "checkpoints": Path("/tmp/stride-checkpoints"),
            "results": Path("/tmp/stride-results"),
            "test": Path("/tmp/stride-test"),
            "result_log": Path("/tmp/stride-results.txt"),
        }
        _, command = _command(job, roots)
        self.assertEqual(command[command.index("--mamba_token_stride") + 1], "8")

    def test_command_wraps_scalar_mamba_lists(self):
        """Validated scalar list shorthands must remain launchable."""
        job = {
            "phase": "scalar_lists", "dataset": "ETTh1", "model": "MambaACF1",
            "pred_len": 96, "seed": 2021, "autocon_lambda": None,
            "config": {
                "features": "M", "enc_in": 7, "dec_in": 7, "c_out": 7,
                "mamba_scales": 1, "mamba_acf_lags": 24,
                "mamba_collect_eval_acf_diagnostics": True,
            },
        }
        roots = {
            "checkpoints": Path("/tmp/scalar-list-checkpoints"),
            "results": Path("/tmp/scalar-list-results"),
            "test": Path("/tmp/scalar-list-test"),
            "result_log": Path("/tmp/scalar-list-results.txt"),
        }
        _, command = _command(job, roots)
        scale_index = command.index("--mamba_scales")
        lag_index = command.index("--mamba_acf_lags")
        self.assertEqual(command[scale_index + 1:scale_index + 2], ["1"])
        self.assertEqual(command[lag_index + 1:lag_index + 2], ["24"])
        diagnostics_index = command.index("--mamba_collect_eval_acf_diagnostics")
        self.assertEqual(command[diagnostics_index + 1], "true")

    def test_plan_validator_covers_strict_representation_causality_contract(self):
        config = {
            "seq_len": 96,
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
        }
        errors = _validate_mamba_options(
            "strict_representation", config, ["MambaACFAutoCon"]
        )
        self.assertTrue(any("window_full_normalization_forbidden" in error for error in errors))
        errors = _validate_mamba_options(
            "strict_representation",
            {**config, "mamba_scales": [1, 2]},
            ["MambaACFAutoCon"],
        )
        self.assertTrue(any("mamba_scales_must_be_single_scale_one" in error for error in errors))
        mixed = _validate_mamba_options(
            "strict_representation", config, ["MambaACFAutoCon", "Mamba2"]
        )
        self.assertTrue(any("only MambaACF" in error for error in mixed))

        job = {
            "phase": "strict_representation", "dataset": "ETTh1",
            "model": "MambaACFAutoCon", "pred_len": 96, "seed": 2021,
            "autocon_lambda": 0.01,
            "config": {
                "features": "M", "enc_in": 7, "dec_in": 7, "c_out": 7,
                **config,
            },
        }
        roots = {
            "checkpoints": Path("/tmp/strict-representation-checkpoints"),
            "results": Path("/tmp/strict-representation-results"),
            "test": Path("/tmp/strict-representation-test"),
            "result_log": Path("/tmp/strict-representation-results.txt"),
        }
        _, command = _command(job, roots)
        self.assertIn("--mamba_require_strict_representation_causality", command)

    def test_result_validator_covers_strict_representation_causality_contract(self):
        config = {
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
        }
        self.assertEqual(
            _validate_strict_representation_causality_contract(
                config, "MambaACFAutoCon"
            ),
            [
                "config_strict_representation_causality_"
                "window_full_normalization_forbidden"
            ],
        )
        self.assertEqual(
            _validate_strict_representation_causality_contract(
                {**config, "mamba_token_stride": 2}, "MambaACFAutoCon"
            ),
            [
                "config_strict_representation_causality_"
                "mamba_token_stride_must_be_one",
                "config_strict_representation_causality_"
                "window_full_normalization_forbidden",
            ],
        )
        self.assertEqual(
            _validate_strict_representation_causality_contract(config, "Mamba2"),
            ["config_strict_representation_causality_model_unsupported"],
        )

    def test_plan_validator_covers_relation_overlap_control(self):
        valid = _validate_mamba_options(
            "relation_no_overlap", {
                "seq_len": 96,
                "mamba_acf_relation_loss": True,
                "mamba_acf_relation_lag_mask": True,
                "mamba_acf_relation_min_lag": 96,
            }, ["MambaACFAutoCon"],
        )
        self.assertEqual(valid, [])
        independent_min_lag = _validate_mamba_options(
            "relation_overlap_invalid", {
                "seq_len": 96,
                "mamba_acf_relation_loss": True,
                "mamba_acf_relation_min_lag": 96,
            }, ["MambaACFAutoCon"],
        )
        self.assertEqual(independent_min_lag, [])
        disabled_relation = _validate_mamba_options(
            "relation_min_lag_without_objective", {
                "seq_len": 96,
                "mamba_acf_relation_loss": False,
                "mamba_acf_relation_min_lag": 24,
            }, ["PatchTST"],
        )
        self.assertTrue(any(
            "requires mamba_acf_relation_loss=true" in error
            for error in disabled_relation
        ))
        invalid_max_lag = _validate_mamba_options(
            "relation_empty", {
                "seq_len": 96,
                "mamba_acf_relation_loss": True,
                "mamba_acf_relation_min_lag": 721,
                "mamba_acf_max_lag": 720,
            }, ["MambaACFAutoCon"],
        )
        self.assertTrue(any("cannot exceed mamba_acf_max_lag" in error for error in invalid_max_lag))

        invalid_mode = _validate_mamba_options(
            "relation_mode_invalid", {
                "seq_len": 96,
                "mamba_acf_relation_loss": True,
                "mamba_acf_relation_target_mode": "unknown",
            }, ["MambaACFAutoCon"],
        )
        self.assertTrue(any("target_mode" in error for error in invalid_mode))
        invalid_threshold = _validate_mamba_options(
            "relation_threshold_invalid", {
                "seq_len": 96,
                "mamba_acf_relation_loss": True,
                "mamba_acf_relation_soft_min_abs": 1.1,
            }, ["MambaACFAutoCon"],
        )
        self.assertTrue(any("soft_min_abs" in error for error in invalid_threshold))

        malformed_numeric = _validate_mamba_options(
            "relation_numeric_invalid",
            {
                "seq_len": 96,
                "mamba_acf_relation_loss": True,
                "mamba_acf_relation_lag_tolerance": 1.5,
                "mamba_acf_relation_min_lag": np.bool_(True),
                "mamba_acf_min_abs": float("nan"),
                "mamba_acf_confidence_z": float("inf"),
                "AutoCon_temperature": float("nan"),
            },
            ["MambaACFAutoCon"],
        )
        self.assertTrue(any("lag_tolerance" in error for error in malformed_numeric))
        self.assertTrue(any("relation_min_lag" in error for error in malformed_numeric))
        self.assertTrue(any("mamba_acf_min_abs" in error for error in malformed_numeric))
        self.assertTrue(any("mamba_acf_confidence_z" in error for error in malformed_numeric))
        self.assertTrue(any("AutoCon_temperature" in error for error in malformed_numeric))

        job = {
            "phase": "relation_no_overlap", "dataset": "ETTh1",
            "model": "MambaACFAutoCon", "pred_len": 96, "seed": 2021,
            "autocon_lambda": 0.01,
            "config": {
                "features": "M", "enc_in": 7, "dec_in": 7, "c_out": 7,
                "AutoCon": True, "mamba_acf_relation_loss": True,
                "mamba_acf_relation_lag_mask": True,
                "mamba_acf_relation_min_lag": 96,
            },
        }
        roots = {
            "checkpoints": Path("/tmp/relation-checkpoints"),
            "results": Path("/tmp/relation-results"),
            "test": Path("/tmp/relation-test"),
            "result_log": Path("/tmp/relation-results.txt"),
        }
        _, command = _command(job, roots)
        self.assertEqual(
            command[command.index("--mamba_acf_relation_min_lag") + 1], "96"
        )
        self.assertEqual(
            command[command.index("--mamba_acf_relation_target_mode") + 1],
            "hard_selected",
        )

    def test_plan_validator_rejects_strided_local_autocon_relation(self):
        base = {
            "seq_len": 96,
            "mamba_token_stride": 8,
            "mamba_acf_relation_loss": True,
        }
        local_errors = _validate_mamba_options(
            "strided_local_relation",
            {**base, "mamba_acf_relation_scope": "local"},
            ["MambaACFAutoCon"],
        )
        self.assertTrue(
            any(
                "mamba_token_stride=8" in error
                and "relation_scope=local" in error
                for error in local_errors
            )
        )

        both_errors = _validate_mamba_options(
            "strided_both_relation", base, ["MambaACF1AutoCon"]
        )
        self.assertTrue(
            any("mamba_token_stride=8" in error for error in both_errors)
        )

        self.assertEqual(
            _validate_mamba_options(
                "strided_global_relation",
                {**base, "mamba_acf_relation_scope": "global"},
                ["MambaACFAutoCon"],
            ),
            [],
        )
        self.assertEqual(
            _validate_mamba_options(
                "unit_local_relation",
                {
                    "seq_len": 96,
                    "mamba_token_stride": 1,
                    "mamba_acf_relation_loss": True,
                    "mamba_acf_relation_scope": "local",
                },
                ["MambaACFAutoCon"],
            ),
            [],
        )

    def test_plan_validator_covers_acf_channel_conditioning_contract(self):
        valid = _validate_mamba_options(
            "acf_channel", {
                "seq_len": 96,
                "mamba_variable_mixer": True,
                "mamba_acf_channel_conditioning": True,
            },
            ["MambaACF"],
        )
        self.assertEqual(valid, [])
        missing_mixer = _validate_mamba_options(
            "acf_without_mixer", {
                "seq_len": 96,
                "mamba_acf_channel_conditioning": True,
            },
            ["MambaACF"],
        )
        self.assertTrue(any("requires mamba_variable_mixer" in error for error in missing_mixer))
        mixed_models = _validate_mamba_options(
            "acf_mixed", {
                "seq_len": 96,
                "mamba_variable_mixer": True,
                "mamba_acf_channel_conditioning": True,
            },
            ["MambaACF", "Mamba2"],
        )
        self.assertTrue(any("only MambaACF" in error for error in mixed_models))

    def test_command_forwards_acf_channel_conditioning_flag(self):
        config = {
            "features": "M", "enc_in": 7, "dec_in": 7, "c_out": 7,
            "mamba_variable_mixer": True,
            "mamba_acf_channel_conditioning": True,
        }
        job = {
            "phase": "acf_channel", "dataset": "ETTh1", "model": "MambaACF",
            "pred_len": 96, "seed": 2021, "autocon_lambda": None,
            "config": config,
        }
        roots = {
            "checkpoints": Path("/tmp/acf-channel-checkpoints"),
            "results": Path("/tmp/acf-channel-results"),
            "test": Path("/tmp/acf-channel-test"),
            "result_log": Path("/tmp/acf-channel-results.txt"),
        }
        _, command = _command(job, roots)
        self.assertIn("--mamba_variable_mixer", command)
        self.assertIn("--mamba_acf_channel_conditioning", command)

    def test_command_forwards_data_dependent_mixer_once(self):
        job = {
            "phase": "channel_state", "dataset": "ETTh1",
            "model": "MambaACF1", "pred_len": 96, "seed": 2021,
            "autocon_lambda": None,
            "config": {
                "features": "M", "enc_in": 7, "dec_in": 7, "c_out": 7,
                "mamba_data_dependent_channel_mixer": True,
                "mamba_data_dependent_channel_mixer_rank": 4,
            },
        }
        roots = {
            "checkpoints": Path("/tmp/channel-state-checkpoints"),
            "results": Path("/tmp/channel-state-results"),
            "test": Path("/tmp/channel-state-test"),
            "result_log": Path("/tmp/channel-state-results.txt"),
        }
        _, command = _command(job, roots)
        self.assertEqual(command.count("--mamba_data_dependent_channel_mixer"), 1)
        index = command.index("--mamba_data_dependent_channel_mixer_rank")
        self.assertEqual(command[index + 1], "4")

    def test_command_forwards_variable_gdd_mixer_and_rank(self):
        job = {
            "phase": "variable_gdd", "dataset": "ETTh1",
            "model": "MambaACF", "pred_len": 96, "seed": 2021,
            "autocon_lambda": None,
            "config": {
                "features": "M", "enc_in": 7, "dec_in": 7, "c_out": 7,
                "mamba_variable_mixer": True,
                "mamba_variable_gdd_mixer": True,
                "mamba_variable_gdd_mixer_rank": 5,
                "mamba_variable_gdd_init_scale": 0.01,
            },
        }
        roots = {
            "checkpoints": Path("/tmp/variable-gdd-checkpoints"),
            "results": Path("/tmp/variable-gdd-results"),
            "test": Path("/tmp/variable-gdd-test"),
            "result_log": Path("/tmp/variable-gdd-results.txt"),
        }
        _, command = _command(job, roots)
        self.assertEqual(command.count("--mamba_variable_gdd_mixer"), 1)
        index = command.index("--mamba_variable_gdd_mixer_rank")
        self.assertEqual(command[index + 1], "5")
        index = command.index("--mamba_variable_gdd_init_scale")
        self.assertEqual(command[index + 1], "0.01")

    def test_plan_rejects_variable_gdd_without_variable_mixer(self):
        errors = _validate_mamba_options(
            "variable_gdd",
            {
                "seq_len": 96,
                "mamba_variable_gdd_mixer": True,
                "mamba_variable_gdd_mixer_rank": 4,
                "mamba_variable_mixer": False,
            },
            ["MambaACF"],
        )
        self.assertTrue(any("requires mamba_variable_mixer=true" in error for error in errors))

    def test_plan_rejects_variable_gdd_for_non_acf_phase(self):
        errors = _validate_mamba_options(
            "variable_gdd",
            {"mamba_variable_gdd_mixer": True, "mamba_variable_gdd_mixer_rank": 4},
            ["Mamba2"],
        )
        self.assertTrue(any("only MambaACF models" in error for error in errors))

    def test_fixed_repr_contract_allows_gdd_screen_with_repr_enabled(self):
        plan = {
            "global": {"features": "M"},
            "phases": [
                {
                    "name": "off", "datasets": ["ETTh1"],
                    "models": ["MambaACFAutoCon"], "seeds": [2021],
                    "horizons": [96], "mamba_variable_mixer": True,
                    "mamba_variable_repr_fusion": True,
                    "mamba_variable_gdd_mixer": False,
                },
                {
                    "name": "on", "datasets": ["ETTh1"],
                    "models": ["MambaACFAutoCon"], "seeds": [2021],
                    "horizons": [96], "mamba_variable_mixer": True,
                    "mamba_variable_repr_fusion": True,
                    "mamba_variable_gdd_mixer": True,
                    "mamba_variable_gdd_mixer_rank": 8,
                },
            ],
            "fixed_variable_repr_fusion_contract": {
                "phases": ["off", "on"], "state": "enabled",
            },
            "paired_variable_gdd_mixer_contract": {
                "off_phase": "off", "on_phase": "on",
                "allowed_difference": [
                    "mamba_variable_gdd_mixer",
                    "mamba_variable_gdd_mixer_rank",
                ],
            },
        }
        # The fixed contract is checked by the full plan validator; exercise
        # the dedicated GDD pairing helper here without dataset I/O.
        self.assertEqual(_validate_variable_gdd_mixer_contract(plan, plan["phases"]), [])

    def test_plan_validator_and_command_cover_local_acf_agreement(self):
        config = {
            "features": "M", "enc_in": 7, "dec_in": 7, "c_out": 7,
            "mamba_use_acf_memory": True,
            "mamba_acf_local_agreement": True,
            "mamba_acf_local_min_pairs": 32,
            "mamba_acf_lags": [24],
            "autocon_acf_source": "raw",
        }
        self.assertEqual(
            _validate_mamba_options("acf_local", {"seq_len": 96, **config}, ["MambaACF"]),
            [],
        )
        missing_memory = _validate_mamba_options(
            "acf_local_memory", {
                "seq_len": 96, "mamba_acf_local_agreement": True,
                "mamba_acf_lags": [24], "autocon_acf_source": "raw",
            }, ["MambaACF"],
        )
        self.assertTrue(any("requires mamba_use_acf_memory" in error for error in missing_memory))
        bad_source = _validate_mamba_options(
            "acf_local_source", {
                "seq_len": 96, "mamba_use_acf_memory": True,
                "mamba_acf_local_agreement": True, "mamba_acf_lags": [24],
                "autocon_acf_source": "residual",
            }, ["MambaACF"],
        )
        self.assertTrue(any("requires autocon_acf_source=raw" in error for error in bad_source))
        no_support = _validate_mamba_options(
            "acf_local_support", {
                "seq_len": 96, "mamba_use_acf_memory": True,
                "mamba_acf_local_agreement": True, "mamba_acf_local_min_pairs": 90,
                "mamba_acf_lags": [24], "autocon_acf_source": "raw",
            }, ["MambaACF"],
        )
        self.assertTrue(any("minimum pair support" in error for error in no_support))
        job = {
            "phase": "acf_local", "dataset": "ETTh1", "model": "MambaACF",
            "pred_len": 96, "seed": 2021, "autocon_lambda": None,
            "config": config,
        }
        roots = {
            "checkpoints": Path("/tmp/acf-local-checkpoints"),
            "results": Path("/tmp/acf-local-results"),
            "test": Path("/tmp/acf-local-test"),
            "result_log": Path("/tmp/acf-local-results.txt"),
        }
        _, command = _command(job, roots)
        self.assertIn("--mamba_acf_local_agreement", command)
        self.assertIn("--mamba_acf_local_min_pairs", command)
        self.assertEqual(
            command[command.index("--mamba_acf_local_min_pairs") + 1], "32"
        )

    def test_plan_validator_and_command_cover_reliability_interpolation(self):
        config = {
            "features": "M", "enc_in": 7, "dec_in": 7, "c_out": 7,
            "mamba_use_acf_memory": True,
            "mamba_acf_reliability_interpolation": True,
            "mamba_acf_step_reliability": False,
            "mamba_dynamic_acf": False,
            "autocon_acf_source": "raw",
        }
        self.assertEqual(
            _validate_mamba_options("acf_reliability", {"seq_len": 96, **config}, ["MambaACF"]),
            [],
        )
        cases = [
            ({**config, "mamba_use_acf_memory": False}, "requires mamba_use_acf_memory"),
            ({**config, "mamba_dynamic_acf": True}, "requires mamba_dynamic_acf=false"),
            ({**config, "autocon_acf_source": "residual"}, "requires autocon_acf_source=raw"),
            ({**config, "mamba_acf_step_reliability": True}, "incompatible"),
        ]
        for candidate, message in cases:
            errors = _validate_mamba_options(
                "acf_reliability_invalid", {"seq_len": 96, **candidate}, ["MambaACF"]
            )
            self.assertTrue(any(message in error for error in errors), errors)
        mixed = _validate_mamba_options(
            "acf_reliability_mixed", {"seq_len": 96, **config}, ["MambaACF", "Mamba2"]
        )
        self.assertTrue(any("only MambaACF" in error for error in mixed))
        job = {
            "phase": "acf_reliability", "dataset": "ETTh1", "model": "MambaACF",
            "pred_len": 96, "seed": 2021, "autocon_lambda": None,
            "config": config,
        }
        roots = {
            "checkpoints": Path("/tmp/acf-reliability-checkpoints"),
            "results": Path("/tmp/acf-reliability-results"),
            "test": Path("/tmp/acf-reliability-test"),
            "result_log": Path("/tmp/acf-reliability-results.txt"),
        }
        _, command = _command(job, roots)
        index = command.index("--mamba_acf_reliability_interpolation")
        self.assertEqual(command[index + 1], "true")

    def test_plan_validator_and_command_cover_stepwise_reliability(self):
        config = {
            "features": "M", "enc_in": 7, "dec_in": 7, "c_out": 7,
            "mamba_use_acf_memory": True,
            "mamba_dynamic_acf": False,
            "mamba_acf_step_reliability": True,
            "autocon_acf_source": "raw",
        }
        self.assertEqual(
            _validate_mamba_options(
                "acf_stepwise", {"seq_len": 96, **config}, ["MambaACF"]
            ),
            [],
        )
        job = {
            "phase": "acf_stepwise", "dataset": "ETTh1",
            "model": "MambaACF", "pred_len": 96, "seed": 2021,
            "autocon_lambda": None, "config": config,
        }
        roots = {
            "checkpoints": Path("/tmp/acf-stepwise-checkpoints"),
            "results": Path("/tmp/acf-stepwise-results"),
            "test": Path("/tmp/acf-stepwise-test"),
            "result_log": Path("/tmp/acf-stepwise-results.txt"),
        }
        _, command = _command(job, roots)
        index = command.index("--mamba_acf_step_reliability")
        self.assertEqual(command[index + 1], "true")

        malformed = _validate_mamba_options(
            "acf_stepwise_invalid",
            {"seq_len": 96, **config, "mamba_acf_step_reliability": "maybe"},
            ["MambaACF"],
        )
        self.assertTrue(
            any("mamba_acf_step_reliability" in error for error in malformed),
            malformed,
        )

    def test_plan_validator_and_command_cover_channel_lag_bank(self):
        config = {
            "features": "M", "enc_in": 7, "dec_in": 7, "c_out": 7,
            "mamba_use_acf_memory": True,
            "mamba_dynamic_acf": False,
            "mamba_acf_lags": [24, 168, 336],
            "mamba_acf_channel_lag_bank": True,
            "mamba_acf_channel_lag_top_k": 2,
        }
        self.assertEqual(
            _validate_mamba_options(
                "acf_channel_lag_bank", {"seq_len": 96, **config}, ["MambaACF"]
            ),
            [],
        )
        invalid_cases = [
            ({**config, "mamba_use_acf_memory": False}, "requires mamba_use_acf_memory=true"),
            ({**config, "mamba_dynamic_acf": True}, "requires static ACF memory"),
            ({**config, "mamba_acf_channel_lag_top_k": 4}, "cannot exceed mamba_acf_lags count"),
            ({**config, "mamba_acf_lags": [24, 24], "mamba_acf_channel_lag_top_k": 2}, "unique positive integers"),
        ]
        for candidate, message in invalid_cases:
            errors = _validate_mamba_options(
                "acf_channel_lag_bank_invalid",
                {"seq_len": 96, **candidate},
                ["MambaACF"],
            )
            self.assertTrue(any(message in error for error in errors), errors)
        mixed_errors = _validate_mamba_options(
            "acf_channel_lag_bank_mixed", {"seq_len": 96, **config},
            ["MambaACF", "Mamba2"],
        )
        self.assertTrue(any("only MambaACF" in error for error in mixed_errors))

        job = {
            "phase": "acf_channel_lag_bank", "dataset": "ETTh1",
            "model": "MambaACF", "pred_len": 96, "seed": 2021,
            "autocon_lambda": None, "config": config,
        }
        roots = {
            "checkpoints": Path("/tmp/acf-channel-lag-checkpoints"),
            "results": Path("/tmp/acf-channel-lag-results"),
            "test": Path("/tmp/acf-channel-lag-test"),
            "result_log": Path("/tmp/acf-channel-lag-results.txt"),
        }
        _, command = _command(job, roots)
        self.assertIn("--mamba_acf_channel_lag_bank", command)
        top_k_index = command.index("--mamba_acf_channel_lag_top_k")
        self.assertEqual(command[top_k_index + 1], "2")

    def test_exploratory_plan_may_use_exploratory_output_root(self):
        phase = {
            "name": "lag_bank_screen",
            "datasets": ["ETTh1"],
            "models": ["MambaACF"],
            "seeds": [2021],
            "horizons": [96],
            "features": "S",
            "output_root": "artifacts/exploratory/lag_bank_screen",
        }
        plan = {
            "version": 1,
            "status": "exploratory_lag_bank_screen",
            "global": {"features": "S", "seq_len": 96},
            "phases": [phase],
        }
        self.assertEqual(_validate_plan(plan), [])

        plan["status"] = "approved_for_launch"
        errors = _validate_plan(plan)
        self.assertTrue(any("artifacts/paper_final" in error for error in errors))

    def test_command_forwards_camel_case_acf_source_alias(self):
        job = {
            "phase": "camel_source", "dataset": "ETTh1", "model": "MambaACF",
            "pred_len": 96, "seed": 2021, "autocon_lambda": None,
            "config": {
                "features": "M", "enc_in": 7, "dec_in": 7, "c_out": 7,
                "mamba_use_acf_memory": True,
                "mamba_acf_local_agreement": True,
                "mamba_acf_lags": [24],
                "AutoCon_acf_source": "raw",
            },
        }
        roots = {
            "checkpoints": Path("/tmp/camel-source-checkpoints"),
            "results": Path("/tmp/camel-source-results"),
            "test": Path("/tmp/camel-source-test"),
            "result_log": Path("/tmp/camel-source-results.txt"),
        }
        _, command = _command(job, roots)
        self.assertEqual(
            command[command.index("--AutoCon_acf_source") + 1], "raw"
        )

    def test_command_prefers_lowercase_acf_source_when_aliases_conflict(self):
        job = {
            "phase": "canonical_source", "dataset": "ETTh1", "model": "MambaACF",
            "pred_len": 96, "seed": 2021, "autocon_lambda": None,
            "config": {
                "features": "M", "enc_in": 7, "dec_in": 7, "c_out": 7,
                "autocon_acf_source": "raw",
                "AutoCon_acf_source": "residual",
            },
        }
        roots = {
            "checkpoints": Path("/tmp/canonical-source-checkpoints"),
            "results": Path("/tmp/canonical-source-results"),
            "test": Path("/tmp/canonical-source-test"),
            "result_log": Path("/tmp/canonical-source-results.txt"),
        }
        _, command = _command(job, roots)
        self.assertEqual(
            command[command.index("--AutoCon_acf_source") + 1], "raw"
        )

    def test_command_forwards_camel_case_wnorm_alias(self):
        """Legacy plan spelling must not silently fall back to LastVal."""

        job = {
            "phase": "camel_wnorm", "dataset": "ETTh1", "model": "Mamba2",
            "pred_len": 96, "seed": 2021, "autocon_lambda": None,
            "config": {
                "features": "M", "enc_in": 7, "dec_in": 7, "c_out": 7,
                "AutoCon_wnorm": "MeanStd",
            },
        }
        roots = {
            "checkpoints": Path("/tmp/camel-wnorm-checkpoints"),
            "results": Path("/tmp/camel-wnorm-results"),
            "test": Path("/tmp/camel-wnorm-test"),
            "result_log": Path("/tmp/camel-wnorm-results.txt"),
        }
        _, command = _command(job, roots)
        self.assertEqual(
            command[command.index("--AutoCon_wnorm") + 1], "MeanStd"
        )

    def test_command_prefers_lowercase_wnorm_when_aliases_conflict(self):
        job = {
            "phase": "canonical_wnorm", "dataset": "ETTh1", "model": "Mamba2",
            "pred_len": 96, "seed": 2021, "autocon_lambda": None,
            "config": {
                "features": "M", "enc_in": 7, "dec_in": 7, "c_out": 7,
                "autocon_wnorm": "MeanStd", "AutoCon_wnorm": "LastVal",
            },
        }
        roots = {
            "checkpoints": Path("/tmp/canonical-wnorm-checkpoints"),
            "results": Path("/tmp/canonical-wnorm-results"),
            "test": Path("/tmp/canonical-wnorm-test"),
            "result_log": Path("/tmp/canonical-wnorm-results.txt"),
        }
        _, command = _command(job, roots)
        self.assertEqual(
            command[command.index("--AutoCon_wnorm") + 1], "MeanStd"
        )

    def test_command_forwards_acf_representation_fusion_flag(self):
        config = {
            "features": "M", "enc_in": 7, "dec_in": 7, "c_out": 7,
            "mamba_acf_repr_fusion": True,
            "mamba_acf_relation_loss": True,
        }
        self.assertEqual(
            _validate_mamba_options(
                "acf_repr", {"seq_len": 96, **config}, ["MambaACFAutoCon"]
            ),
            [],
        )
        job = {
            "phase": "acf_repr", "dataset": "ETTh1", "model": "MambaACFAutoCon",
            "pred_len": 96, "seed": 2021, "autocon_lambda": 0.03,
            "config": config,
        }
        roots = {
            "checkpoints": Path("/tmp/acf-repr-checkpoints"),
            "results": Path("/tmp/acf-repr-results"),
            "test": Path("/tmp/acf-repr-test"),
            "result_log": Path("/tmp/acf-repr-results.txt"),
        }
        _, command = _command(job, roots)
        self.assertIn("--mamba_acf_repr_fusion", command)

    def test_variable_representation_fusion_requires_variable_autocon_and_forwards(self):
        config = {
            "features": "M", "enc_in": 7, "dec_in": 7, "c_out": 7,
            "mamba_variable_repr_fusion": True,
            "mamba_variable_mixer": True,
            "mamba_acf_relation_loss": True,
        }
        self.assertEqual(
            _validate_mamba_options(
                "variable_repr", {"seq_len": 96, **config},
                ["MambaACFAutoCon"],
            ),
            [],
        )
        missing_mixer = _validate_mamba_options(
            "variable_repr_missing", {
                "seq_len": 96, **config, "mamba_variable_mixer": False,
            }, ["MambaACFAutoCon"],
        )
        self.assertTrue(any("requires mamba_variable_mixer" in error for error in missing_mixer))
        wrong_model = _validate_mamba_options(
            "variable_repr_wrong", {"seq_len": 96, **config}, ["Mamba2AutoCon"],
        )
        self.assertTrue(any("only MambaACF AutoCon" in error for error in wrong_model))
        job = {
            "phase": "variable_repr", "dataset": "ETTh1",
            "model": "MambaACFAutoCon", "pred_len": 96, "seed": 2021,
            "autocon_lambda": 0.03, "config": config,
        }
        roots = {
            "checkpoints": Path("/tmp/variable-repr-checkpoints"),
            "results": Path("/tmp/variable-repr-results"),
            "test": Path("/tmp/variable-repr-test"),
            "result_log": Path("/tmp/variable-repr-results.txt"),
        }
        _, command = _command(job, roots)
        self.assertIn("--mamba_variable_mixer", command)
        self.assertIn("--mamba_variable_repr_fusion", command)

    def test_plan_validator_and_command_cover_acf_channel_gate(self):
        config = {
            "features": "M", "enc_in": 7, "dec_in": 7, "c_out": 7,
            "mamba_channel_gate": True,
            "mamba_acf_channel_gate": True,
        }
        self.assertEqual(
            _validate_mamba_options("acf_gate", {"seq_len": 96, **config}, ["MambaACF"]),
            [],
        )
        missing_gate = _validate_mamba_options(
            "acf_gate", {"seq_len": 96, "mamba_acf_channel_gate": True}, ["MambaACF"]
        )
        self.assertTrue(any("requires mamba_channel_gate" in error for error in missing_gate))
        mixed_models = _validate_mamba_options(
            "acf_gate", {"seq_len": 96, **config}, ["MambaACF", "Mamba2"]
        )
        self.assertTrue(any("only MambaACF" in error for error in mixed_models))
        job = {
            "phase": "acf_gate", "dataset": "ETTh1", "model": "MambaACF",
            "pred_len": 96, "seed": 2021, "autocon_lambda": None,
            "config": config,
        }
        roots = {
            "checkpoints": Path("/tmp/acf-gate-checkpoints"),
            "results": Path("/tmp/acf-gate-results"),
            "test": Path("/tmp/acf-gate-test"),
            "result_log": Path("/tmp/acf-gate-results.txt"),
        }
        _, command = _command(job, roots)
        self.assertIn("--mamba_channel_gate", command)
        self.assertIn("--mamba_acf_channel_gate", command)

    def test_plan_validator_and_command_cover_acf_strength_order(self):
        config = {
            "features": "M", "enc_in": 7, "dec_in": 7, "c_out": 7,
            "mamba_variable_mixer": True,
            "mamba_acf_channel_order": "strength",
        }
        self.assertEqual(
            _validate_mamba_options("acf_order", {"seq_len": 96, **config}, ["MambaACF"]),
            [],
        )
        missing_mixer = _validate_mamba_options(
            "acf_order", {"seq_len": 96, "mamba_acf_channel_order": "strength"},
            ["MambaACF"],
        )
        self.assertTrue(any("requires mamba_variable_mixer" in error for error in missing_mixer))
        invalid_order = _validate_mamba_options(
            "acf_order", {"seq_len": 96, "mamba_acf_channel_order": "unknown"},
            ["MambaACF"],
        )
        self.assertTrue(any("must be input, strength, or similarity" in error for error in invalid_order))
        similarity_config = {
            "features": "M", "enc_in": 7, "dec_in": 7, "c_out": 7,
            "mamba_variable_mixer": True,
            "mamba_acf_channel_order": "similarity",
        }
        self.assertEqual(
            _validate_mamba_options(
                "acf_similarity", {"seq_len": 96, **similarity_config}, ["MambaACF"]
            ),
            [],
        )
        similarity_job = {
            "phase": "acf_similarity", "dataset": "ETTh1", "model": "MambaACF",
            "pred_len": 96, "seed": 2021, "autocon_lambda": None,
            "config": similarity_config,
        }
        roots = {
            "checkpoints": Path("/tmp/acf-order-checkpoints"),
            "results": Path("/tmp/acf-order-results"),
            "test": Path("/tmp/acf-order-test"),
            "result_log": Path("/tmp/acf-order-results.txt"),
        }
        _, similarity_command = _command(similarity_job, roots)
        similarity_index = similarity_command.index("--mamba_acf_channel_order")
        self.assertEqual(similarity_command[similarity_index + 1], "similarity")
        mixed = _validate_mamba_options(
            "acf_order", {"seq_len": 96, **config}, ["MambaACF", "Mamba2"],
        )
        self.assertTrue(any("only MambaACF" in error for error in mixed))

        job = {
            "phase": "acf_order", "dataset": "ETTh1", "model": "MambaACF",
            "pred_len": 96, "seed": 2021, "autocon_lambda": None,
            "config": config,
        }
        _, command = _command(job, roots)
        index = command.index("--mamba_acf_channel_order")
        self.assertEqual(command[index + 1], "strength")

    def test_grouped_variable_scan_is_validated_and_forwarded(self):
        config = {
            "features": "M", "enc_in": 862, "dec_in": 862, "c_out": 862,
            "mamba_variable_mixer": True,
            "mamba_variable_group_size": 32,
        }
        self.assertEqual(
            _validate_mamba_options("grouped", {"seq_len": 96, **config}, ["MambaACF"]),
            [],
        )
        invalid = _validate_mamba_options(
            "grouped_plain", {"seq_len": 96, "mamba_variable_group_size": 4}, ["Mamba2"]
        )
        self.assertTrue(any("only MambaACF" in error for error in invalid))
        job = {
            "phase": "grouped", "dataset": "traffic", "model": "MambaACF",
            "pred_len": 96, "seed": 2021, "autocon_lambda": None,
            "config": config,
        }
        roots = {
            "checkpoints": Path("/tmp/grouped-checkpoints"),
            "results": Path("/tmp/grouped-results"),
            "test": Path("/tmp/grouped-test"),
            "result_log": Path("/tmp/grouped-results.txt"),
        }
        _, command = _command(job, roots)
        index = command.index("--mamba_variable_group_size")
        self.assertEqual(command[index + 1], "32")

    def test_variable_mixer_arm_alias_resolves_to_paired_mamba_acf_config(self):
        self.assertEqual(MODEL_SPECS["MambaACFVariable"]["model"], "MambaACF")
        plan = {"global": {"features": "M", "seq_len": 96}}
        phase = {
            "name": "variable_screen", "datasets": ["ETTh1"],
            "models": ["MambaACF", "MambaACFVariable"],
            "features": "M", "horizons": [96], "seeds": [2021],
            "mamba_variable_mixer": False, "mamba_variable_group_size": 1,
        }
        jobs = _job_specs(plan, phase, {})
        self.assertEqual(len(jobs), 2)
        by_model = {job["model"]: job for job in jobs}
        self.assertFalse(by_model["MambaACF"]["config"]["mamba_variable_mixer"])
        self.assertTrue(by_model["MambaACFVariable"]["config"]["mamba_variable_mixer"])
        roots = {
            "checkpoints": Path("/tmp/variable-arm-checkpoints"),
            "results": Path("/tmp/variable-arm-results"),
            "test": Path("/tmp/variable-arm-test"),
            "result_log": Path("/tmp/variable-arm-results.txt"),
        }
        _, plain_command = _command(by_model["MambaACF"], roots)
        _, variable_command = _command(by_model["MambaACFVariable"], roots)
        self.assertNotIn("--mamba_variable_mixer", plain_command)
        self.assertIn("--mamba_variable_mixer", variable_command)
        self.assertEqual(
            variable_command[variable_command.index("--model") + 1], "MambaACF"
        )

    def test_variable_autocon_arm_alias_resolves_to_autocon_with_mixer(self):
        self.assertEqual(
            MODEL_SPECS["MambaACFVariableAutoCon"]["model"], "MambaACFAutoCon"
        )
        plan = {"global": {"features": "M", "seq_len": 96}}
        phase = {
            "name": "variable_autocon_screen", "datasets": ["ETTh1"],
            "models": ["MambaACFVariableAutoCon"], "features": "M",
            "horizons": [96], "seeds": [2021],
            "mamba_variable_mixer": False, "mamba_variable_group_size": 1,
            "mamba_acf_relation_loss": True,
        }
        jobs = _job_specs(plan, phase, {})
        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        self.assertTrue(job["config"]["mamba_variable_mixer"])
        self.assertTrue(job["config"]["mamba_acf_relation_loss"])
        roots = {
            "checkpoints": Path("/tmp/variable-autocon-checkpoints"),
            "results": Path("/tmp/variable-autocon-results"),
            "test": Path("/tmp/variable-autocon-test"),
            "result_log": Path("/tmp/variable-autocon-results.txt"),
        }
        _, command = _command({**job, "autocon_lambda": 0.01}, roots)
        self.assertEqual(command[command.index("--model") + 1], "MambaACFAutoCon")
        self.assertIn("--mamba_variable_mixer", command)
        self.assertIn("--mamba_acf_relation_loss", command)

    def test_lite_autocon_arm_alias_resolves_to_single_scale_low_rank_config(self):
        self.assertEqual(
            MODEL_SPECS["MambaACFLiteAutoCon"]["model"], "MambaACFAutoCon"
        )
        plan = {"global": {"features": "M", "seq_len": 96}}
        phase = {
            "name": "lite_autocon_screen", "datasets": ["ETTh1"],
            "models": ["MambaACFLiteAutoCon"], "features": "M",
            "horizons": [96], "seeds": [2021],
            "mamba_scales": [1], "mamba_causal_embedding": False,
            "mamba_low_rank_head": False, "mamba_temporal_rank": 16,
            "mamba_acf_relation_loss": True,
        }
        jobs = _job_specs(plan, phase, {})
        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        self.assertEqual(job["model"], "MambaACFLiteAutoCon")
        self.assertEqual(job["config"]["mamba_scales"], [1])
        self.assertTrue(job["config"]["mamba_causal_embedding"])
        self.assertTrue(job["config"]["mamba_low_rank_head"])
        self.assertEqual(job["config"]["mamba_temporal_rank"], 8)
        self.assertFalse(job["config"]["mamba_variable_mixer"])
        self.assertFalse(job["config"]["mamba_use_acf_memory"])
        roots = {
            "checkpoints": Path("/tmp/lite-autocon-checkpoints"),
            "results": Path("/tmp/lite-autocon-results"),
            "test": Path("/tmp/lite-autocon-test"),
            "result_log": Path("/tmp/lite-autocon-results.txt"),
        }
        _, command = _command(job, roots)
        self.assertEqual(command[command.index("--model") + 1], "MambaACFAutoCon")
        scales_index = command.index("--mamba_scales")
        self.assertEqual(command[scales_index + 1], "1")
        self.assertIn("--mamba_low_rank_head", command)
        self.assertIn("--mamba_causal_embedding", command)
        errors = _validate_mamba_options(
            "lite_autocon", job["config"], ["MambaACFLiteAutoCon"]
        )
        self.assertEqual(errors, [])

    def test_local_patch_adapter_is_validated_and_forwarded(self):
        config = {
            "features": "M", "enc_in": 862, "dec_in": 862, "c_out": 862,
            "mamba_local_mixer": True,
            "mamba_local_patch_len": 16,
            "mamba_local_patch_stride": 8,
            "mamba_local_width": 12,
            "mamba_local_rank": 4,
            "mamba_local_depth": 2,
        }
        self.assertEqual(
            _validate_mamba_options("local", {"seq_len": 96, **config}, ["MambaACF"]),
            [],
        )
        invalid_model = _validate_mamba_options(
            "local_plain", {"seq_len": 96, **config}, ["Mamba2"]
        )
        self.assertTrue(any("only MambaACF" in error for error in invalid_model))
        invalid_independent = _validate_mamba_options(
            "local_ci", {"seq_len": 96, **config, "mamba_channel_independent": True},
            ["MambaACF"],
        )
        self.assertTrue(any("channel_independent" in error for error in invalid_independent))
        job = {
            "phase": "local", "dataset": "traffic", "model": "MambaACF",
            "pred_len": 96, "seed": 2021, "autocon_lambda": None,
            "config": config,
        }
        roots = {
            "checkpoints": Path("/tmp/local-checkpoints"),
            "results": Path("/tmp/local-results"),
            "test": Path("/tmp/local-test"),
            "result_log": Path("/tmp/local-results.txt"),
        }
        _, command = _command(job, roots)
        self.assertIn("--mamba_local_mixer", command)
        self.assertEqual(
            command[command.index("--mamba_local_width") + 1], "12"
        )

    def test_local_patch_null_width_uses_model_width(self):
        job = {
            "phase": "local_legacy", "dataset": "traffic", "model": "MambaACF",
            "pred_len": 96, "seed": 2021, "autocon_lambda": None,
            "config": {
                "features": "M", "enc_in": 862, "dec_in": 862, "c_out": 862,
                "d_model": 32, "mamba_local_mixer": True,
                "mamba_local_width": None,
            },
        }
        roots = {
            "checkpoints": Path("/tmp/local-legacy-checkpoints"),
            "results": Path("/tmp/local-legacy-results"),
            "test": Path("/tmp/local-legacy-test"),
            "result_log": Path("/tmp/local-legacy-results.txt"),
        }
        _, command = _command(job, roots)
        self.assertEqual(
            command[command.index("--mamba_local_width") + 1], "32"
        )

    def test_resume_rejects_changed_artifact_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "results" / "run"
            checkpoint = root / "checkpoints" / "run"
            result.mkdir(parents=True)
            checkpoint.mkdir(parents=True)
            (result / "config.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
            (result / "metrics.npy").write_bytes(b"metrics")
            (result / "original_metrics.npy").write_bytes(b"original")
            (result / "channel_metrics.npy").write_bytes(b"channels")
            (result / "training_complete.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")
            (checkpoint / "checkpoint.pth").write_bytes(b"checkpoint")
            (checkpoint / "loss_logs.json").write_text(json.dumps({"val_loss": [1.0]}), encoding="utf-8")
            record = {
                "expected_artifact": str(result / "training_complete.json"),
                "command": ["--checkpoints", str(root / "checkpoints"), "--no_save"],
                "artifact_hashes": _artifact_hashes(result, checkpoint),
            }
            self.assertTrue(_artifact_exists(record))
            (result / "metrics.npy").write_bytes(b"edited")
            self.assertFalse(_artifact_exists(record))

    def test_resume_accepts_stream_metrics_with_compact_test_artifacts(self):
        """Streaming test output keeps metrics.npy as the runner anchor.

        ``write_streaming_artifacts`` also emits the compact JSON/NPZ files,
        but the manifest intentionally continues to point at metrics.npy so
        existing validators and readers retain one stable artifact contract.
        ``--no_save`` must not make prediction arrays mandatory.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "results" / "stream_run"
            checkpoint = root / "checkpoints" / "stream_run"
            result.mkdir(parents=True)
            checkpoint.mkdir(parents=True)

            (result / "config.json").write_text(
                json.dumps({"stream_metrics": True}), encoding="utf-8"
            )
            np.save(result / "metrics.npy", np.asarray([0.1, 0.2, 0.3]))
            np.save(result / "original_metrics.npy", np.asarray([0.4, 0.5, 0.6]))
            np.save(result / "channel_metrics.npy", np.asarray([[0.1, 0.2]]))
            (result / "stream_metrics.json").write_text(
                json.dumps({"schema_version": 1, "scaled": {"mse": 0.2}}),
                encoding="utf-8",
            )
            np.savez_compressed(
                result / "stream_samples.npz",
                pred=np.zeros((0, 1, 1), dtype=np.float32),
                true=np.zeros((0, 1, 1), dtype=np.float32),
            )
            (checkpoint / "checkpoint.pth").write_bytes(b"checkpoint")
            (checkpoint / "loss_logs.json").write_text(
                json.dumps({"val_loss": [1.0]}), encoding="utf-8"
            )
            record = {
                "status": "ok",
                "expected_artifact": str(result / "metrics.npy"),
                "command": [
                    "--checkpoints", str(root / "checkpoints"),
                    "--stream_metrics", "--no_save",
                ],
            }
            record["artifact_hashes"] = _artifact_hashes(result, checkpoint)

            self.assertTrue(_artifact_exists(record))
            self.assertFalse((result / "pred.npy").exists())
            self.assertFalse((result / "true.npy").exists())

            # Removing the canonical metrics anchor must invalidate resume,
            # even when compact stream files remain present.
            (result / "metrics.npy").unlink()
            self.assertFalse(_artifact_exists(record))

    def test_validator_uses_legacy_defaults_for_new_optional_fields(self):
        """A pre-streaming config remains revalidatable when defaults are known."""
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "results" / "legacy"
            result.mkdir(parents=True)
            np.save(result / "metrics.npy", np.asarray([1.0, 2.0, 2.0]))

            # This is the shape of a legacy DLinear config, intentionally
            # omitting fields introduced after the original phase ran.
            config = {
                "data": "ETTh1", "root_path": "dataset/ETT-small",
                "data_path": "ETTh1.csv", "target": "OT", "freq": "h",
                "pred_len": 96, "run_seed": 2021, "model": "DLinear",
                "AutoCon": False, "features": "S", "seq_len": 96,
                "label_len": 48, "enc_in": 1, "dec_in": 1, "c_out": 1,
                "d_model": 32, "d_ff": 32, "e_layers": 1, "n_heads": 8,
                "d_layers": 1, "factor": 1, "top_k": 5, "num_kernels": 6,
                "moving_avg": 25, "patch_len": 16, "patch_stride": 8,
                "dropout": 0.1, "embed": "timeF", "activation": "gelu",
                "wo_freq": "", "individual": False, "use_amp": False,
                "mamba_d_state": 16, "mamba_d_conv": 4, "mamba_expand": 2,
                "mamba_headdim": 16, "AutoCon_multiscales": [96],
                "AutoCon_warmup_epochs": 1, "batch_size": 32,
                "train_epochs": 20, "patience": 5, "learning_rate": 1e-3,
                "lradj": "type1", "weight_decay": 1e-5, "num_workers": 0,
                "train_ratio": 0.6, "save": False, "skip_test": False,
                "AutoCon_wnorm": "LastVal", "AutoCon_acf_source": "residual",
            }
            (result / "config.json").write_text(json.dumps(config), encoding="utf-8")
            job_config = {
                **config,
                "min_learning_rate": 0.0,
                "drop_last_train": False,
                "stream_metrics": False,
                "stream_sample_windows": 4,
                "origin_stride": None,
            }
            job = {
                "phase": "legacy", "dataset": "ETTh1", "model": "DLinear",
                "pred_len": 96, "seed": 2021, "autocon_lambda": None,
                "config": job_config,
            }
            record = {
                "status": "ok",
                "expected_artifact": str(result / "metrics.npy"),
                "command": [],
            }
            self.assertEqual(_validate_record(record, job), [])

    def test_validator_checks_mamba_acf_architecture_and_relation_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "results" / "mamba_acf"
            result.mkdir(parents=True)
            np.save(result / "metrics.npy", np.asarray([1.0, 2.0, 2.0]))
            config = {
                "data": "ETTh1", "root_path": "dataset/ETT-small",
                "data_path": "ETTh1.csv", "target": "OT", "freq": "h",
                "pred_len": 96, "run_seed": 2021, "model": "MambaACFAutoCon",
                "AutoCon": True, "features": "S", "seq_len": 96,
                "label_len": 48, "enc_in": 1, "dec_in": 1, "c_out": 1,
                "d_model": 32, "d_ff": 32, "e_layers": 1, "n_heads": 8,
                "d_layers": 1, "factor": 1, "top_k": 5, "num_kernels": 6,
                "moving_avg": 25, "patch_len": 16, "patch_stride": 8,
                "dropout": 0.1, "embed": "timeF", "activation": "gelu",
                "wo_freq": "", "individual": False, "use_amp": False,
                "mamba_d_state": 16, "mamba_d_conv": 4, "mamba_expand": 2,
                "mamba_headdim": 16, "mamba_scales": [2, 1],
                "mamba_token_stride": 8,
                "mamba_bidirectional": "true", "mamba_channel_independent": "false",
                "mamba_channel_gate": True, "mamba_channel_gate_rank": 4,
                "mamba_causal_embedding": True, "mamba_variable_mixer": True,
                "mamba_variable_mixer_depth": 2, "mamba_variable_permutation": True,
                "mamba_low_rank_head": False, "mamba_temporal_rank": 16,
                "mamba_acf_lags": [24, 48], "mamba_acf_lag_mode": "fixed",
                "mamba_acf_max_lag": 720, "mamba_acf_top_k": 8,
                "mamba_acf_min_lag": 2, "mamba_acf_min_abs": 0.1,
                "mamba_acf_confidence_z": 1.96, "mamba_acf_min_separation": 2,
                "mamba_acf_detrend": "mean", "mamba_acf_chunk": 64,
                "mamba_use_acf_memory": True, "mamba_dynamic_acf": False,
                "mamba_acf_temperature": 4.0, "mamba_acf_relation_loss": True,
                "mamba_acf_relation_batching": "lag_pairs",
                "mamba_acf_relation_channel_reduce": "mean",
                "mamba_acf_relation_scope": "global",
                "mamba_acf_relation_lag_mask": True,
                "mamba_acf_relation_lag_tolerance": 1,
                "AutoCon_multiscales": [96], "AutoCon_warmup_epochs": 0,
                "AutoCon_acf_source": "residual", "AutoCon_temperature": 0.2,
                "AutoCon_relation_temperature": 0.25, "AutoCon_sign_weight": 0.25,
                "AutoCon_sign_margin": 0.1, "batch_size": 32, "train_epochs": 3,
                "patience": 2, "learning_rate": 1e-3, "lradj": "type1",
                "min_learning_rate": 0.0, "weight_decay": 1e-5, "num_workers": 0,
                "drop_last_train": False, "train_ratio": 0.6, "save": False,
                "skip_test": False, "stream_metrics": False,
                "stream_sample_windows": 4, "AutoCon_wnorm": "LastVal",
                "AutoCon_lambda": 0.05, "AutoCon_acf_source": "residual",
            }
            (result / "config.json").write_text(json.dumps(config), encoding="utf-8")
            job_config = {
                **config, "autocon_scale": [96], "autocon_warmup": 0,
                "epochs": 3, "save_predictions": False,
            }
            job_config["mamba_scales"] = [1, 2]
            job_config["mamba_bidirectional"] = True
            job_config["mamba_channel_independent"] = False
            job = {
                "phase": "mamba_acf", "dataset": "ETTh1", "model": "MambaACFAutoCon",
                "pred_len": 96, "seed": 2021, "autocon_lambda": 0.05,
                "config": job_config,
            }
            record = {
                "status": "ok", "expected_artifact": str(result / "metrics.npy"),
                "command": [],
            }
            self.assertEqual(_validate_record(record, job), [])
            config["mamba_token_stride"] = 1
            (result / "config.json").write_text(json.dumps(config), encoding="utf-8")
            errors = _validate_record(record, job)
            self.assertTrue(any("mamba_token_stride" in error for error in errors))
            config["mamba_token_stride"] = 8
            (result / "config.json").write_text(json.dumps(config), encoding="utf-8")
            config["mamba_variable_mixer_depth"] = 1
            (result / "config.json").write_text(json.dumps(config), encoding="utf-8")
            errors = _validate_record(record, job)
            self.assertTrue(any("mamba_variable_mixer_depth" in error for error in errors))
            config["mamba_variable_mixer_depth"] = 2
            config["mamba_acf_lag_mode"] = "unregistered"
            job_config["mamba_acf_lag_mode"] = "unregistered"
            (result / "config.json").write_text(json.dumps(config), encoding="utf-8")
            errors = _validate_record(record, job)
            self.assertIn("config_mamba_acf_lag_mode_invalid", errors)

    def test_relation_v3_logs_require_soft_target_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "results" / "run"
            checkpoint = root / "checkpoints" / "run"
            result.mkdir(parents=True)
            checkpoint.mkdir(parents=True)
            np.save(result / "metrics.npy", np.asarray([1.0, 2.0, 2.0]))
            config = {
                "data": "ETTh1", "root_path": "dataset/ETT-small",
                "data_path": "ETTh1.csv", "target": "OT", "freq": "h",
                "pred_len": 96, "run_seed": 2021, "model": "MambaACFAutoCon",
                "AutoCon": True, "features": "S", "seq_len": 96,
                "label_len": 48, "enc_in": 1, "dec_in": 1, "c_out": 1,
                "d_model": 32, "d_ff": 32, "e_layers": 1, "n_heads": 8,
                "d_layers": 1, "factor": 1, "top_k": 5, "num_kernels": 6,
                "moving_avg": 25, "patch_len": 16, "patch_stride": 8,
                "dropout": 0.1, "embed": "timeF", "activation": "gelu",
                "wo_freq": "", "individual": False, "use_amp": False,
                "mamba_d_state": 16, "mamba_d_conv": 4, "mamba_expand": 2,
                "mamba_headdim": 16, "mamba_scales": [1, 2],
                "mamba_token_stride": 1, "mamba_bidirectional": False,
                "mamba_channel_independent": False, "mamba_channel_gate": False,
                "mamba_channel_gate_rank": 8, "mamba_causal_embedding": False,
                "mamba_variable_mixer": False, "mamba_variable_mixer_depth": 1,
                "mamba_variable_permutation": False, "mamba_low_rank_head": False,
                "mamba_temporal_rank": 16, "mamba_acf_lags": [24, 48],
                "mamba_acf_lag_mode": "fixed", "mamba_acf_max_lag": 720,
                "mamba_acf_top_k": 8, "mamba_acf_min_lag": 2,
                "mamba_acf_min_abs": 0.1, "mamba_acf_confidence_z": 1.96,
                "mamba_acf_min_separation": 2, "mamba_acf_detrend": "mean",
                "mamba_acf_chunk": 64, "mamba_use_acf_memory": False,
                "mamba_dynamic_acf": False, "mamba_acf_temperature": 4.0,
                "mamba_acf_step_reliability": False, "mamba_acf_relation_loss": True,
                "mamba_acf_relation_batching": "lag_pairs",
                "mamba_acf_relation_channel_reduce": "mean",
                "mamba_acf_relation_scope": "global",
                "mamba_acf_relation_lag_mask": False,
                "mamba_acf_relation_lag_tolerance": 0,
                "mamba_acf_relation_min_lag": 96,
                "mamba_acf_relation_target_mode": "soft_multi_peak",
                "mamba_acf_relation_soft_min_abs": 0.1,
                "mamba_acf_relation_selected_lag_boost": 1.0,
                "AutoCon_multiscales": [96], "AutoCon_warmup_epochs": 0,
                "AutoCon_acf_source": "residual", "AutoCon_temperature": 0.2,
                "AutoCon_relation_temperature": 0.25, "AutoCon_sign_weight": 0.25,
                "AutoCon_sign_margin": 0.1, "batch_size": 32, "train_epochs": 3,
                "patience": 2, "learning_rate": 1e-3, "lradj": "type1",
                "min_learning_rate": 0.0, "weight_decay": 1e-5, "num_workers": 0,
                "drop_last_train": False, "train_ratio": 0.6, "save": False,
                "skip_test": False, "stream_metrics": False,
                "stream_sample_windows": 4, "AutoCon_wnorm": "LastVal",
                "AutoCon_lambda": 0.05,
            }
            (result / "config.json").write_text(json.dumps(config), encoding="utf-8")
            job_config = {
                **config, "autocon_scale": [96], "autocon_warmup": 0,
                "epochs": 3, "save_predictions": False,
            }
            job = {
                "phase": "relation_v3", "dataset": "ETTh1",
                "model": "MambaACFAutoCon", "pred_len": 96, "seed": 2021,
                "autocon_lambda": 0.05, "config": job_config,
            }
            record = {
                "status": "ok", "expected_artifact": str(result / "metrics.npy"),
                "command": ["--checkpoints", str(root / "checkpoints")],
            }
            (checkpoint / "checkpoint.pth").write_bytes(b"checkpoint")
            (checkpoint / "loss_logs.json").write_text(json.dumps({
                "val_loss": [1.0], "acf_relation_objective_version": 3,
                "acf_relation_global_candidate_fraction": [1.0],
                "acf_relation_global_candidate_active_anchor_fraction": [1.0],
                "acf_relation_global_positive_count_mean": [1.0],
                "acf_relation_global_high_acf_excluded_fraction": [0.0],
                "acf_relation_local_target_fraction": [0.0],
                "acf_relation_local_active_position_fraction": [0.0],
            }), encoding="utf-8")
            self.assertEqual(_validate_record(record, job), [])
            (checkpoint / "loss_logs.json").write_text(json.dumps({
                "val_loss": [1.0], "acf_relation_objective_version": 2,
            }), encoding="utf-8")
            errors = _validate_record(record, job)
            self.assertIn("acf_relation_objective_version_missing_or_old", errors)
            self.assertIn("acf_relation_local_target_fraction_missing", errors)


if __name__ == "__main__":
    unittest.main()
