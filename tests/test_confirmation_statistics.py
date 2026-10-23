import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import stats

from scripts.analyze_mamba_acf_confirmation import (
    _analyze_contrast,
    _canonical_hash,
    _confirmation_phase_names,
    _holm_adjust,
    _manifest_records,
    _peak_placebo_contract,
    _paired_t_upper,
    _registered_job_counts_contract,
    _relation_log_contract,
    _validate_memory_contract,
    _statistical_contract,
    _threshold_contract,
    _validate_analysis_authorization,
    _validate_selected_lambda_binding,
    _wilcoxon_less_p,
    build_analysis,
)
from scripts.compute_paper_statistics import main as compute_paper_statistics_main
from scripts.validate_training_phase import _validate_relation_contract


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hash(root: Path) -> tuple[dict[str, str], str]:
    files = {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return files, hashlib.sha256(encoded).hexdigest()


class ConfirmationStatisticsUnitTests(unittest.TestCase):
    def test_registered_accuracy_claim_boundary_matches_primary_gate(self):
        plan_path = Path(__file__).resolve().parents[1] / (
            "docs/research/mamba_acf_confirmation_plan.json"
        )
        with plan_path.open(encoding="utf-8") as handle:
            plan = json.load(handle)
        contract = plan["accuracy_claim_contract"]
        self.assertEqual(
            contract["primary_noninferiority_comparators"],
            ["PatchTST", "Mamba2"],
        )
        primary = "\n".join(plan["statistical_gate"]["primary_comparisons"])
        self.assertIn(":PatchTST", primary)
        self.assertIn(":Mamba2", primary)
        for baseline in contract["secondary_descriptive_baselines"]:
            self.assertNotIn(":" + baseline, primary)

    def test_paper_statistics_unbound_override_requires_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            results.mkdir()
            original_argv = sys.argv
            sys.argv = [
                "compute_paper_statistics.py",
                "--results_dir", str(results),
                "--output_dir", str(root / "out"),
                "--allow-unvalidated",
            ]
            try:
                with self.assertRaisesRegex(ValueError, "--plan is required"):
                    compute_paper_statistics_main()
            finally:
                sys.argv = original_argv

    def test_paper_statistics_cannot_use_unvalidated_confirmation_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            results.mkdir()
            plan_path = root / "plan.json"
            plan_path.write_text(
                json.dumps({
                    "confirmation_contract": {},
                    "phases": [{
                        "name": "phase",
                        "output_root": str(results),
                        "datasets": ["toy"],
                        "models": ["M"],
                        "features": "M",
                        "horizons": [2],
                        "seeds": [1],
                    }],
                }),
                encoding="utf-8",
            )
            original_argv = sys.argv
            sys.argv = [
                "compute_paper_statistics.py",
                "--results_dir", str(results),
                "--output_dir", str(root / "out"),
                "--plan", str(plan_path),
                "--phase", "phase",
                "--allow-unvalidated",
            ]
            try:
                with self.assertRaisesRegex(ValueError, "allow-unvalidated"):
                    compute_paper_statistics_main()
            finally:
                sys.argv = original_argv

    def test_unapproved_confirmation_plan_is_blocked_before_phase_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            plan = {
                "status": "draft_requires_prelaunch_review",
                "execution": {
                    "authorized": False,
                    "status": "draft_requires_prelaunch_review",
                },
                "confirmation_contract": {},
                "statistical_gate": {},
            }
            path.write_text(json.dumps(plan), encoding="utf-8")
            with self.assertRaises(ValueError):
                _validate_analysis_authorization(plan, path)

    def test_all_test_producing_confirmation_phases_are_required(self):
        plan = {
            "phases": [
                {"name": "selection", "skip_test": True},
                {"name": "confirm_a", "skip_test": False},
                {"name": "confirm_b", "skip_test": False},
            ]
        }
        self.assertEqual(_confirmation_phase_names(plan), {"confirm_a", "confirm_b"})

    def test_numeric_threshold_contract_accepts_ratio_and_log_forms(self):
        gate = {
            "noninferiority_margin": {
                "ratio": 1.01,
                "log_threshold": math.log(1.01),
            }
        }
        result = _threshold_contract(
            gate,
            primary_key="noninferiority_margin",
            ratio_keys=(),
            log_keys=(),
            label="noninferiority margin",
            require_ratio_at_least_one=True,
        )
        self.assertAlmostEqual(result["ratio"], 1.01)
        self.assertAlmostEqual(result["log_threshold"], math.log(1.01))

    def test_selected_lambda_binding_requires_copied_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provenance_dir = root / "provenance"
            provenance_dir.mkdir()
            selection_path = provenance_dir / "selected_hyperparameters.json"
            sidecar_path = provenance_dir / "selected_hyperparameters.provenance.json"
            selected = {"MambaACFAutoCon": 0.03}
            selection_path.write_text(
                json.dumps({
                    "phase": "selection_phase",
                    "features": "M",
                    "selected_lambda_by_model": selected,
                }, sort_keys=True),
                encoding="utf-8",
            )
            sidecar_path.write_text("{}\n", encoding="utf-8")
            binding = {
                "selected_lambda_by_model_sha256": _canonical_hash(selected),
                "selection_provenance_publication_ready": False,
                "selection_phase_validation_sha256": "validation-hash",
                "selection_manifest_sha256": "manifest-hash",
            }
            provenance = {
                "selection_provenance": binding,
                "selection_copy": str(selection_path),
                "selection_provenance_copy": str(sidecar_path),
                "selection_sha256": _sha256(selection_path),
                "selection_provenance_sha256": _sha256(sidecar_path),
                "selection_binding_sha256": _canonical_hash(binding),
            }
            phase = {
                "autocon_lambda": "selected",
                "selection_phase": "selection_phase",
                "features": "M",
                "models": ["MambaACFAutoCon"],
            }
            self.assertEqual(
                _validate_selected_lambda_binding(root, phase, provenance), selected
            )
            provenance["selection_binding_sha256"] = "tampered"
            with self.assertRaises(ValueError):
                _validate_selected_lambda_binding(root, phase, provenance)

    def test_numeric_threshold_contract_rejects_conflicting_or_prose_values(self):
        with self.assertRaises(ValueError):
            _threshold_contract(
                {"noninferiority_margin": {"ratio": 1.01, "log_threshold": 0.2}},
                primary_key="noninferiority_margin",
                ratio_keys=(),
                log_keys=(),
                label="noninferiority margin",
                require_ratio_at_least_one=True,
            )
        with self.assertRaises(ValueError):
            _threshold_contract(
                {"noninferiority_margin": "one percent"},
                primary_key="noninferiority_margin",
                ratio_keys=(),
                log_keys=(),
                label="noninferiority margin",
                require_ratio_at_least_one=True,
            )

    def test_gate_direction_is_upper_bound_based(self):
        noninferiority = {"ratio": 1.01, "log_threshold": math.log(1.01)}
        superiority = {"ratio": 1.0, "log_threshold": 0.0}
        contrast = {
            "label": "direction",
            "treatment_phase": "t",
            "treatment_model": "T",
            "reference_phase": "r",
            "reference_model": "R",
        }
        rows = {
            "t": [
                {"dataset": "toy", "features": "M", "seq_len": 4, "pred_len": 2, "seed": 1, "model": "T", "mse": 0.5},
                {"dataset": "toy", "features": "M", "seq_len": 4, "pred_len": 2, "seed": 2, "model": "T", "mse": 0.5},
            ],
            "r": [
                {"dataset": "toy", "features": "M", "seq_len": 4, "pred_len": 2, "seed": 1, "model": "R", "mse": 1.0},
                {"dataset": "toy", "features": "M", "seq_len": 4, "pred_len": 2, "seed": 2, "model": "R", "mse": 1.0},
            ],
        }
        result = _analyze_contrast(contrast, rows, 1, 0.05, noninferiority, superiority)
        cell = result["cells"][0]
        self.assertTrue(cell["noninferiority_pass"])
        self.assertTrue(cell["superiority_pass"])
        self.assertTrue(result["gate_coverage"]["noninferiority_all_cells_pass"])
        self.assertTrue(result["gate_coverage"]["superiority_all_cells_pass"])

        # An exactly equal upper bound is not a strict superiority result.
        equal_rows = {
            "t": [dict(row, model="T", mse=1.0) for row in rows["r"]],
            "r": rows["r"],
        }
        equal_result = _analyze_contrast(
            contrast, equal_rows, 1, 0.05, noninferiority, superiority
        )
        self.assertFalse(equal_result["cells"][0]["superiority_pass"])
    def test_log_ratio_and_bonferroni_one_sided_upper_bound(self):
        values = [math.log(0.5), 0.0, math.log(0.75)]
        result = _paired_t_upper(values, alpha=0.05 / 12.0)
        expected_mean = float(np.mean(values))
        expected_sd = float(np.std(values, ddof=1))
        expected_upper = expected_mean + float(
            stats.t.ppf(1.0 - 0.05 / 12.0, 2)
        ) * expected_sd / math.sqrt(3.0)
        self.assertEqual(result["n_pairs"], 3)
        self.assertAlmostEqual(result["mean"], expected_mean)
        self.assertAlmostEqual(result["upper"], expected_upper)

    def test_holm_adjustment_preserves_order(self):
        self.assertEqual(
            _holm_adjust([0.01, 0.04, 0.20, None]),
            [0.03, 0.08, 0.20, None],
        )

    def test_exact_wilcoxon_less_with_ties_and_zeros(self):
        # The implementation uses the exact conditional sign distribution of
        # midranks after dropping zero differences.
        self.assertAlmostEqual(_wilcoxon_less_p([-1.0, -1.0, 1.0]), 0.5)
        self.assertAlmostEqual(_wilcoxon_less_p([-2.0, 0.0, 1.0, 3.0]), 0.75)
        self.assertEqual(_wilcoxon_less_p([0.0, 0.0]), 1.0)

    def test_nonfinite_and_nonpositive_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            _paired_t_upper([0.0, float("nan")], 0.05)
        with self.assertRaises(ValueError):
            _wilcoxon_less_p([0.0, float("inf")])

    def test_manifest_retries_use_last_record_and_keep_failed_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            path.write_text(
                "".join([
                    json.dumps({"job_id": "j", "status": "failed"}) + "\n",
                    json.dumps({"job_id": "j", "status": "ok"}) + "\n",
                ]),
                encoding="utf-8",
            )
            records, audit = _manifest_records(path)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["status"], "ok")
            self.assertEqual(audit["retry_job_count"], 1)
            self.assertEqual(audit["prior_non_ok_attempt_count"], 1)

    def test_relation_contract_rejects_fallback_and_zero_coverage(self):
        config = {
            "model": "MambaACFAutoCon",
            "mamba_acf_relation_loss": True,
            "mamba_acf_relation_batching": "lag_pairs",
            "mamba_acf_relation_target_mode": "hard_selected",
            "mamba_acf_relation_lag_mask": True,
            "mamba_acf_relation_scope": "global",
            "mamba_acf_relation_batching_requested": "lag_pairs",
            "mamba_acf_relation_batching_effective": "lag_pairs",
            "mamba_acf_relation_batching_reason": "hard_selected_lag_mask",
            "mamba_acf_relation_selected_lag_support": True,
            "mamba_acf_relation_effective_lags": [96],
            "mamba_acf_relation_selected_lags": [96, 120],
            "mamba_acf_train_peak_lags": [96, 120],
            "mamba_acf_prior_max_lag": 720,
            "mamba_acf_relation_min_lag": 96,
            "mamba_acf_lag_mode": "train_peaks_available",
            "mamba_acf_relation_lag_source": "train_peak_subset",
            "mamba_acf_relation_lag_fallback": False,
            "mamba_acf_relation_scope": "global",
        }
        logs = {
            "val_loss": [1.0, 0.9],
            "acf_relation_objective_version": 3,
            "acf_relation_effective_lags": [96],
            "acf_relation_selected_lags": [96, 120],
            "acf_relation_train_peak_lags": [96, 120],
            "acf_relation_lag_source": "train_peak_subset",
            "acf_relation_lag_fallback": False,
            "acf_relation_batching_requested": "lag_pairs",
            "acf_relation_batching_effective": "lag_pairs",
            "acf_relation_batching_reason": "hard_selected_lag_mask",
            "acf_relation_selected_lag_support": True,
            "acf_relation_lag_pair_counts": [{"96": 1}, {"96": 1}],
            "acf_relation_lag_pair_exposure": [{"96": 1.0}, {"96": 1.0}],
            "acf_relation_global_valid_fraction": [0.2, 0.2],
            "acf_relation_global_active_anchor_fraction": [0.8, 0.8],
            "acf_relation_global_candidate_fraction": [0.3, 0.3],
            "acf_relation_global_candidate_active_anchor_fraction": [0.9, 0.9],
            "acf_relation_global_positive_count_mean": [1.2, 1.1],
            "acf_relation_global_high_acf_excluded_fraction": [0.0, 0.1],
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / "loss_logs.json").write_text(
                json.dumps(logs), encoding="utf-8"
            )
            result = _relation_log_contract(config, checkpoint, "job")
            self.assertEqual(result["relation_effective_lags"], [96])
            self.assertEqual(_validate_relation_contract(config, logs, strict=True), [])
            config["mamba_acf_relation_lag_fallback"] = True
            with self.assertRaisesRegex(ValueError, "fallback"):
                _relation_log_contract(config, checkpoint, "job")
            config["mamba_acf_relation_lag_fallback"] = False
            logs["acf_relation_global_valid_fraction"] = [0.0, 0.2]
            errors = _validate_relation_contract(config, logs, strict=True)
            self.assertIn("acf_relation_global_valid_fraction_not_strictly_positive", errors)

    def test_memory_contract_rejects_unqualified_peak_lags(self):
        plan = {"acf_falsification_contract": {
            "fixed_placebo_arm": "placebo", "placebo_lags": [5, 11, 17]
        }}
        metadata = {
            "use_acf_memory": True,
            "effective_lags": [96],
            "effective_lag_source": "configured_fallback",
            "effective_lag_fallback": True,
            "lag_mode": "train_peaks_available",
            "train_peak_lags": [96],
        }
        with self.assertRaisesRegex(ValueError, "fallback"):
            _validate_memory_contract(plan, "peak", "MambaACF", {}, metadata, "job")

    def test_registered_job_counts_are_recomputed(self):
        plan = {
            "confirmation_contract": {},
            "global": {},
            "phases": [
                {"name": "selection_a", "datasets": ["toy"], "models": ["M"], "horizons": [1], "seeds": [1, 2]},
                {"name": "confirm_a", "datasets": ["toy"], "models": ["M"], "horizons": [1], "seeds": [1, 2, 3]},
            ],
            "registered_job_counts": {
                "selection_a": 2, "confirm_a": 3,
                "selection_total": 2, "confirmation_total": 3,
                "training_total": 5,
            },
        }
        result = _registered_job_counts_contract(plan)
        self.assertEqual(result["recomputed"]["training_total"], 5)
        plan["registered_job_counts"]["confirmation_total"] = 4
        with self.assertRaisesRegex(ValueError, "recomputed"):
            _registered_job_counts_contract(plan)

    def test_confirmation_registered_job_counts_cannot_be_omitted(self):
        with self.assertRaisesRegex(ValueError, "requires registered_job_counts"):
            _registered_job_counts_contract({"confirmation_contract": {}})

    def test_strict_confirmation_requires_registered_job_counts(self):
        strict_plan = {
            "confirmation_contract": {},
            "statistical_gate": {},
        }
        with self.assertRaisesRegex(ValueError, "requires registered_job_counts"):
            _registered_job_counts_contract(strict_plan)
        self.assertIsNone(_registered_job_counts_contract({"phases": []}))

    def test_peak_placebo_contract_requires_equal_qualified_lag_cardinality(self):
        plan = {
            "acf_falsification_contract": {
                "peak_arm": "peak",
                "fixed_placebo_arm": "placebo",
                "placebo_lags": [5, 11, 17],
            }
        }
        def row(phase, seed, lags, source, mode, train_peaks):
            return {
                "dataset": "toy", "features": "M", "seq_len": 96,
                "pred_len": 96, "seed": seed, "model": "MambaACF",
                "memory": {
                    "use_acf_memory": True, "effective_lags": list(lags),
                    "effective_lag_source": source,
                    "effective_lag_fallback": False, "lag_mode": mode,
                    "train_peak_lags": list(train_peaks),
                },
            }
        treatment = [row("peak", seed, [96, 120, 168], "train_peaks_available", "train_peaks_available", [96, 120, 168]) for seed in (1, 2)]
        reference = [row("placebo", seed, [5, 11, 17], "fixed_configured", "fixed", [96, 120, 168]) for seed in (1, 2)]
        result = _peak_placebo_contract(
            plan,
            {"treatment_phase": "peak", "reference_phase": "placebo"},
            treatment,
            reference,
        )
        self.assertTrue(result["eligible"])
        self.assertEqual(result["lag_cardinality_by_cell"][0]["peak_count"], 3)
        bad = list(treatment)
        bad[0] = row("peak", 1, [96, 120], "train_peaks_available", "train_peaks_available", [96, 120, 168])
        with self.assertRaisesRegex(ValueError, "cardinality"):
            _peak_placebo_contract(
                plan,
                {"treatment_phase": "peak", "reference_phase": "placebo"},
                bad,
                reference,
            )


class ConfirmationStatisticsSyntheticPhaseTests(unittest.TestCase):
    def _make_phase(
        self,
        root: Path,
        plan_path: Path,
        phase_name: str,
        model: str,
        values_by_seed: dict[int, float],
    ) -> None:
        root.mkdir(parents=True)
        provenance_dir = root / "provenance"
        snapshot = provenance_dir / "source_snapshot"
        snapshot.mkdir(parents=True)
        (snapshot / "source.py").write_text("x = 1\n", encoding="utf-8")
        analyzer_source = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "analyze_mamba_acf_confirmation.py"
        )
        analyzer_copy = snapshot / "scripts" / analyzer_source.name
        analyzer_copy.parent.mkdir(parents=True)
        analyzer_copy.write_bytes(analyzer_source.read_bytes())
        files, source_tree_hash = _tree_hash(snapshot)
        (provenance_dir / "final_training_plan.json").write_bytes(plan_path.read_bytes())
        plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
        fingerprint = plan_payload.get("data_fingerprints", {}).get("toy", {})
        dataset_path = Path(str(fingerprint.get("relative_path"))).expanduser().resolve()
        provenance = {
            "schema": 1,
            "provenance_id": f"prov-{phase_name}",
            "phase": phase_name,
            "plan_sha256": _sha256(plan_path),
            "selection_sha256": None,
            "selection_provenance_sha256": None,
            "selection_binding_sha256": None,
            "dataset_hashes": {
                "toy": _sha256(dataset_path)
            },
            "source_snapshot": str(snapshot),
            "source_files": files,
            "source_tree_sha256": source_tree_hash,
        }
        (root / "provenance.json").write_text(
            json.dumps(provenance, indent=2), encoding="utf-8"
        )
        result_root = root / "results"
        result_root.mkdir()
        records = []
        artifact_hashes = {}
        for seed, mse in sorted(values_by_seed.items()):
            job_id = f"{phase_name}_{model}_{seed}"
            result_dir = result_root / job_id
            result_dir.mkdir()
            config = {
                "data": "toy",
                "features": "M",
                "seq_len": 4,
                "pred_len": 2,
                "model": model,
                "seed": seed,
                "run_seed": seed,
                "skip_test": False,
                "plan_job_id": job_id,
            }
            (result_dir / "config.json").write_text(
                json.dumps(config, sort_keys=True), encoding="utf-8"
            )
            np.save(result_dir / "metrics.npy", np.asarray([0.1, mse, math.sqrt(mse), 0.0, 0.0]))
            artifact_hashes[job_id] = _artifact_hashes_for_test(result_dir)
            records.append({
                "job_id": job_id,
                "phase": phase_name,
                "dataset": "toy",
                "model": model,
                "features": "M",
                "seq_len": 4,
                "pred_len": 2,
                "seed": seed,
                "status": "ok",
                "provenance_id": provenance["provenance_id"],
                "source_tree_sha256_before": source_tree_hash,
                "source_tree_sha256_after": source_tree_hash,
                "expected_artifact": str(result_dir / "metrics.npy"),
                "artifact_hashes": artifact_hashes[job_id],
            })
        logs = root / "logs"
        logs.mkdir()
        manifest = logs / "job_manifest.jsonl"
        manifest.write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in records),
            encoding="utf-8",
        )
        validation = {
            "valid": True,
            "phase": phase_name,
            "plan_sha256": _sha256(plan_path),
            "manifest": str(manifest),
            "manifest_sha256": _sha256(manifest),
            "provenance": str(root / "provenance.json"),
            "provenance_id": provenance["provenance_id"],
            "expected_jobs": len(records),
            "manifest_unique_jobs": len(records),
            "malformed_manifest_lines": 0,
            "errors": [],
            "artifact_hashes_by_job": artifact_hashes,
        }
        (root / "phase_validation.json").write_text(
            json.dumps(validation, indent=2), encoding="utf-8"
        )

    def test_synthetic_phase_is_loaded_and_paired(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "plan.json"
            toy_path = root / "toy.csv"
            toy_path.write_text("date,value\n2020-01-01,1\n", encoding="utf-8")
            plan = {
                "global": {"features": "M", "seq_len": 4},
                "data_fingerprints": {
                    "toy": {"relative_path": str(toy_path), "sha256": _sha256(toy_path)}
                },
                "phases": [
                    {
                        "name": "treatment_phase",
                        "output_root": str(root / "treatment"),
                        "datasets": ["toy"],
                        "models": ["Treat"],
                        "features": "M",
                        "seq_len": 4,
                        "horizons": [2],
                        "seeds": [1, 2, 3],
                    },
                    {
                        "name": "reference_phase",
                        "output_root": str(root / "reference"),
                        "datasets": ["toy"],
                        "models": ["Ref"],
                        "features": "M",
                        "seq_len": 4,
                        "horizons": [2],
                        "seeds": [1, 2, 3],
                    },
                ],
                "statistical_gate": {
                    "interval_alpha": 0.05,
                    "simultaneous_m_cells": 1,
                    "interval_construction_rule": "Bonferroni adjusted-alpha one-sided t upper bound",
                    "noninferiority_margin": {"ratio": 1.01, "log_threshold": math.log(1.01)},
                    "superiority_threshold": {"ratio": 1.0, "log_threshold": 0.0},
                    "primary_comparisons": [
                        "treatment_phase:Treat::reference_phase:Ref"
                    ],
                },
            }
            plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
            self._make_phase(root / "treatment", plan_path, "treatment_phase", "Treat", {1: 1.0, 2: 1.2, 3: 0.8})
            self._make_phase(root / "reference", plan_path, "reference_phase", "Ref", {1: 1.5, 2: 1.5, 3: 1.0})
            payload = build_analysis(
                plan_path,
                {
                    "treatment_phase": root / "treatment",
                    "reference_phase": root / "reference",
                },
            )
            self.assertEqual(payload["status"], "validated_statistics_only")
            cell = payload["contrasts"][0]["cells"][0]
            self.assertEqual(cell["coverage"]["seeds"], [1, 2, 3])
            self.assertAlmostEqual(
                cell["mean_log_ratio"],
                float(np.mean([math.log(1.0 / 1.5), math.log(1.2 / 1.5), math.log(0.8 / 1.0)])),
            )
            self.assertIn("noninferiority_pass", cell)
            self.assertIn("superiority_pass", cell)
            self.assertEqual(
                payload["analysis"]["noninferiority_threshold"]["ratio"], 1.01
            )
            self.assertIn("gate_coverage", payload["contrasts"][0])

    def test_missing_seed_pair_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "plan.json"
            toy_path = root / "toy.csv"
            toy_path.write_text("date,value\n2020-01-01,1\n", encoding="utf-8")
            plan = {
                "global": {"features": "M", "seq_len": 4},
                "data_fingerprints": {
                    "toy": {"relative_path": str(toy_path), "sha256": _sha256(toy_path)}
                },
                "phases": [
                    {
                        "name": "treatment_phase",
                        "output_root": str(root / "treatment"),
                        "datasets": ["toy"], "models": ["Treat"],
                        "features": "M", "seq_len": 4,
                        "horizons": [2], "seeds": [1, 2, 3],
                    },
                    {
                        "name": "reference_phase",
                        "output_root": str(root / "reference"),
                        "datasets": ["toy"], "models": ["Ref"],
                        "features": "M", "seq_len": 4,
                        "horizons": [2], "seeds": [1, 2],
                    },
                ],
                "statistical_gate": {
                    "interval_alpha": 0.05,
                    "simultaneous_m_cells": 1,
                    "interval_construction_rule": "Bonferroni adjusted-alpha one-sided t upper bound",
                    "noninferiority_margin": {"ratio": 1.01, "log_threshold": math.log(1.01)},
                    "superiority_threshold": {"ratio": 1.0, "log_threshold": 0.0},
                    "primary_comparisons": [
                        "treatment_phase:Treat::reference_phase:Ref"
                    ],
                },
            }
            plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
            self._make_phase(root / "treatment", plan_path, "treatment_phase", "Treat", {1: 1.0, 2: 1.2, 3: 0.8})
            self._make_phase(root / "reference", plan_path, "reference_phase", "Ref", {1: 1.5, 2: 1.5})
            with self.assertRaises(ValueError):
                build_analysis(
                    plan_path,
                    {
                        "treatment_phase": root / "treatment",
                        "reference_phase": root / "reference",
                    },
                )


def _artifact_hashes_for_test(result_dir: Path) -> dict[str, str]:
    return {
        path.name: _sha256(path)
        for path in (result_dir / "config.json", result_dir / "metrics.npy")
    }


if __name__ == "__main__":
    unittest.main()
