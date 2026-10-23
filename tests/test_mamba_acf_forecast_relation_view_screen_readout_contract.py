"""Contract tests for the validation-only forecast relation-view readout."""

from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import analyze_mamba_acf_forecast_relation_view_screen as analyzer
from scripts.run_training_plan import _load_json, _validate_plan


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "docs/research/mamba_acf_forecast_relation_view_screen_plan_20260903.json"


def _diagnostics():
    return {
        key: {"n": 1, "mean": 0.5, "min": 0.5, "max": 0.5, "last": 0.5}
        for key in analyzer.EXPECTED_RELATION_LOG_KEYS
    }


def _rows():
    offsets = {
        ("encoder", "null"): 0.0,
        ("encoder", "active"): -0.1,
        ("forecast", "null"): 0.1,
        ("forecast", "active"): -0.2,
    }
    rows = []
    for (route, arm), offset in offsets.items():
        for dataset in analyzer.EXPECTED_DATASETS:
            for seed in analyzer.EXPECTED_SEEDS:
                rows.append(
                    {
                        "route": route,
                        "arm": arm,
                        "dataset": dataset,
                        "seed": seed,
                        "horizon": 96,
                        "mse": 1.0 + offset,
                        "mae": 0.5 + offset,
                        "elapsed_sec": 2.0,
                        "sidecar_sha256": "a" * 64,
                        "config_signature": {"common": "same"},
                        "autocon_loss": {"mean": 0.2},
                        "effective_lambda": {"last": 0.03 if arm == "active" else 0.0},
                        "relation_diagnostics": _diagnostics(),
                        "job_id": f"{route}-{arm}-{dataset}-{seed}",
                    }
                )
    return rows


def test_plan_contract_is_registered_and_unlaunched():
    plan = _load_json(PLAN_PATH)
    assert _validate_plan(plan) == []
    assert analyzer._plan_errors(plan) == []
    assert plan["execution"]["authorized"] is False
    assert plan["execution"]["training_started"] is False
    assert sum(plan["registered_job_counts"][name] for name, *_ in analyzer.EXPECTED_PHASES) == 24


def test_build_readout_computes_paired_active_null_and_did():
    plan = _load_json(PLAN_PATH)
    plan = copy.deepcopy(plan)
    plan["_path"] = str(PLAN_PATH)
    result = analyzer.build_readout(plan, _rows())

    pairs = result["pairs"]
    assert len(pairs) == 6
    first = pairs[0]
    assert first["encoder_active_minus_null_mse"] == pytest.approx(-0.1)
    assert first["forecast_active_minus_null_mse"] == pytest.approx(-0.3)
    assert first["difference_in_differences_mse"] == pytest.approx(-0.2)

    assert result["contrasts"]["encoder_active_minus_null"]["mse"]["mean"] == pytest.approx(-0.1)
    assert result["contrasts"]["forecast_active_minus_null"]["mse"]["mean"] == pytest.approx(-0.3)
    assert result["contrasts"]["difference_in_differences"]["mse"]["mean"] == pytest.approx(-0.2)
    assert result["estimands"]["difference_in_differences"]["mse"]["n"] == 6


def test_pairing_rejects_missing_or_duplicate_coordinates():
    rows = _rows()
    with pytest.raises(ValueError, match="exactly the registered 24 rows"):
        analyzer._make_pairs(rows[:-1])
    duplicate = rows + [dict(rows[0])]
    with pytest.raises(ValueError, match="duplicate row coordinate"):
        analyzer._make_pairs(duplicate)


def test_phase_artifact_bindings_require_manifest_path_and_matching_hash_map(tmp_path):
    manifest = tmp_path / "logs" / "job_manifest.jsonl"
    manifest.parent.mkdir()
    manifest.write_text("", encoding="utf-8")
    records = [{"job_id": "job-1", "artifact_hashes": {"config.json": "a"}}]
    base = {
        "manifest": str(manifest),
        "artifact_hashes_by_job": {"job-1": {"config.json": "a"}},
    }
    analyzer._validate_phase_artifact_bindings(base, manifest, "phase", records)

    wrong_path = dict(base, manifest=str(tmp_path / "other.jsonl"))
    with pytest.raises(ValueError, match="another manifest"):
        analyzer._validate_phase_artifact_bindings(wrong_path, manifest, "phase", records)

    wrong_hashes = dict(base, artifact_hashes_by_job={"job-1": {"config.json": "b"}})
    with pytest.raises(ValueError, match="hashes differ"):
        analyzer._validate_phase_artifact_bindings(wrong_hashes, manifest, "phase", records)

    missing_hashes = dict(base)
    missing_hashes.pop("artifact_hashes_by_job")
    with pytest.raises(ValueError, match="hash map is missing"):
        analyzer._validate_phase_artifact_bindings(missing_hashes, manifest, "phase", records)


def test_record_config_hash_is_required_and_must_match(monkeypatch):
    monkeypatch.setattr(analyzer, "_config_hash", lambda job, provenance: "expected")
    registered = {"phase": "phase"}
    with pytest.raises(ValueError, match="config hash is missing"):
        analyzer._validate_record_config_hash({}, registered, "job-1")
    with pytest.raises(ValueError, match="config hash mismatch"):
        analyzer._validate_record_config_hash({"config_hash": "wrong"}, registered, "job-1")
    analyzer._validate_record_config_hash({"config_hash": "expected"}, registered, "job-1")


def test_not_run_cli_does_not_create_readout_files(tmp_path):
    output_json = tmp_path / "readout.json"
    output_md = tmp_path / "readout.md"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/analyze_mamba_acf_forecast_relation_view_screen.py"),
            "--plan",
            str(PLAN_PATH),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 3
    assert "NOT_RUN" in completed.stderr
    assert not output_json.exists()
    assert not output_md.exists()
