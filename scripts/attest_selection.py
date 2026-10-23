#!/usr/bin/env python
"""Attest an already validated AutoCon lambda selection.

The selector output predates the explicit selection-provenance gate.  This
script does not rewrite that JSON; it verifies its binding to the registered
plan, validator output, manifest, and immutable selection snapshot, then
writes a separate sidecar marked ``posthoc_verified``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hash_file_map(root: Path) -> tuple[dict[str, str], str]:
    """Return a canonical hash for a frozen source tree."""
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}:
            files[path.relative_to(root).as_posix()] = _sha256(path)
    return files, _canonical_hash(files)


def _sidecar_path(selection_path: Path) -> Path:
    return selection_path.with_name(f"{selection_path.stem}.provenance.json")


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _phase(plan: dict[str, Any], phase_name: str) -> dict[str, Any]:
    for phase in plan.get("phases", []):
        if phase.get("name") == phase_name:
            return phase
    raise ValueError(f"phase not found in plan: {phase_name}")


def _command_arg(command: list[str], name: str) -> str | None:
    try:
        index = command.index(name)
    except ValueError:
        return None
    return command[index + 1] if index + 1 < len(command) else None


def _artifact_hashes(result_dir: Path, checkpoint_dir: Path | None) -> dict[str, str]:
    names = (
        "config.json", "metrics.npy", "original_metrics.npy", "channel_metrics.npy",
        "pred.npy", "true.npy", "training_complete.json", "acf_prior.npz",
    )
    paths = [result_dir / name for name in names]
    if checkpoint_dir is not None:
        paths.extend([checkpoint_dir / "checkpoint.pth", checkpoint_dir / "loss_logs.json"])
    return {path.name: _sha256(path) for path in paths if path.is_file()}


def _expected_selection(plan: dict[str, Any], phase: dict[str, Any]) -> tuple[set[tuple], set[float]]:
    global_cfg = dict(plan.get("global", {}))
    cfg = dict(global_cfg)
    cfg.update(phase)
    candidates = {float(value) for value in cfg.get("autocon_lambdas", [])}
    if not candidates:
        raise ValueError("selection phase has no registered lambda candidates")
    expected: set[tuple] = set()
    horizons_map = cfg.get("horizons_by_dataset")
    for model in cfg.get("models", []):
        for dataset in cfg.get("datasets", []):
            horizons = horizons_map.get(dataset, []) if horizons_map else cfg.get("horizons", [])
            for horizon in horizons:
                for seed in cfg.get("seeds", []):
                    expected.add((str(model), str(dataset), int(horizon), int(seed)))
    if not expected:
        raise ValueError("selection phase has no registered cells")
    return expected, candidates


def _recompute_selection(
    selection: dict[str, Any],
    validation: dict[str, Any],
    plan: dict[str, Any],
    phase: dict[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    """Recompute the selector criterion from frozen validation trajectories."""
    expected_cells, expected_candidates = _expected_selection(plan, phase)
    records: dict[str, dict[str, Any]] = {}
    malformed = 0
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if record.get("job_id"):
                records[str(record["job_id"])] = record
    if malformed or len(records) != int(validation["expected_jobs"]):
        raise ValueError("manifest is not a complete clean selection manifest")
    rows: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    expected_hashes = validation["artifact_hashes_by_job"]
    for job_id, record in records.items():
        if record.get("status") != "ok" or record.get("autocon_lambda") is None:
            raise ValueError(f"selection manifest has an invalid record: {job_id}")
        command = list(record.get("command", []))
        if "--skip_test" not in command:
            raise ValueError(f"selection record did not disable test evaluation: {job_id}")
        checkpoint_root = _command_arg(command, "--checkpoints")
        result_root = _command_arg(command, "--results_dir")
        artifact_text = record.get("expected_artifact")
        if not checkpoint_root or not result_root or not artifact_text:
            raise ValueError(f"selection record has incomplete paths: {job_id}")
        result_dir = Path(str(artifact_text)).resolve().parent
        checkpoint_dir = Path(checkpoint_root).resolve() / result_dir.name
        actual_hashes = _artifact_hashes(result_dir, checkpoint_dir)
        if actual_hashes != expected_hashes.get(job_id):
            raise ValueError(f"selection artifact hash mismatch: {job_id}")
        loss_path = checkpoint_dir / "loss_logs.json"
        if not loss_path.is_file():
            raise ValueError(f"selection validation trajectory is missing: {job_id}")
        logs = _read_json(loss_path)
        values = [float(value) for value in logs.get("val_loss", [])]
        if not values or any(not math.isfinite(value) for value in values):
            raise ValueError(f"selection validation trajectory is invalid: {job_id}")
        lambda_value = float(record["autocon_lambda"])
        if not math.isfinite(lambda_value) or lambda_value not in expected_candidates:
            raise ValueError(f"unregistered selection lambda: {job_id}")
        coordinate = (str(record["model"]), str(record["dataset"]), int(record["pred_len"]), int(record["seed"]))
        if coordinate not in expected_cells:
            raise ValueError(f"unexpected selection coordinate: {coordinate!r}")
        key = (*coordinate, lambda_value)
        if key in seen:
            raise ValueError(f"duplicate selection coordinate: {key!r}")
        seen.add(key)
        rows.append({
            "model": coordinate[0], "dataset": coordinate[1],
            "pred_len": coordinate[2], "seed": coordinate[3],
            "lambda": lambda_value, "best_val_loss": min(values),
        })
    cells: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        cells.setdefault((row["model"], row["dataset"], row["pred_len"], row["seed"]), []).append(row)
    if set(cells) != expected_cells or any({row["lambda"] for row in values} != expected_candidates for values in cells.values()):
        raise ValueError("selection trajectory coverage does not match the registered plan")
    normalized: list[dict[str, Any]] = []
    for values in cells.values():
        mean_loss = sum(row["best_val_loss"] for row in values) / len(values)
        if not math.isfinite(mean_loss) or mean_loss <= 0:
            raise ValueError("selection cell has invalid mean validation loss")
        normalized.extend({**row, "relative_val_loss": row["best_val_loss"] / mean_loss} for row in values)
    by_candidate: dict[tuple, list[float]] = {}
    for row in normalized:
        by_candidate.setdefault((row["model"], row["lambda"]), []).append(row["relative_val_loss"])
    candidates: dict[str, list[dict[str, Any]]] = {}
    for (model, lambda_value), values in by_candidate.items():
        candidates.setdefault(model, []).append({
            "lambda": lambda_value,
            "mean_relative_val_loss": sum(values) / len(values),
            "n_jobs": len(values),
        })
    recomputed_selected: dict[str, float] = {}
    for model, values in sorted(candidates.items()):
        values.sort(key=lambda item: (item["mean_relative_val_loss"], item["lambda"]))
        recomputed_selected[model] = values[0]["lambda"]
    original_rows = selection.get("rows", [])
    if len(original_rows) != len(rows):
        raise ValueError("selection JSON trajectory count differs from manifest replay")
    rows_match = all(left == right for left, right in zip(original_rows, rows))
    if recomputed_selected != {str(k): float(v) for k, v in selection["selected_lambda_by_model"].items()}:
        raise ValueError("selection mapping does not match recomputed validation criterion")
    original_candidates = selection.get("candidates", {})
    max_difference = 0.0
    for model, values in candidates.items():
        old_values = {float(item["lambda"]): float(item["mean_relative_val_loss"]) for item in original_candidates.get(model, [])}
        for item in values:
            if float(item["lambda"]) not in old_values:
                raise ValueError(f"selection candidate summary is incomplete for {model}")
            max_difference = max(max_difference, abs(float(item["mean_relative_val_loss"]) - old_values[float(item["lambda"])]))
    canonical_rows = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return {
        "algorithm": "independent_validation_trajectory_recompute",
        "mapping_matches": True,
        "rows_match": rows_match,
        "manifest_rows": len(rows),
        "expected_cells": len(expected_cells),
        "candidate_max_abs_difference": max_difference,
        "candidate_summary": candidates,
        "recomputed_rows_sha256": hashlib.sha256(canonical_rows).hexdigest(),
    }


def _validate(
    selection_path: Path,
    validation_path: Path,
    plan_path: Path,
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    selection = _read_json(selection_path)
    validation = _read_json(validation_path)
    plan = _read_json(plan_path)
    if validation.get("valid") is not True:
        raise ValueError("selection phase validation is not valid")
    if validation.get("legacy_unknown_provenance") is not False:
        raise ValueError("selection validation has legacy or unknown provenance")
    phase_name = str(selection.get("phase", ""))
    if not phase_name:
        raise ValueError("selection JSON has no phase")
    phase = _phase(plan, phase_name)
    if phase.get("autocon_lambda") == "selected":
        raise ValueError("cannot attest a confirmatory selection binding as a selection phase")
    expected_features = phase.get("features", plan.get("global", {}).get("features", "S"))
    if selection.get("features", expected_features) != expected_features:
        raise ValueError("selection features do not match the registered phase")
    if validation.get("phase") != phase_name:
        raise ValueError("validation phase does not match selection phase")
    if Path(str(validation.get("manifest", ""))).resolve() != manifest_path:
        raise ValueError("validation refers to a different manifest")
    if validation.get("manifest_sha256") != _sha256(manifest_path):
        raise ValueError("manifest changed after validation")
    if validation.get("plan_sha256") != _sha256(plan_path):
        raise ValueError("plan changed after validation")
    expected_jobs = int(validation.get("expected_jobs", -1))
    unique_jobs = int(validation.get("manifest_unique_jobs", -1))
    if expected_jobs <= 0 or unique_jobs != expected_jobs:
        raise ValueError("selection validation coverage is incomplete")
    if validation.get("errors") != []:
        raise ValueError("selection validation contains errors")
    if int(validation.get("malformed_manifest_lines", -1)) != 0:
        raise ValueError("selection manifest has malformed lines")
    if validation.get("status_counts") != {"ok": expected_jobs}:
        raise ValueError("selection manifest does not contain exactly the expected successful jobs")
    artifact_hashes = validation.get("artifact_hashes_by_job")
    if not isinstance(artifact_hashes, dict) or len(artifact_hashes) != expected_jobs:
        raise ValueError("selection validation artifact-hash coverage is incomplete")

    selected = selection.get("selected_lambda_by_model")
    if not isinstance(selected, dict) or not selected:
        raise ValueError("selection JSON has no selected_lambda_by_model mapping")
    registered_models = {str(model) for model in phase.get("models", [])}
    expected_models = {model for model in registered_models if model.endswith("AutoCon")}
    if set(selected) != expected_models:
        raise ValueError(f"selected model keys do not match registered AutoCon models: {sorted(selected)}")
    candidates = {float(value) for value in phase.get("autocon_lambdas", [])}
    for model, value in selected.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"non-numeric selected lambda for {model}") from exc
        if not math.isfinite(numeric) or numeric <= 0 or numeric not in candidates:
            raise ValueError(f"selected lambda for {model} is not a registered finite candidate: {value!r}")
    if int(selection.get("n_trajectories", -1)) != expected_jobs:
        raise ValueError("selection JSON trajectory count does not match validation coverage")

    provenance_path = Path(str(validation.get("provenance", ""))).resolve()
    if not provenance_path.is_file():
        raise ValueError("selection provenance sidecar is missing")
    provenance = _read_json(provenance_path)
    if provenance.get("phase") != phase_name or provenance.get("provenance_id") != validation.get("provenance_id"):
        raise ValueError("selection provenance does not match validator output")
    snapshot = Path(str(provenance.get("source_snapshot", ""))).resolve()
    selector_snapshot = snapshot / "scripts/select_hyperparameters.py"
    if not selector_snapshot.is_file():
        raise ValueError("selection snapshot does not contain select_hyperparameters.py")
    return selection, validation, plan, {
        "phase": phase,
        "phase_name": phase_name,
        "expected_jobs": expected_jobs,
        "provenance": provenance,
        "provenance_path": provenance_path,
        "snapshot": snapshot,
        "selector_snapshot": selector_snapshot,
    }


def _require_equal(sidecar: dict[str, Any], key: str, expected: Any) -> None:
    if sidecar.get(key) != expected:
        raise ValueError(f"selection provenance {key} does not match its frozen inputs")


def _require_replay_equal(recorded: dict[str, Any], recomputed: dict[str, Any]) -> None:
    exact_keys = (
        "algorithm", "mapping_matches", "rows_match", "manifest_rows",
        "expected_cells", "recomputed_rows_sha256",
    )
    for key in exact_keys:
        if recorded.get(key) != recomputed.get(key):
            raise ValueError(f"selection provenance replay {key} does not match recomputation")
    if recorded.get("mapping_matches") is not True:
        raise ValueError("selection provenance replay mapping did not match recomputation")
    if recorded.get("rows_match") is not True:
        raise ValueError("selection provenance replay rows did not match recomputation")
    try:
        difference_matches = math.isclose(
            float(recorded.get("candidate_max_abs_difference")),
            float(recomputed.get("candidate_max_abs_difference")),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        difference_is_small = float(recorded.get("candidate_max_abs_difference")) <= 1e-12
    except (TypeError, ValueError):
        difference_matches = False
        difference_is_small = False
    if not difference_matches:
        raise ValueError("selection provenance replay candidate difference does not match recomputation")
    if not difference_is_small:
        raise ValueError("selection provenance replay candidate difference is too large")

    recorded_summary = recorded.get("candidate_summary")
    recomputed_summary = recomputed.get("candidate_summary")
    if not isinstance(recorded_summary, dict) or not isinstance(recomputed_summary, dict):
        raise ValueError("selection provenance replay candidate summary is missing")
    if set(recorded_summary) != set(recomputed_summary):
        raise ValueError("selection provenance replay candidate model set does not match")
    for model in sorted(recomputed_summary):
        recorded_candidates = recorded_summary[model]
        recomputed_candidates = recomputed_summary[model]
        if not isinstance(recorded_candidates, list) or not isinstance(recomputed_candidates, list):
            raise ValueError("selection provenance replay candidate summary is malformed")
        def by_lambda(items: list[dict[str, Any]]) -> dict[float, dict[str, Any]]:
            result: dict[float, dict[str, Any]] = {}
            for item in items:
                try:
                    lambda_value = float(item["lambda"])
                    n_jobs = int(item["n_jobs"])
                    mean_loss = float(item["mean_relative_val_loss"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError("selection provenance replay candidate summary is malformed") from exc
                if not math.isfinite(lambda_value) or not math.isfinite(mean_loss):
                    raise ValueError("selection provenance replay candidate summary is non-finite")
                if lambda_value in result:
                    raise ValueError("selection provenance replay candidate summary has duplicate lambdas")
                result[lambda_value] = {"n_jobs": n_jobs, "mean_relative_val_loss": mean_loss}
            return result

        recorded_by_lambda = by_lambda(recorded_candidates)
        recomputed_by_lambda = by_lambda(recomputed_candidates)
        if set(recorded_by_lambda) != set(recomputed_by_lambda):
            raise ValueError(f"selection provenance replay lambda set does not match for {model}")
        for lambda_value, expected_item in recomputed_by_lambda.items():
            observed_item = recorded_by_lambda[lambda_value]
            if observed_item["n_jobs"] != expected_item["n_jobs"]:
                raise ValueError(f"selection provenance replay job count does not match for {model}/{lambda_value}")
            if not math.isclose(
                observed_item["mean_relative_val_loss"],
                expected_item["mean_relative_val_loss"],
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"selection provenance replay mean relative loss does not match for {model}/{lambda_value}"
                )


def verify_selection_attestation(
    selection_path: Path,
    plan_path: Path,
    confirmatory_phase: dict[str, Any],
    selection_provenance_path: Path | None = None,
) -> dict[str, Any]:
    """Verify a selection sidecar before launching a confirmatory phase.

    This performs a fresh replay of all frozen validation trajectories.  The
    returned compact binding is suitable for inclusion in a confirmatory
    phase's immutable provenance record.
    """
    selection_path = _resolve(selection_path)
    plan_path = _resolve(plan_path)
    for path in (selection_path, plan_path):
        if not path.is_file():
            raise ValueError(f"missing selection binding input: {path}")
    selection_stub = _read_json(selection_path)
    selection_phase_name = str(selection_stub.get("phase", ""))
    expected_phase_name = str(confirmatory_phase.get("selection_phase", ""))
    if selection_phase_name != expected_phase_name:
        raise ValueError(
            f"confirmatory phase requires selection phase {expected_phase_name!r}; "
            f"received {selection_phase_name!r}"
        )
    plan_stub = _read_json(plan_path)
    selection_phase = _phase(plan_stub, selection_phase_name)
    selection_root = _resolve(Path(selection_phase.get("output_root", "")))
    expected_selection_path = selection_root / "selected_hyperparameters.json"
    if selection_path != expected_selection_path:
        raise ValueError("selection file is not the registered selection artifact")
    validation_path = selection_root / "phase_validation.json"
    manifest_path = selection_root / "logs" / "job_manifest.jsonl"
    sidecar_path = _sidecar_path(selection_path)
    if selection_provenance_path is not None:
        supplied_sidecar = _resolve(selection_provenance_path)
        if supplied_sidecar != sidecar_path:
            raise ValueError("selection provenance file is not the registered sidecar")
    for path in (validation_path, manifest_path, sidecar_path):
        if not path.is_file():
            raise ValueError(f"selection binding artifact is missing: {path}")

    selection, validation, plan, context = _validate(
        selection_path, validation_path, plan_path, manifest_path
    )
    expected_features = confirmatory_phase.get("features", plan.get("global", {}).get("features", "S"))
    if selection.get("features") != expected_features:
        raise ValueError("selection features do not match confirmatory phase features")
    expected_models = {
        str(model) for model in confirmatory_phase.get("models", []) if str(model).endswith("AutoCon")
    }
    selected = selection.get("selected_lambda_by_model", {})
    if set(selected) != expected_models:
        raise ValueError("selected AutoCon models do not match the confirmatory model set")

    frozen_files, frozen_tree_hash = _hash_file_map(context["snapshot"])
    if (frozen_files != context["provenance"].get("source_files")
            or frozen_tree_hash != context["provenance"].get("source_tree_sha256")):
        raise ValueError("selection source snapshot changed")
    replay = _recompute_selection(selection, validation, plan, context["phase"], manifest_path)
    sidecar = _read_json(sidecar_path)
    if sidecar.get("schema_version") != 1 or sidecar.get("kind") != "autocon_lambda_selection_provenance":
        raise ValueError("selection provenance sidecar has an unsupported schema")
    if sidecar.get("attestation_mode") != "posthoc_verified":
        raise ValueError("selection provenance sidecar is not a posthoc verification")
    if sidecar.get("publication_ready") is not False:
        raise ValueError("selection provenance must remain non-publication-ready")
    expected_fields = {
        "selection_path": str(selection_path),
        "selection_sha256": _sha256(selection_path),
        "selected_lambda_by_model_sha256": _canonical_hash(selected),
        "selection_phase": selection_phase_name,
        "selection_features": expected_features,
        "selection_plan": str(plan_path),
        "selection_plan_sha256": _sha256(plan_path),
        "phase_validation_path": str(validation_path),
        "phase_validation_sha256": _sha256(validation_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "validation_provenance_path": str(context["provenance_path"]),
        "validation_provenance_id": context["provenance"].get("provenance_id"),
        "validation_expected_jobs": context["expected_jobs"],
        "validation_manifest_unique_jobs": int(validation["manifest_unique_jobs"]),
        "selector_snapshot_path": str(context["selector_snapshot"]),
        "selector_snapshot_sha256": _sha256(context["selector_snapshot"]),
        "selector_sha256": _sha256(context["selector_snapshot"]),
        "selected_lambda_by_model": selected,
        "attester_path": str(Path(__file__).resolve()),
        "attester_sha256": _sha256(Path(__file__).resolve()),
    }
    for key, value in expected_fields.items():
        _require_equal(sidecar, key, value)
    _require_replay_equal(sidecar.get("replay", {}), replay)
    return {
        "schema_version": 1,
        "binding_mode": "prelaunch_replay_of_posthoc_attestation",
        "selection_provenance_path": str(sidecar_path),
        "selection_provenance_sha256": _sha256(sidecar_path),
        "selection_provenance_mode": sidecar["attestation_mode"],
        "selection_provenance_publication_ready": sidecar["publication_ready"],
        "selected_lambda_by_model_sha256": _canonical_hash(selected),
        "selection_phase_validation_sha256": _sha256(validation_path),
        "selection_manifest_sha256": _sha256(manifest_path),
        "selection_provenance_id": context["provenance"].get("provenance_id"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None,
                        help="sidecar path; defaults to <selection>.provenance.json")
    args = parser.parse_args()

    selection_path = _resolve(args.selection)
    validation_path = _resolve(args.validation)
    plan_path = _resolve(args.plan)
    manifest_path = _resolve(args.manifest)
    for path in (selection_path, validation_path, plan_path, manifest_path):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    try:
        selection, validation, plan, context = _validate(
            selection_path, validation_path, plan_path, manifest_path
        )
        replay = _recompute_selection(
            selection, validation, plan, context["phase"], manifest_path
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"selection attestation failed: {exc}") from exc

    selected = selection["selected_lambda_by_model"]
    selector_live = REPO_ROOT / "scripts/select_hyperparameters.py"
    sidecar = _resolve(args.output) if args.output is not None else selection_path.with_name(
        f"{selection_path.stem}.provenance.json"
    )
    payload = {
        "schema_version": 1,
        "kind": "autocon_lambda_selection_provenance",
        "attestation_mode": "posthoc_verified",
        "publication_ready": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_path": str(selection_path),
        "selection_sha256": _sha256(selection_path),
        "selected_lambda_by_model_sha256": _canonical_hash(selected),
        "selection_phase": context["phase_name"],
        "selection_features": selection.get("features"),
        "selection_plan": str(plan_path),
        "selection_plan_sha256": _sha256(plan_path),
        "phase_validation_path": str(validation_path),
        "phase_validation_sha256": _sha256(validation_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "validation_provenance_path": str(context["provenance_path"]),
        "validation_provenance_id": context["provenance"].get("provenance_id"),
        "validation_expected_jobs": context["expected_jobs"],
        "validation_manifest_unique_jobs": int(validation["manifest_unique_jobs"]),
        "selector_snapshot_path": str(context["selector_snapshot"]),
        "selector_snapshot_sha256": _sha256(context["selector_snapshot"]),
        "selector_path": str(selector_live),
        "selector_sha256": _sha256(selector_live),
        "selected_lambda_by_model": selected,
        "replay": replay,
        "attester_path": str(Path(__file__).resolve()),
        "attester_sha256": _sha256(Path(__file__).resolve()),
    }
    _atomic_write(sidecar, payload)
    print(json.dumps({
        "sidecar": str(sidecar),
        "attestation_mode": payload["attestation_mode"],
        "publication_ready": payload["publication_ready"],
        "selection_sha256": payload["selection_sha256"],
        "selected_lambda_by_model_sha256": payload["selected_lambda_by_model_sha256"],
    }, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
