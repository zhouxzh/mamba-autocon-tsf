#!/usr/bin/env python3
"""Run a narrowly scoped exploratory plan from an immutable source snapshot.

This launcher is intentionally separate from the confirmation runner.  It
accepts only a plan whose execution metadata is explicitly exploratory, never
turns an exploratory plan into an authorized confirmation, and records a
read-only source snapshot plus per-phase provenance before starting jobs.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_training_plan import (
    DATASETS,
    _artifact_exists,
    _artifact_hashes,
    _command,
    _command_hash,
    _copy_source_snapshot,
    _environment,
    _hash_file_map,
    _job_specs,
    _load_json,
    _sha256,
    _utc_now,
    _validate_plan,
)


def _find_setting(result_root: Path, job_id: str) -> Path | None:
    matches = []
    for candidate in sorted(result_root.glob("*")):
        config_path = candidate / "config.json"
        if not config_path.is_file():
            continue
        try:
            config = _load_json(config_path)
        except (OSError, ValueError, TypeError):
            continue
        if config.get("plan_job_id") == job_id:
            matches.append(candidate)
    if len(matches) > 1:
        raise RuntimeError("multiple result directories match {}".format(job_id))
    return matches[0] if matches else None


def _relocate_phase_output_roots(plan: dict, artifact_root: Path) -> dict:
    """Return a plan copy whose phase roots live below ``artifact_root``.

    ``--artifact-root`` is a safety boundary for bounded exploratory runs.  A
    source snapshot alone is insufficient if generated commands still carry
    the original plan roots, because a smoke can then write into a registered
    matrix.  Preserve the phase-relative layout below the common original
    parent while replacing that parent with the requested fresh root.
    """

    phases = plan.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError("plan has no phases/output roots")
    original_roots = []
    for phase in phases:
        if not isinstance(phase, dict) or not phase.get("output_root"):
            raise ValueError("every phase must define output_root")
        raw = Path(str(phase["output_root"]))
        original_roots.append((raw if raw.is_absolute() else ROOT / raw).resolve())
    common_parent = Path(os.path.commonpath([str(path) for path in original_roots]))
    relocated = copy.deepcopy(plan)
    for phase, original in zip(relocated["phases"], original_roots):
        try:
            relative = original.relative_to(common_parent)
        except ValueError as exc:
            raise ValueError("phase output roots cannot be relocated") from exc
        phase["output_root"] = str((artifact_root / relative).resolve())
    return relocated


def _phase_provenance(phase_root: Path, plan_path: Path, plan: dict,
                      phase: dict, source_snapshot: Path, source_files: dict,
                      source_hash: str, env: dict) -> Path:
    phase_root.mkdir(parents=True, exist_ok=True)
    path = phase_root / "provenance.json"
    payload = {
        "kind": "exploratory_frozen_plan_provenance",
        "schema_version": 1,
        "plan": str(plan_path.resolve()),
        "plan_sha256": _sha256(plan_path),
        "plan_status": plan.get("status"),
        "phase": phase.get("name"),
        "provenance_id": "exploratory-" + source_hash[:16],
        "source_snapshot": str(source_snapshot.resolve()),
        "source_tree_sha256": source_hash,
        "source_files": source_files,
        "dataset_hashes": {
            name: _sha256(ROOT / DATASETS[name]["root"] / DATASETS[name]["path"])
            for name in phase.get("datasets", [])
        },
        "environment": env,
        "test_selection_allowed": False,
        "created_at_utc": _utc_now(),
    }
    if path.exists():
        existing = _load_json(path)
        if existing != payload:
            raise RuntimeError("immutable provenance already differs: {}".format(path))
        return path
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    path.chmod(0o444)
    return path


def _write_environment(path: Path, plan_path: Path, phase: dict, env: dict,
                       provenance: Path) -> None:
    if path.exists():
        return
    payload = {
        "plan": str(plan_path.resolve()),
        "phase": phase.get("name"),
        "environment": env,
        "provenance": str(provenance.resolve()),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _write_phase_validation(phase_root: Path, phase: dict, manifest_path: Path,
                            expected_jobs: int, plan_path: Path,
                            source_hash: str) -> None:
    """Persist a small immutable validation record for exploratory phases.

    The full confirmation validator has stricter launch-binding semantics and
    is intentionally not used by this unauthorized launcher.  Exploratory
    readers still need a stable manifest hash, coverage count, and status
    summary before consuming a phase, so record that narrow contract here.
    """
    lines = manifest_path.read_text(encoding="utf-8").splitlines() if manifest_path.is_file() else []
    records = []
    malformed = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if isinstance(value, dict):
            records.append(value)
        else:
            malformed += 1
    status_counts = {}
    for record in records:
        status = str(record.get("status", "missing"))
        status_counts[status] = status_counts.get(status, 0) + 1
    valid = (
        malformed == 0
        and len(records) == int(expected_jobs)
        and status_counts == {"ok": int(expected_jobs)}
        and all(record.get("source_tree_sha256_before") == source_hash for record in records)
    )
    payload = {
        "schema_version": 1,
        "kind": "exploratory_frozen_phase_validation",
        "valid": bool(valid),
        "phase": phase.get("name"),
        "plan": str(plan_path.resolve()),
        "plan_sha256": _sha256(plan_path),
        "source_tree_sha256": source_hash,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path) if manifest_path.is_file() else None,
        "expected_jobs": int(expected_jobs),
        "manifest_unique_jobs": len({record.get("job_id") for record in records}),
        "malformed_manifest_lines": malformed,
        "status_counts": status_counts,
        "errors": [] if valid else ["exploratory_manifest_incomplete"],
        "test_selection_allowed": False,
    }
    path = phase_root / "phase_validation.json"
    if path.exists():
        existing = _load_json(path)
        if existing != payload:
            raise RuntimeError("immutable phase validation already differs: {}".format(path))
        return
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    path.chmod(0o444)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("docs/research/mamba_acf_anchor_current_test_replication_plan_20260904.json"),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--artifact-root", type=Path, default=None,
        help="fresh exploratory output root; defaults to the common phase root",
    )
    args = parser.parse_args()

    plan_path = args.plan if args.plan.is_absolute() else ROOT / args.plan
    plan_path = plan_path.resolve()
    plan = _load_json(plan_path)
    errors = _validate_plan(plan)
    if errors:
        for error in errors:
            print("PLAN ERROR:", error, file=sys.stderr)
        return 2
    execution = plan.get("execution")
    status = str(plan.get("status", ""))
    if not status.startswith("exploratory_") or not isinstance(execution, dict):
        print("PLAN ERROR: launcher accepts exploratory plans only", file=sys.stderr)
        return 2
    if execution.get("authorized") is not False or execution.get("training_started") is not False:
        print("PLAN ERROR: exploratory plan must remain unauthorized/unstarted", file=sys.stderr)
        return 2
    global_cfg = plan.get("global", {})
    skip_test = global_cfg.get("skip_test")
    stream_metrics = global_cfg.get("stream_metrics")
    if not isinstance(skip_test, bool) or not isinstance(stream_metrics, bool):
        print("PLAN ERROR: global skip_test and stream_metrics must be booleans", file=sys.stderr)
        return 2
    if skip_test and stream_metrics:
        print("PLAN ERROR: validation-only frozen runs cannot enable stream_metrics", file=sys.stderr)
        return 2
    if global_cfg.get("save_predictions") is not False:
        print("PLAN ERROR: frozen exploratory launcher requires save_predictions=false", file=sys.stderr)
        return 2
    if args.limit < 0:
        print("PLAN ERROR: --limit must be non-negative", file=sys.stderr)
        return 2

    if args.artifact_root is not None:
        artifact_root = args.artifact_root if args.artifact_root.is_absolute() else ROOT / args.artifact_root
    else:
        output_roots = [ROOT / str(phase.get("output_root", "")) for phase in plan.get("phases", [])]
        if not output_roots:
            print("PLAN ERROR: plan has no phases/output roots", file=sys.stderr)
            return 2
        try:
            artifact_root = output_roots[0].parent
            if any(root.parent != artifact_root for root in output_roots):
                raise ValueError
        except ValueError:
            print("PLAN ERROR: phase output roots must share one parent or pass --artifact-root", file=sys.stderr)
            return 2
    artifact_root = artifact_root.resolve()
    effective_plan = (
        _relocate_phase_output_roots(plan, artifact_root)
        if args.artifact_root is not None else plan
    )
    snapshot_root = artifact_root / "source_snapshot"
    if snapshot_root.exists():
        print("PLAN ERROR: existing output root cannot be resumed by this frozen launcher", file=sys.stderr)
        return 2
    artifact_root.mkdir(parents=True, exist_ok=False)
    source_files, source_hash = _copy_source_snapshot(snapshot_root)
    env = _environment()
    (artifact_root / "launch_record.json").write_text(
        json.dumps({
            "launcher": str(Path(__file__).resolve()),
            "launcher_sha256": _sha256(Path(__file__).resolve()),
            "plan": str(plan_path),
            "plan_sha256": _sha256(plan_path),
            "source_snapshot": str(snapshot_root.resolve()),
            "source_tree_sha256": source_hash,
            "source_files": source_files,
            "phase_output_roots": {
                str(phase.get("name")): str(phase.get("output_root"))
                for phase in effective_plan.get("phases", [])
            },
            "environment": env,
            "started_at_utc": _utc_now(),
            "exploratory_only": True,
        }, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    all_jobs = []
    for phase in effective_plan["phases"]:
        all_jobs.extend(_job_specs(effective_plan, phase, {}, allow_unresolved=False))
    random.Random(int(plan.get("job_order_seed", 1729))).shuffle(all_jobs)
    if args.limit:
        all_jobs = all_jobs[: args.limit]
    print("Validated {} jobs (order_seed={})".format(
        len(all_jobs), plan.get("job_order_seed", 1729)
    ), flush=True)

    failed = 0
    provenance_id = "exploratory-" + source_hash[:16]
    for phase in effective_plan["phases"]:
        phase_jobs = [job for job in all_jobs if job["phase"] == phase["name"]]
        if not phase_jobs:
            continue
        phase_root = ROOT / str(phase["output_root"])
        roots = {
            "results": phase_root / "results",
            "checkpoints": phase_root / "checkpoints",
            "test": phase_root / "test_results",
            "logs": phase_root / "logs",
            "result_log": phase_root / "results.txt",
        }
        for path in roots.values():
            path.parent.mkdir(parents=True, exist_ok=True) if path.suffix else path.mkdir(parents=True, exist_ok=True)
        provenance = _phase_provenance(
            phase_root, plan_path, effective_plan, phase, snapshot_root, source_files,
            source_hash, env,
        )
        _write_environment(phase_root / "environment.json", plan_path, phase, env, provenance)
        manifest_path = roots["logs"] / "job_manifest.jsonl"
        previous = {}
        if manifest_path.exists():
            for line in manifest_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    previous[row["job_id"]] = row
        with manifest_path.open("a", encoding="utf-8") as manifest:
            for index, job in enumerate(phase_jobs, 1):
                job_id, command = _command(
                    job, roots, provenance_id=provenance_id,
                    run_script=snapshot_root / "run.py",
                )
                prior = previous.get(job_id)
                if (
                    not args.no_resume and prior and prior.get("status") == "ok"
                    and _artifact_exists(prior)
                ):
                    print("[{}/{}] skip {} (resume)".format(index, len(phase_jobs), job_id), flush=True)
                    continue
                log_path = roots["logs"] / (job_id + ".log")
                record = {
                    "job_id": job_id,
                    "phase": job["phase"],
                    "dataset": job["dataset"],
                    "model": job["model"],
                    "pred_len": job["pred_len"],
                    "seed": job["seed"],
                    "command": command,
                    "command_sha256": _command_hash(command),
                    "config_hash": job_id.rsplit("_c", 1)[-1],
                    "provenance_id": provenance_id,
                    "source_tree_sha256_before": source_hash,
                    "started_at_utc": _utc_now(),
                    "status": "planned",
                    "log": str(log_path.resolve()),
                }
                print("[{}/{}] run {}".format(index, len(phase_jobs), job_id), flush=True)
                started = time.time()
                with log_path.open("w", encoding="utf-8") as output:
                    completed = subprocess.run(
                        command, cwd=ROOT, stdout=output,
                        stderr=subprocess.STDOUT, text=True,
                    )
                setting_dir = _find_setting(roots["results"], job_id)
                expected = None
                if setting_dir is not None:
                    expected_name = "metrics.npy" if not bool(job["config"].get("skip_test", False)) else "training_complete.json"
                    expected = setting_dir / expected_name
                    record["setting_dir"] = str(setting_dir.resolve())
                    record["expected_artifact"] = str(expected.resolve())
                    checkpoint_dir = roots["checkpoints"] / setting_dir.name
                    record["artifact_hashes"] = _artifact_hashes(setting_dir, checkpoint_dir)
                    loss_path = checkpoint_dir / "loss_logs.json"
                    if loss_path.is_file():
                        loss = _load_json(loss_path)
                        record["best_epoch"] = loss.get("best_epoch")
                        record["epochs_ran"] = loss.get("epochs_ran", len(loss.get("val_loss", [])))
                record["returncode"] = completed.returncode
                record["elapsed_sec"] = round(time.time() - started, 3)
                record["finished_at_utc"] = _utc_now()
                record["source_tree_sha256_after"] = source_hash
                record["status"] = "ok" if completed.returncode == 0 and expected is not None and expected.is_file() else "failed"
                if record["status"] != "ok":
                    failed += 1
                manifest.write(json.dumps(record, ensure_ascii=True) + "\n")
                manifest.flush()
                print(json.dumps({
                    "dataset": job["dataset"], "seed": job["seed"],
                    "status": record["status"], "elapsed_sec": record["elapsed_sec"],
                }, ensure_ascii=True), flush=True)
        _write_phase_validation(
            phase_root, phase, manifest_path, len(phase_jobs), plan_path, source_hash
        )
    print("Frozen exploratory plan complete; failed_jobs={}".format(failed), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
