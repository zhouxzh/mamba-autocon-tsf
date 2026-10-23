#!/usr/bin/env python3
"""Recover one interrupted validation-only job from its immutable snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_training_plan import _artifact_hashes, _command, _command_hash, _job_specs, _load_json, _sha256, _utc_now, _validate_plan


TEST_ARTIFACTS = ("metrics.npy", "original_metrics.npy", "stream_metrics.json", "pred.npy", "true.npy")


def _find_setting(result_root: Path, job_id: str) -> Path | None:
    matches = []
    for candidate in sorted(result_root.glob("*")):
        config_path = candidate / "config.json"
        if config_path.is_file():
            try:
                if _load_json(config_path).get("plan_job_id") == job_id:
                    matches.append(candidate)
            except (OSError, ValueError, TypeError):
                pass
    if len(matches) > 1:
        raise RuntimeError(f"multiple result directories match {job_id}")
    return matches[0] if matches else None


def _phase_validation(phase_root: Path, phase: dict, plan_path: Path, manifest_path: Path, source: str) -> None:
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = len(_job_specs(_load_json(plan_path), phase, {}, allow_unresolved=False))
    valid = len(records) == expected and all(
        record.get("status") == "ok"
        and record.get("source_tree_sha256_before") == source
        and record.get("source_tree_sha256_after") == source
        for record in records
    )
    payload = {
        "schema_version": 1,
        "kind": "exploratory_frozen_phase_validation",
        "valid": bool(valid),
        "phase": phase["name"],
        "plan": str(plan_path.resolve()),
        "plan_sha256": _sha256(plan_path),
        "source_tree_sha256": source,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "expected_jobs": expected,
        "manifest_unique_jobs": len({record.get("job_id") for record in records}),
        "malformed_manifest_lines": 0,
        "status_counts": {status: sum(record.get("status") == status for record in records) for status in sorted({record.get("status") for record in records})},
        "errors": [] if valid else ["exploratory_manifest_incomplete"],
        "test_selection_allowed": False,
    }
    path = phase_root / "phase_validation.json"
    if path.exists():
        if _load_json(path) != payload:
            raise RuntimeError(f"immutable phase validation differs: {path}")
        return
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    path.chmod(0o444)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    plan_path = (args.plan if args.plan.is_absolute() else ROOT / args.plan).resolve()
    artifact_root = (args.artifact_root if args.artifact_root.is_absolute() else ROOT / args.artifact_root).resolve()
    plan = _load_json(plan_path)
    errors = _validate_plan(plan)
    if errors:
        raise ValueError("invalid plan: " + "; ".join(errors))
    launch = _load_json(artifact_root / "launch_record.json")
    if launch.get("plan_sha256") != _sha256(plan_path):
        raise ValueError("plan hash differs from the frozen launch")
    phases = {phase["name"]: phase for phase in plan["phases"]}
    if args.phase not in phases:
        raise ValueError(f"unknown phase: {args.phase}")
    phase = phases[args.phase]
    phase_root = artifact_root / phase["name"].replace("_h12_validation", "")
    # The plan's registered root is authoritative; resolve it rather than infer it.
    phase_root = (ROOT / str(phase["output_root"])).resolve()
    provenance = _load_json(phase_root / "provenance.json")
    source_snapshot = Path(provenance["source_snapshot"]).resolve()
    source = str(provenance["source_tree_sha256"])
    if source_snapshot != (artifact_root / "source_snapshot").resolve() or source != launch.get("source_tree_sha256"):
        raise ValueError("immutable source binding differs")
    jobs = [job for job in _job_specs(plan, phase, {}, allow_unresolved=False) if job["dataset"] == args.dataset and int(job["seed"]) == args.seed]
    if len(jobs) != 1:
        raise ValueError("target coordinate is not unique")
    job = jobs[0]
    roots = {
        "results": phase_root / "results",
        "checkpoints": phase_root / "checkpoints",
        "test": phase_root / "test_results",
        "logs": phase_root / "logs",
        "result_log": phase_root / "results.txt",
    }
    job_id, command = _command(
        job, roots, provenance_id=provenance["provenance_id"], run_script=source_snapshot / "run.py"
    )
    manifest_path = roots["logs"] / "job_manifest.jsonl"
    existing = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(record.get("job_id") == job_id and record.get("status") == "ok" for record in existing):
        raise ValueError("target job is already complete")
    log_path = roots["logs"] / (job_id + ".recovery.log")
    started = time.time()
    print(json.dumps({"job_id": job_id, "dataset": args.dataset, "seed": args.seed, "status": "running"}), flush=True)
    with log_path.open("w", encoding="utf-8") as output:
        completed = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, text=True)
    setting_dir = _find_setting(roots["results"], job_id)
    if setting_dir is None:
        raise RuntimeError("recovery did not produce a result directory")
    expected = setting_dir / "training_complete.json"
    if completed.returncode != 0 or not expected.is_file():
        raise RuntimeError(f"recovery failed with returncode={completed.returncode}")
    if any((setting_dir / name).exists() for name in TEST_ARTIFACTS):
        raise RuntimeError("recovery produced a test artifact")
    checkpoint_dir = roots["checkpoints"] / setting_dir.name
    record = {
        "job_id": job_id, "phase": job["phase"], "dataset": job["dataset"], "model": job["model"],
        "pred_len": job["pred_len"], "seed": job["seed"], "command": command,
        "command_sha256": _command_hash(command), "config_hash": job_id.rsplit("_c", 1)[-1],
        "provenance_id": provenance["provenance_id"], "source_tree_sha256_before": source,
        "started_at_utc": _utc_now(), "status": "ok", "log": str(log_path.resolve()),
        "setting_dir": str(setting_dir.resolve()), "expected_artifact": str(expected.resolve()),
        "artifact_hashes": _artifact_hashes(setting_dir, checkpoint_dir), "returncode": completed.returncode,
        "elapsed_sec": round(time.time() - started, 3), "finished_at_utc": _utc_now(),
        "source_tree_sha256_after": source, "recovered": True,
    }
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    _phase_validation(phase_root, phase, plan_path, manifest_path, source)
    print(json.dumps({"job_id": job_id, "status": "ok", "elapsed_sec": record["elapsed_sec"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
