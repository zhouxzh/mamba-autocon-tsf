#!/usr/bin/env python3
"""Audit the unapproved latent-temporal high-dimensional validation plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_LATENCIES = {
    "electricity": (0.3762720078229904, 0.42552000284194946),
    "traffic": (0.5037599802017212, 0.8727359771728516),
}


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("JSON object required: {}".format(path))
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _records(path: Path) -> dict[tuple[str, str], dict]:
    payload = _load(path)
    result = {}
    for record in payload.get("resource_records", []):
        key = (str(record.get("dataset")), str(record.get("model")))
        if key in result:
            raise ValueError("duplicate resource record: {}".format(key))
        result[key] = record
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan_path = args.plan.resolve() if args.plan.is_absolute() else (ROOT / args.plan).resolve()
    plan = _load(plan_path)
    errors = []
    execution = plan.get("execution", {})
    counts = plan.get("registered_job_counts", {})
    if plan.get("status") != "exploratory_latent_temporal_mamba_highdim_autocon_validation":
        errors.append("unexpected_plan_status")
    if execution.get("authorized") is not False or execution.get("training_started") is not False:
        errors.append("plan_must_remain_unapproved_and_unstarted")
    if counts.get("total") != 60 or counts.get("training_total") != 60:
        errors.append("registered_job_count_mismatch")
    phases = {phase.get("name"): phase for phase in plan.get("phases", [])}
    expected_phase_names = {
        "latent_temporal_null_h96_validation", "latent_temporal_active_h96_validation",
        "mambatsv3_reference_h96_validation", "strong_baselines_h96_validation",
    }
    if set(phases) != expected_phase_names:
        errors.append("phase_set_mismatch")
    for phase in phases.values():
        if phase.get("skip_test") is not True or phase.get("stream_metrics") is not False:
            errors.append("validation_only_boundary_mismatch")
            break
    if phases.get("latent_temporal_null_h96_validation", {}).get("autocon_lambda") != 0.0:
        errors.append("null_lambda_mismatch")
    if phases.get("latent_temporal_active_h96_validation", {}).get("autocon_lambda") != 0.001:
        errors.append("active_lambda_mismatch")

    resource = plan.get("resource_preflight", {})
    resource_paths = {
        "latent_electricity": (resource.get("electricity_resource_json"), resource.get("electricity_resource_sha256")),
        "latent_traffic": (resource.get("traffic_resource_json"), resource.get("traffic_resource_sha256")),
        "stride24_reference_electricity": (resource.get("stride24_reference_electricity_resource_json"), resource.get("stride24_reference_electricity_resource_sha256")),
        "stride24_reference_traffic": (resource.get("stride24_reference_traffic_resource_json"), resource.get("stride24_reference_traffic_resource_sha256")),
    }
    resolved = {}
    for name, (text_path, expected_hash) in resource_paths.items():
        path = ROOT / str(text_path)
        if not path.is_file() or _sha256(path) != expected_hash:
            errors.append("resource_hash_mismatch:{}".format(name))
        else:
            resolved[name] = path
    observed = {}
    if len(resolved) == len(resource_paths):
        latent_electricity = _records(resolved["latent_electricity"])
        latent_traffic = _records(resolved["latent_traffic"])
        for dataset, (candidate_expected, reference_expected) in EXPECTED_LATENCIES.items():
            candidate_records = latent_electricity if dataset == "electricity" else latent_traffic
            candidate = candidate_records.get((dataset, "LatentTemporalMambaAutoCon"))
            reference = _records(resolved["stride24_reference_" + dataset])
            baseline = reference.get((dataset, "MambaTSv3AutoCon"))
            if not candidate or not baseline:
                errors.append("resource_coordinate_missing:{}".format(dataset))
                continue
            candidate_latency = float(candidate.get("gpu_inference", {}).get("latency_ms_median", float("nan")))
            baseline_latency = float(baseline.get("gpu_inference", {}).get("latency_ms_median", float("nan")))
            observed[dataset] = {"latent_ms": candidate_latency, "stride24_ms": baseline_latency}
            if candidate.get("gpu_inference", {}).get("status") != "ok" or baseline.get("gpu_inference", {}).get("status") != "ok":
                errors.append("resource_status_not_ok:{}".format(dataset))
            for record, label in ((candidate, "candidate"), (baseline, "reference")):
                gpu = record.get("gpu_inference", {})
                if (
                    gpu.get("forward_path") != "prediction_plus_auxiliary_representation"
                    or gpu.get("auxiliary_representation_requested") is not True
                    or gpu.get("auxiliary_representation_controlled") is not True
                    or gpu.get("auxiliary_representation_returned") is not True
                    or gpu.get("auxiliary_representation_shape") != [32, 96, 64]
                ):
                    errors.append("representation_contract_failed:{}:{}".format(dataset, label))
            if abs(candidate_latency - candidate_expected) > 1e-6 or abs(baseline_latency - reference_expected) > 1e-6:
                errors.append("resource_latency_binding_mismatch:{}".format(dataset))
            if not candidate_latency < baseline_latency:
                errors.append("speed_gate_failed:{}".format(dataset))

    payload = {
        "status": "review_packet_valid" if not errors else "review_packet_invalid",
        "plan": str(plan_path), "plan_sha256": _sha256(plan_path),
        "authorized": execution.get("authorized"), "launch_started": False,
        "observed_latencies_ms": observed, "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "errors": len(errors)}, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
