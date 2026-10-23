#!/usr/bin/env python3
"""Strict reader for the relation-auxiliary-only high-dimensional test release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

from scipy import stats


CANDIDATE_MODEL = "MambaTSv3AutoCon"
REQUIRED_DATASETS = ("electricity", "traffic")
TEST_ARTIFACTS = ("metrics.npy", "original_metrics.npy", "pred.npy", "true.npy")


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _merged_phase(plan: dict, phase: dict) -> dict:
    """Resolve the compact plan convention used by the runner."""

    merged = dict(plan.get("global", {}))
    merged.update(phase)
    return merged


def _release_layout(plan: dict) -> dict:
    """Derive strict phase/seed/model roles from a registered release plan."""

    phases = plan.get("phases")
    if not isinstance(phases, list) or len(phases) != 3:
        raise ValueError("high-dimensional release requires exactly three phases")
    resolved = [_merged_phase(plan, phase) for phase in phases if isinstance(phase, dict)]
    if len(resolved) != 3:
        raise ValueError("high-dimensional release phases must be objects")

    selection = [phase for phase in resolved if phase.get("skip_test") is True]
    candidate = [
        phase for phase in resolved
        if phase.get("skip_test") is False and phase.get("models") == [CANDIDATE_MODEL]
    ]
    baselines = [
        phase for phase in resolved
        if phase.get("skip_test") is False and phase.get("models") != [CANDIDATE_MODEL]
    ]
    if len(selection) != 1 or len(candidate) != 1 or len(baselines) != 1:
        raise ValueError("unable to identify selection, candidate, and baseline phases")
    selection, candidate, baselines = selection[0], candidate[0], baselines[0]
    if selection.get("models") != [CANDIDATE_MODEL]:
        raise ValueError("selection phase must contain only the candidate")
    if not isinstance(baselines.get("models"), list) or not baselines["models"]:
        raise ValueError("baseline phase must contain one or more models")
    if CANDIDATE_MODEL in baselines["models"]:
        raise ValueError("baseline phase cannot include the candidate")
    if tuple(candidate.get("datasets", ())) != REQUIRED_DATASETS:
        raise ValueError("candidate phase datasets must be electricity and traffic")
    if tuple(selection.get("datasets", ())) != REQUIRED_DATASETS or tuple(baselines.get("datasets", ())) != REQUIRED_DATASETS:
        raise ValueError("all release phases must cover electricity and traffic")
    for phase in (selection, candidate, baselines):
        if phase.get("horizons") != [96]:
            raise ValueError("all release phases must use H96")
    selection_seeds = tuple(int(seed) for seed in selection.get("seeds", ()))
    confirmation_seeds = tuple(int(seed) for seed in candidate.get("seeds", ()))
    if len(selection_seeds) != 3 or len(set(selection_seeds)) != 3:
        raise ValueError("selection phase requires three unique seeds")
    if len(confirmation_seeds) != 5 or len(set(confirmation_seeds)) != 5:
        raise ValueError("candidate phase requires five unique confirmation seeds")
    if tuple(int(seed) for seed in baselines.get("seeds", ())) != confirmation_seeds:
        raise ValueError("baseline confirmation seeds do not match candidate seeds")
    if candidate.get("stream_metrics") is not True or baselines.get("stream_metrics") is not True:
        raise ValueError("test phases require stream_metrics=true")
    candidate_fields = (
        "mamba_ts_variant", "mamba_ts_mixer_type", "mamba_ts_repr_forecast_fusion",
        "patch_len", "patch_stride",
    )
    candidate_contract = {key: candidate.get(key) for key in candidate_fields}
    # Plans retain the lower-case runner field while config.json serializes
    # the runtime spelling. Normalize once before checking trained artifacts.
    candidate_contract["AutoCon_lambda"] = candidate.get(
        "AutoCon_lambda", candidate.get("autocon_lambda")
    )
    if candidate_contract["mamba_ts_variant"] != "v3":
        raise ValueError("candidate must use MambaTS v3")
    if candidate_contract["mamba_ts_mixer_type"] not in {None, "mamba1"}:
        raise ValueError("candidate must use the registered Mamba-1 mixer")
    if candidate_contract["mamba_ts_repr_forecast_fusion"] is not False:
        raise ValueError("candidate requires prediction fusion disabled")
    if abs(float(candidate_contract["AutoCon_lambda"]) - 0.001) > 1e-12:
        raise ValueError("candidate requires AutoCon lambda=.001")
    phase_by_name = {str(phase["name"]): phase for phase in resolved}
    counts = plan.get("registered_job_counts", {})
    if not isinstance(counts, dict):
        raise ValueError("registered_job_counts must be an object")
    return {
        "selection": selection,
        "candidate": candidate,
        "baselines": baselines,
        "selection_seeds": selection_seeds,
        "confirmation_seeds": confirmation_seeds,
        "baseline_models": tuple(baselines["models"]),
        "candidate_contract": candidate_contract,
        "expected_jobs": {
            name: int(counts[name]) for name in phase_by_name
        },
    }


def phase_rows(root: Path, phase: dict, expected: int, *, models: set[str], test_producing: bool, plan_hash: str, valid_seeds: tuple[int, ...], candidate_contract: dict) -> tuple[str, list[dict]]:
    validation = load(root / "phase_validation.json")
    phase_name = str(phase["name"])
    if validation.get("valid") is not True or validation.get("phase") != phase_name or validation.get("expected_jobs") != expected:
        raise ValueError(f"invalid phase validation: {root}")
    provenance = load(root / "provenance.json")
    source = str(provenance.get("source_tree_sha256", ""))
    launch = provenance.get("launch_authorization", {})
    if not source or launch.get("authorized") is not True or launch.get("enforced") is not True:
        raise ValueError(f"missing enforced launch binding: {root}")
    if launch.get("plan_sha256") != plan_hash:
        raise ValueError(f"plan hash mismatch: {root}")
    rows, seen = [], set()
    manifest = root / "logs" / "job_manifest.jsonl"
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        record = json.loads(raw)
        job_id = str(record.get("job_id", ""))
        if not job_id or job_id in seen or record.get("status") != "ok":
            raise ValueError(f"invalid manifest row: {job_id}")
        seen.add(job_id)
        if record.get("source_tree_sha256_before") != source or record.get("source_tree_sha256_after") != source:
            raise ValueError(f"source drift: {job_id}")
        result = Path(record["expected_artifact"]).parent
        config = load(result / "config.json")
        dataset = str(config.get("data"))
        model = str(config.get("model"))
        seed = int(config.get("seed", -1))
        if dataset not in REQUIRED_DATASETS or model not in models or int(config.get("pred_len", -1)) != 96:
            raise ValueError(f"coordinate mismatch: {job_id}")
        if test_producing:
            if seed not in valid_seeds or config.get("skip_test") is True or config.get("stream_metrics") is not True:
                raise ValueError(f"test contract mismatch: {job_id}")
            stream = load(result / "stream_metrics.json")
            if stream.get("aggregation") != "streaming_window_weighted":
                raise ValueError(f"stream aggregation mismatch: {job_id}")
            scaled, original = stream["scaled"], stream["original"]
            row = {"phase": phase_name, "dataset": dataset, "model": model, "horizon": 96, "seed": seed,
                   "elapsed_sec": float(record["elapsed_sec"]), "scaled_mse": float(scaled["mse"]),
                   "scaled_mae": float(scaled["mae"]), "original_mse": float(original["mse"]),
                   "original_mae": float(original["mae"])}
        else:
            if seed not in valid_seeds or config.get("skip_test") is not True or not (result / "training_complete.json").is_file():
                raise ValueError(f"selection contract mismatch: {job_id}")
            if any((result / name).exists() for name in TEST_ARTIFACTS) or (result / "stream_metrics.json").exists():
                raise ValueError(f"selection test artifact present: {job_id}")
            row = {"phase": phase_name, "dataset": dataset, "model": model, "horizon": 96, "seed": seed,
                   "elapsed_sec": float(record["elapsed_sec"])}
        if model == CANDIDATE_MODEL:
            if config.get("AutoCon") is not True or any(
                config.get(key) != value
                for key, value in candidate_contract.items()
                if key != "mamba_ts_mixer_type" or value is not None
            ):
                raise ValueError(f"candidate isolation mismatch: {job_id}")
            if candidate_contract["mamba_ts_mixer_type"] is None and config.get("mamba_ts_mixer_type", "mamba1") != "mamba1":
                raise ValueError(f"candidate mixer mismatch: {job_id}")
        else:
            if config.get("AutoCon") is not False or config.get("mamba_ts_repr_forecast_fusion") is not False:
                raise ValueError(f"baseline isolation mismatch: {job_id}")
        rows.append(row)
    if len(rows) != expected:
        raise ValueError(f"coverage mismatch for {phase_name}: {len(rows)} != {expected}")
    return source, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan = load(args.plan)
    if plan.get("status") != "launch_authorized":
        raise ValueError("plan is not launch authorized")
    plan_hash = sha256(args.plan)
    root = args.artifact_root.resolve()
    layout = _release_layout(plan)
    sources, rows = set(), []
    for role, phase, models, test_producing, seeds in (
        ("selection", layout["selection"], {CANDIDATE_MODEL}, False, layout["selection_seeds"]),
        ("candidate", layout["candidate"], {CANDIDATE_MODEL}, True, layout["confirmation_seeds"]),
        ("baselines", layout["baselines"], set(layout["baseline_models"]), True, layout["confirmation_seeds"]),
    ):
        relative = Path(str(phase.get("output_root", ""))).name
        if relative != role:
            raise ValueError("{} phase output root must end in {}".format(role, role))
        source, current = phase_rows(
            root / relative,
            phase,
            layout["expected_jobs"][str(phase["name"])],
            models=models,
            test_producing=test_producing,
            plan_hash=plan_hash,
            valid_seeds=seeds,
            candidate_contract=layout["candidate_contract"],
        )
        sources.add(source)
        rows.extend(current)
    if len(sources) != 1:
        raise ValueError("phase source snapshots differ")
    selection_name = str(layout["selection"]["name"])
    index = {(row["dataset"], row["model"], row["horizon"], row["seed"]): row for row in rows if row["phase"] != selection_name}
    expected = {
        (dataset, model, 96, seed)
        for dataset in REQUIRED_DATASETS
        for model in (CANDIDATE_MODEL,) + layout["baseline_models"]
        for seed in layout["confirmation_seeds"]
    }
    if set(index) != expected:
        raise ValueError("incomplete or duplicate confirmation coordinates")
    summary = []
    for dataset in REQUIRED_DATASETS:
        candidate = [index[(dataset, CANDIDATE_MODEL, 96, seed)] for seed in layout["confirmation_seeds"]]
        for baseline in layout["baseline_models"]:
            comparator = [index[(dataset, baseline, 96, seed)] for seed in layout["confirmation_seeds"]]
            ratios = [left["scaled_mse"] / right["scaled_mse"] for left, right in zip(candidate, comparator)]
            logs = [math.log(value) for value in ratios]
            upper = math.exp(statistics.fmean(logs) + stats.t.ppf(0.95, len(logs) - 1) * statistics.stdev(logs) / math.sqrt(len(logs)))
            summary.append({"dataset": dataset, "horizon": 96, "baseline": baseline,
                            "candidate_mse_mean": statistics.fmean(row["scaled_mse"] for row in candidate),
                            "baseline_mse_mean": statistics.fmean(row["scaled_mse"] for row in comparator),
                            "mean_mse_change_pct": 100 * (statistics.fmean(ratios) - 1),
                            "candidate_wins": sum(value < 1 for value in ratios), "n": len(ratios),
                            "upper_ratio_95_one_sided": upper, "noninferior_ratio_1p02": upper <= 1.02,
                            "candidate_elapsed_mean_sec": statistics.fmean(row["elapsed_sec"] for row in candidate),
                            "baseline_elapsed_mean_sec": statistics.fmean(row["elapsed_sec"] for row in comparator)})
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "selection_rows.csv", [row for row in rows if row["phase"] == selection_name])
    write_csv(out / "test_rows.csv", [row for row in rows if row["phase"] != selection_name])
    write_csv(out / "paired_test_summary.csv", summary)
    lines = ["# Held-out Electricity / Traffic H96 Relation-Auxiliary-Only Release", "", "Five fresh paired seeds; one-sided 95% paired log-MSE upper ratios use the registered 1.02 non-inferiority margin.", "", "| Dataset | Baseline | Candidate MSE | Baseline MSE | Change | Wins | Upper ratio | Non-inferior |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    lines.extend("| {dataset} | {baseline} | {candidate_mse_mean:.6f} | {baseline_mse_mean:.6f} | {mean_mse_change_pct:+.3f}% | {candidate_wins}/{n} | {upper_ratio_95_one_sided:.4f} | {noninferior_ratio_1p02} |".format(**row) for row in summary)
    lines.extend(["", "This release is restricted to Electricity/Traffic H96. The candidate uses AutoCon relation training without prediction-connected fusion; the result cannot establish universal accuracy, causality, or speed."])
    (out / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    metadata = {"plan_sha256": plan_hash, "source_tree_sha256": next(iter(sources)), "selection_rows": len(layout["selection_seeds"]) * len(REQUIRED_DATASETS), "test_rows": len(index), "paired_cells": len(summary), "test_metrics_produced": True}
    (out / "readout_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: metadata[key] for key in ("selection_rows", "test_rows", "paired_cells", "test_metrics_produced")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
