#!/usr/bin/env python3
"""Strict readout for the frozen Solar H720 forecast-pair confirmation.

This reader is intentionally validation-only: it verifies the launch manifest,
artifact hashes, exact seed pairing and required telemetry, then reads the
minimum validation loss from each completed training run.  It never invokes a
model and rejects any test artifact under a phase root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
PLAN_DEFAULT = ROOT / "docs/research/nlinear_variable_mamba_solar_h720_forecast_pair_confirmation_20260928.json"
PHASES = (
    ("solar_h720_forecast_pair_off", False, 0.0),
    ("solar_h720_forecast_pair_on", True, 0.01),
)
SEEDS = (2041, 2042, 2043)
TELEMETRY = (
    "forecast_lag_pair_loss",
    "forecast_lag_pair_count",
    "forecast_lag_pair_active_anchor_fraction",
)


def _read(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} is non-finite")
    return number


def _phase(plan: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    for item in plan.get("phases", []):
        if isinstance(item, Mapping) and item.get("name") == name:
            return item
    raise ValueError(f"plan has no phase {name!r}")


def _artifact_path(record: Mapping[str, Any], phase_root: Path) -> Path:
    raw = record.get("expected_artifact")
    if not raw:
        raise ValueError(f"missing expected artifact: {record.get('job_id')}")
    path = Path(str(raw)).resolve()
    results_root = (phase_root / "results").resolve()
    try:
        path.relative_to(results_root)
    except ValueError as exc:
        raise ValueError(f"artifact outside results root: {record.get('job_id')}") from exc
    if path.name != "training_complete.json" or not path.is_file():
        raise ValueError(f"invalid validation artifact: {path}")
    return path


def _validated_records(phase_root: Path, expected_name: str) -> list[Mapping[str, Any]]:
    validation = _read(phase_root / "phase_validation.json")
    if validation.get("valid") is not True or validation.get("phase") != expected_name:
        raise ValueError(f"invalid phase validation: {phase_root}")
    manifest = phase_root / "logs/job_manifest.jsonl"
    if validation.get("manifest_sha256") != _sha256(manifest):
        raise ValueError(f"manifest changed after validation: {manifest}")
    records: list[Mapping[str, Any]] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if not isinstance(item, Mapping):
                raise ValueError(f"manifest row is not an object: {manifest}")
            records.append(item)
    if len(records) != 3 or validation.get("status_counts") != {"ok": 3}:
        raise ValueError(f"incomplete phase manifest: {phase_root}")
    if any(row.get("status") != "ok" or row.get("phase") != expected_name for row in records):
        raise ValueError(f"unsuccessful or mislabelled manifest row: {phase_root}")
    return records


def _load_arm(plan: Mapping[str, Any], name: str, enabled: bool, weight: float) -> dict[int, dict[str, Any]]:
    phase = _phase(plan, name)
    root = (ROOT / str(phase["output_root"])).resolve()
    if phase.get("models") != ["NLinearVariableMambaAutoCon"] or phase.get("horizons") != [720]:
        raise ValueError(f"phase contract mismatch: {name}")
    if phase.get("seeds") != list(SEEDS) or phase.get("skip_test") is not True:
        raise ValueError(f"seed/test contract mismatch: {name}")
    if bool(phase.get("mamba_acf_forecast_pair_consistency")) != enabled:
        raise ValueError(f"treatment flag mismatch: {name}")
    if not math.isclose(_finite(phase.get("mamba_acf_forecast_pair_weight"), name), weight):
        raise ValueError(f"treatment weight mismatch: {name}")
    if any((root / "test_results").rglob("*") if (root / "test_results").exists() else []):
        raise ValueError(f"test artifacts present in validation-only phase: {root}")

    rows: dict[int, dict[str, Any]] = {}
    for record in _validated_records(root, name):
        seed = int(record.get("seed"))
        if seed in rows or seed not in SEEDS or record.get("dataset") != "solar":
            raise ValueError(f"bad coordinate in {name}: {record.get('job_id')}")
        artifact = _artifact_path(record, root)
        result_dir = artifact.parent
        checkpoint = root / "checkpoints" / result_dir.name
        hashes = record.get("artifact_hashes")
        if not isinstance(hashes, Mapping):
            raise ValueError(f"missing artifact hashes: {record.get('job_id')}")
        for filename in ("config.json", "training_complete.json"):
            if hashes.get(filename) != _sha256(result_dir / filename):
                raise ValueError(f"result hash mismatch {filename}: {record.get('job_id')}")
        if hashes.get("loss_logs.json") != _sha256(checkpoint / "loss_logs.json"):
            raise ValueError(f"loss hash mismatch: {record.get('job_id')}")
        config = _read(result_dir / "config.json")
        expected = {
            "model": "NLinearVariableMambaAutoCon", "AutoCon": True,
            "pred_len": 720, "seed": seed, "skip_test": True,
            "mamba_variable_repr_fusion": True, "mamba_acf_relation_loss": True,
            "mamba_acf_forecast_pair_consistency": enabled,
        }
        for key, value in expected.items():
            if config.get(key) != value:
                raise ValueError(f"config {key} mismatch: {record.get('job_id')}")
        if not math.isclose(_finite(config.get("mamba_acf_forecast_pair_weight"), key), weight):
            raise ValueError(f"config pair weight mismatch: {record.get('job_id')}")
        logs = _read(checkpoint / "loss_logs.json")
        val_loss = [_finite(value, f"val_loss:{record.get('job_id')}") for value in logs.get("val_loss", [])]
        if not val_loss:
            raise ValueError(f"missing validation losses: {record.get('job_id')}")
        telemetry = {}
        if enabled:
            for metric in TELEMETRY:
                values = [_finite(value, f"{metric}:{record.get('job_id')}") for value in logs.get(metric, [])]
                if not values or min(values) <= 0:
                    raise ValueError(f"non-positive treatment telemetry {metric}: {record.get('job_id')}")
                telemetry[metric] = statistics.fmean(values)
        rows[seed] = {"mse": min(val_loss), "elapsed_sec": _finite(record.get("elapsed_sec"), "elapsed"), "telemetry": telemetry}
    if set(rows) != set(SEEDS):
        raise ValueError(f"incomplete exact seed coverage: {name}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN_DEFAULT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    plan = _read(plan_path)
    if plan.get("status") != "exploratory_nlinear_variable_mamba_solar_h720_forecast_pair_confirmation":
        raise ValueError("unexpected plan status")
    if plan.get("execution", {}).get("authorized") is not False:
        raise ValueError("exploratory plan must retain authorized=false")
    off = _load_arm(plan, *PHASES[0])
    on = _load_arm(plan, *PHASES[1])
    deltas = [on[seed]["mse"] - off[seed]["mse"] for seed in SEEDS]
    wins = sum(delta < 0 for delta in deltas)
    passed = statistics.fmean(deltas) < 0 and wins >= 2
    output = args.output or (ROOT / "artifacts/exploratory/nlinear_variable_mamba_solar_h720_forecast_pair_confirmation_20260928/readout/readout.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Solar H720 forecast-lag-pair confirmation readout", "",
        "Validation-only result; no test metrics were generated or read.", "",
        "| Seed | Term off MSE | Term on MSE | On change | On better |",
        "| ---: | ---: | ---: | ---: | :---: |",
    ]
    for seed, delta in zip(SEEDS, deltas):
        lines.append(f"| {seed} | {off[seed]['mse']:.6f} | {on[seed]['mse']:.6f} | {delta / off[seed]['mse'] * 100:+.4f}% | {'yes' if delta < 0 else 'no'} |")
    lines.extend([
        "", f"Mean MSE: off `{statistics.fmean(row['mse'] for row in off.values()):.6f}`, on `{statistics.fmean(row['mse'] for row in on.values()):.6f}`; paired change `{statistics.fmean(deltas) / statistics.fmean(row['mse'] for row in off.values()) * 100:+.4f}%`, wins `{wins}/3`.",
        f"Registered validation gate: **{'PASS' if passed else 'FAIL'}**.",
        "", "The enabled arm had positive forecast-pair loss, pair count and active-anchor fraction in every seed. This is a two-horizon Solar validation mechanism check, not a held-out result or a universal AutoCon claim.", "",
        f"Plan SHA-256: `{_sha256(plan_path)}`.",
    ])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
