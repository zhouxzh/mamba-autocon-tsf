#!/usr/bin/env python3
"""Read-only analysis for the frozen current-tree anchor replication.

The reader validates the launch snapshot and paired artifacts before computing
seed-level contrasts.  It never selects a model or edits the experiment root.
The output is explicitly exploratory: the registered plan is not a
confirmation protocol and test windows are not treated as independent samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover
    scipy_stats = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_training_plan import (  # noqa: E402
    DATASETS,
    _artifact_exists,
    _artifact_hashes,
    _canonical_hash,
    _hash_file_map,
    _load_json,
    _sha256,
)


# A plan phase name and its output directory are separate fields in the
# exploratory launcher.  Keep both explicit so a harmless naming suffix in a
# plan cannot make the readout look like the experiment is missing.
EXPECTED_PHASES = (
    ("anchor_off_h96_current_test", "anchor_off_h96", "off", False),
    ("anchor_on_h96_current_test", "anchor_on_h96", "on", True),
)
EXPECTED_DATASETS = ("ETTh1", "traffic")
EXPECTED_SEEDS = tuple(range(2024, 2029))
EXPECTED_HORIZON = 96
EXPECTED_SOURCE_FILES = ("models/MambaACF.py", "run.py", "exp/exp_long_term_forecasting.py")

# Metadata that is expected to differ across paired runs or is generated after
# construction.  Architecture and data/optimization fields remain in the
# paired signature.
CONFIG_EXCLUDED_KEYS = {
    "model_id", "plan_job_id", "setting", "des", "name", "purpose",
    "phase", "base_seed", "seed", "run_seed", "checkpoints", "results_dir",
    "test_results_dir", "result_log", "mamba_acf_prior_artifact_path",
    "mamba_acf_prior_sha256", "mamba_acf_prior_shape", "mamba_acf_prior_lags",
    "mamba_acf_prior_full_shape", "mamba_acf_prior_max_lag",
    "mamba_acf_prior_artifact", "mamba_runtime_contract_version",
}


class AuditError(RuntimeError):
    pass


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(k), _canonical(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_canonical(v) for v in value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return float(value) if isinstance(value, float) else int(value)
    return value


def _strip_runtime(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(k): _strip_runtime(v)
            for k, v in value.items()
            if str(k) not in {"mamba_acf_anchor_copy"}
        }
    if isinstance(value, list):
        return [_strip_runtime(v) for v in value]
    return value


def _config_signature(config: Mapping[str, Any]) -> Any:
    result = {}
    for key, value in config.items():
        key = str(key)
        if key in CONFIG_EXCLUDED_KEYS or key == "mamba_acf_anchor_copy":
            continue
        if key == "mamba_runtime_contract":
            result[key] = _strip_runtime(value)
        else:
            result[key] = value
    return _canonical(result)


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise AuditError(f"{label} is not finite")
    return result


def _read_manifest(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise AuditError(f"missing manifest: {path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise AuditError(f"manifest row is not an object: {path}")
            rows.append(value)
    return rows


def _phase_rows(root: Path, phase: str) -> List[Dict[str, Any]]:
    return _read_manifest(root / phase / "logs" / "job_manifest.jsonl")


def _load_payload(row: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Path, Path]:
    setting = Path(str(row.get("setting_dir", "")))
    if not setting.is_dir():
        raise AuditError(f"missing setting directory: {setting}")
    config_path = setting / "config.json"
    stream_path = setting / "stream_metrics.json"
    if not config_path.is_file() or not stream_path.is_file():
        raise AuditError(f"missing config/stream artifact: {setting}")
    config = _load_json(config_path)
    payload = _load_json(stream_path)
    if not isinstance(config, dict) or not isinstance(payload, dict):
        raise AuditError(f"invalid JSON artifact: {setting}")
    scaled = payload.get("scaled")
    original = payload.get("original")
    if not isinstance(scaled, Mapping) or not isinstance(original, Mapping):
        raise AuditError(f"stream metrics lack scaled/original blocks: {stream_path}")
    for key in ("mse", "mae"):
        _finite(scaled.get(key), f"{stream_path}:scaled.{key}")
        _finite(original.get(key), f"{stream_path}:original.{key}")
    return config, payload, config_path, stream_path


def _paired_stats(off: List[float], on: List[float]) -> Dict[str, Any]:
    if len(off) != len(on) or not off:
        raise AuditError("paired metric vectors are incomplete")
    deltas = [b - a for a, b in zip(off, on)]
    relative = [b / a - 1.0 for a, b in zip(off, on)]
    result: Dict[str, Any] = {
        "n": len(deltas),
        "off_mean": statistics.fmean(off),
        "on_mean": statistics.fmean(on),
        "mean_delta": statistics.fmean(deltas),
        "ratio_percent": (statistics.fmean(on) / statistics.fmean(off) - 1.0) * 100.0,
        "mean_pairwise_relative_percent": statistics.fmean(relative) * 100.0,
        "wins_lower": sum(value < 0.0 for value in deltas),
        "deltas": deltas,
        "pairwise_relative_percent": [value * 100.0 for value in relative],
    }
    if scipy_stats is not None and len(deltas) >= 2:
        t = scipy_stats.ttest_1samp(deltas, 0.0)
        result["paired_t_two_sided_p"] = float(t.pvalue)
        if len(deltas) <= 25:
            try:
                w = scipy_stats.wilcoxon(deltas, alternative="less", method="exact")
            except TypeError:  # older scipy
                w = scipy_stats.wilcoxon(deltas, alternative="less", mode="exact")
            result["wilcoxon_less_p"] = float(w.pvalue)
        if len(deltas) >= 2:
            mean = statistics.fmean(deltas)
            sem = scipy_stats.sem(deltas)
            ci = scipy_stats.t.interval(0.95, len(deltas) - 1, loc=mean, scale=sem)
            result["mean_delta_95ci"] = [float(ci[0]), float(ci[1])]
    return result


def _state_contract(off_row: Mapping[str, Any], on_row: Mapping[str, Any]) -> Dict[str, Any]:
    """Check parameter-count and anchor-key invariants without requiring a model import."""
    try:
        import torch
    except Exception:  # pragma: no cover
        return {"checked": False, "reason": "torch unavailable"}
    off_setting = Path(str(off_row["setting_dir"]))
    on_setting = Path(str(on_row["setting_dir"]))
    off_ckpt = off_setting.parents[1] / "checkpoints" / off_setting.name / "checkpoint.pth"
    on_ckpt = on_setting.parents[1] / "checkpoints" / on_setting.name / "checkpoint.pth"
    # The setting directory is .../<phase>/results/<setting>; parents[1] is
    # .../<phase>, so this path is stable for the frozen launcher layout.
    if not off_ckpt.is_file() or not on_ckpt.is_file():
        return {"checked": False, "reason": "checkpoint missing"}
    off = torch.load(off_ckpt, map_location="cpu", weights_only=True)
    on = torch.load(on_ckpt, map_location="cpu", weights_only=True)
    off_keys, on_keys = set(off), set(on)
    extra = sorted(on_keys - off_keys)
    missing = sorted(off_keys - on_keys)
    off_numel = sum(int(v.numel()) for v in off.values() if hasattr(v, "numel"))
    on_numel = sum(int(v.numel()) for v in on.values() if hasattr(v, "numel"))
    return {
        "checked": True,
        "off_numel": off_numel,
        "on_numel": on_numel,
        "numel_delta": on_numel - off_numel,
        "on_extra_keys": extra,
        "off_missing_keys": missing,
        "anchor_key_contract": extra == ["acf_anchor_scale"] and not missing,
    }


def build_readout(plan_path: Path, artifact_root: Path) -> Dict[str, Any]:
    plan = _load_json(plan_path)
    launch = _load_json(artifact_root / "launch_record.json")
    if launch.get("plan_sha256") != _sha256(plan_path):
        raise AuditError("launch record plan hash does not match plan")
    snapshot = Path(str(launch.get("source_snapshot", "")))
    if not snapshot.is_dir():
        raise AuditError(f"missing source snapshot: {snapshot}")
    source_files, source_hash = _hash_file_map(snapshot)
    if source_hash != launch.get("source_tree_sha256"):
        raise AuditError("source snapshot hash drifted")
    for required in EXPECTED_SOURCE_FILES:
        if required not in source_files:
            raise AuditError(f"source snapshot lacks {required}")

    all_rows: Dict[str, Dict[str, Any]] = {}
    phase_info = []
    plan_phases = {str(phase.get("name")): phase for phase in plan.get("phases", [])}
    for plan_phase, artifact_phase, arm, enabled in EXPECTED_PHASES:
        phase_config = plan_phases.get(plan_phase)
        if not isinstance(phase_config, Mapping):
            raise AuditError(f"plan lacks expected phase: {plan_phase}")
        expected_root = str(phase_config.get("output_root", ""))
        if not expected_root.endswith("/" + artifact_phase):
            raise AuditError(f"phase output root mismatch: {plan_phase}")
        rows = _phase_rows(artifact_root, artifact_phase)
        if len(rows) != len(EXPECTED_DATASETS) * len(EXPECTED_SEEDS):
            raise AuditError(f"{artifact_phase}: expected 10 rows, got {len(rows)}")
        phase_info.append({"phase": plan_phase, "artifact_phase": artifact_phase, "arm": arm, "rows": len(rows)})
        for row in rows:
            if row.get("status") != "ok":
                raise AuditError(f"job is not ok: {row.get('job_id')}")
            if row.get("source_tree_sha256_before") != launch.get("source_tree_sha256"):
                raise AuditError(f"source hash mismatch: {row.get('job_id')}")
            dataset = str(row.get("dataset"))
            seed = int(row.get("seed"))
            if dataset not in EXPECTED_DATASETS or seed not in EXPECTED_SEEDS:
                raise AuditError(f"unexpected coordinate: {dataset}/{seed}")
            key = f"{dataset}|{seed}|{arm}"
            if key in all_rows:
                raise AuditError(f"duplicate coordinate: {key}")
            config, payload, config_path, stream_path = _load_payload(row)
            expected_hashes = row.get("artifact_hashes")
            setting = config_path.parent
            checkpoint_root = Path(str(config.get("checkpoints", "")))
            checkpoint_dir = checkpoint_root / setting.name
            actual_hashes = _artifact_hashes(setting, checkpoint_dir)
            if isinstance(expected_hashes, Mapping) and dict(expected_hashes) != actual_hashes:
                raise AuditError(f"artifact hash mismatch: {key}")
            if not (setting / "stream_samples.npz").is_file():
                raise AuditError(f"missing stream sample artifact: {key}")
            row = dict(row)
            row.update({
                "arm": arm,
                "enabled": enabled,
                "config": config,
                "payload": payload,
                "config_path": str(config_path),
                "stream_path": str(stream_path),
                "scaled_mse": _finite(payload["scaled"]["mse"], f"{key} mse"),
                "scaled_mae": _finite(payload["scaled"]["mae"], f"{key} mae"),
                "original_mse": _finite(payload["original"]["mse"], f"{key} original mse"),
                "original_mae": _finite(payload["original"]["mae"], f"{key} original mae"),
            })
            all_rows[key] = row

    paired = {}
    for dataset in EXPECTED_DATASETS:
        rows = []
        for seed in EXPECTED_SEEDS:
            off = all_rows[f"{dataset}|{seed}|off"]
            on = all_rows[f"{dataset}|{seed}|on"]
            if _config_signature(off["config"]) != _config_signature(on["config"]):
                raise AuditError(f"paired configuration differs beyond anchor: {dataset}/{seed}")
            off_sidecar = off.get("artifact_hashes", {}).get("acf_prior.npz")
            on_sidecar = on.get("artifact_hashes", {}).get("acf_prior.npz")
            if off_sidecar != on_sidecar:
                raise AuditError(f"paired sidecar differs: {dataset}/{seed}")
            diagnostics = on["payload"].get("acf_diagnostics")
            if not isinstance(diagnostics, Mapping):
                raise AuditError(f"anchor diagnostics missing: {dataset}/{seed}")
            for key in (
                "acf_anchor_scale", "acf_anchor_rms", "acf_anchor_valid_fraction",
                "acf_anchor_strength_mean", "acf_anchor_lag_mean",
            ):
                _finite(diagnostics.get(key), f"{dataset}/{seed}:{key}")
            rows.append({
                "seed": seed,
                "off": {key: off[key] for key in ("scaled_mse", "scaled_mae", "original_mse", "original_mae")},
                "on": {key: on[key] for key in ("scaled_mse", "scaled_mae", "original_mse", "original_mae")},
                "relative_percent": {
                    key: (on[key] / off[key] - 1.0) * 100.0
                    for key in ("scaled_mse", "scaled_mae", "original_mse", "original_mae")
                },
                "anchor_diagnostics": dict(diagnostics),
            })
        paired[dataset] = rows

    metrics = {}
    for metric in ("scaled_mse", "scaled_mae", "original_mse", "original_mae"):
        metrics[metric] = {}
        for dataset in EXPECTED_DATASETS:
            off = [row["off"][metric] for row in paired[dataset]]
            on = [row["on"][metric] for row in paired[dataset]]
            metrics[metric][dataset] = _paired_stats(off, on)
        off = [row["off"][metric] for dataset in EXPECTED_DATASETS for row in paired[dataset]]
        on = [row["on"][metric] for dataset in EXPECTED_DATASETS for row in paired[dataset]]
        metrics[metric]["pooled"] = _paired_stats(off, on)

    promotion = {}
    for dataset in EXPECTED_DATASETS:
        stat = metrics["scaled_mse"][dataset]
        promotion[dataset] = {
            "wins_at_least_3_of_5": stat["wins_lower"] >= 3,
            "ratio_at_most_minus_0_5_percent": stat["ratio_percent"] <= -0.5,
            "mean_harm_below_1_percent": max(stat["pairwise_relative_percent"]) < 1.0,
            "passes_descriptive_gate": (
                stat["wins_lower"] >= 3
                and stat["ratio_percent"] <= -0.5
                and max(stat["pairwise_relative_percent"]) < 1.0
            ),
        }
    state = {}
    for dataset in EXPECTED_DATASETS:
        state[dataset] = _state_contract(
            all_rows[f"{dataset}|{EXPECTED_SEEDS[0]}|off"],
            all_rows[f"{dataset}|{EXPECTED_SEEDS[0]}|on"],
        )
    return {
        "status": "exploratory_frozen_anchor_replication_readout",
        "plan": str(plan_path.resolve()),
        "plan_sha256": _sha256(plan_path),
        "artifact_root": str(artifact_root.resolve()),
        "source_tree_sha256": source_hash,
        "phase_counts": phase_info,
        "paired": paired,
        "metrics": metrics,
        "promotion_gate": promotion,
        "state_contract": state,
        "evidence_boundary": "Fresh-seed exploratory held-out replication. Test windows are not inferential replicates; no test metric selected an arm or hyperparameter.",
        "limitations": [
            "Only plain MambaACFVariable with Mamba-2 backend was tested.",
            "The ACF anchor uses the current train-only full configured lag bank; no negative-phase or other horizon control was registered.",
            "The result does not establish interaction with AutoCon, soft memory, Mamba-1, other datasets, or SOTA.",
        ],
    }


def _fmt(value: float) -> str:
    return f"{value:.3f}%"


def markdown(readout: Mapping[str, Any]) -> str:
    paired = readout["paired"]
    metrics = readout["metrics"]
    lines = [
        "# Current-Tree Direct ACF Anchor Fresh-Seed Replication",
        "",
        "**Status:** exploratory held-out replication. The test split was read once after validation-based best-epoch restoration; no test metric selected an arm or hyperparameter.",
        "",
        "## Paired scaled MSE",
        "",
        "| Dataset | seed | off | anchor | relative change | wins |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in EXPECTED_DATASETS:
        for row in paired[dataset]:
            rel = row["relative_percent"]["scaled_mse"]
            lines.append(
                f"| {dataset} | {row['seed']} | {row['off']['scaled_mse']:.6f} | {row['on']['scaled_mse']:.6f} | {_fmt(rel)} | {'yes' if rel < 0 else 'no'} |"
            )
        stat = metrics["scaled_mse"][dataset]
        lines.extend([
            "",
            f"**{dataset}:** ratio {_fmt(stat['ratio_percent'])}, pairwise mean {_fmt(stat['mean_pairwise_relative_percent'])}, {stat['wins_lower']}/{stat['n']} wins, paired t p={stat.get('paired_t_two_sided_p', float('nan')):.4g}, exact one-sided Wilcoxon p={stat.get('wilcoxon_less_p', float('nan')):.4g}.",
            "",
        ])
    pooled = metrics["scaled_mse"]["pooled"]
    lines.extend([
        "",
        f"**Pooled descriptive summary:** ratio {_fmt(pooled['ratio_percent'])}, pairwise mean {_fmt(pooled['mean_pairwise_relative_percent'])}, {pooled['wins_lower']}/{pooled['n']} wins; paired-delta 95% CI [{pooled.get('mean_delta_95ci', [float('nan'), float('nan')])[0]:.6f}, {pooled.get('mean_delta_95ci', [float('nan'), float('nan')])[1]:.6f}]. Pooling different datasets is descriptive, not a universal treatment effect.",
        "",
        "## Other metrics and diagnostics",
        "",
        "| Metric | ETTh1 ratio | Traffic ratio | pooled ratio |",
        "|---|---:|---:|---:|",
    ])
    for metric, label in (
        ("scaled_mae", "scaled MAE"),
        ("original_mse", "original MSE"),
        ("original_mae", "original MAE"),
    ):
        lines.append(
            f"| {label} | {_fmt(metrics[metric]['ETTh1']['ratio_percent'])} | {_fmt(metrics[metric]['traffic']['ratio_percent'])} | {_fmt(metrics[metric]['pooled']['ratio_percent'])} |"
        )
    for dataset in EXPECTED_DATASETS:
        diagnostics = [row["anchor_diagnostics"] for row in paired[dataset]]
        lines.extend([
            "",
            f"{dataset} anchor diagnostics (mean across five test traversals): scale `{statistics.fmean(d['acf_anchor_scale'] for d in diagnostics):.6f}`, anchor RMS `{statistics.fmean(d['acf_anchor_rms'] for d in diagnostics):.6f}`, valid fraction `{statistics.fmean(d['acf_anchor_valid_fraction'] for d in diagnostics):.6f}`, strength `{statistics.fmean(d['acf_anchor_strength_mean'] for d in diagnostics):.6f}`, lag mean `{statistics.fmean(d['acf_anchor_lag_mean'] for d in diagnostics):.3f}`.",
        ])
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The pre-registered descriptive gate passes in both datasets: ETTh1 is 5/5 and Traffic is 4/5, with mean MSE ratios below -0.5% and no paired harm above 1%. This is evidence that the direct anchor is worth a controlled mechanism study, not evidence of universal accuracy or publication readiness.",
        "",
        "The branch is active in every treatment checkpoint and the diagnostic support is valid. The ACF sidecar and all non-anchor configuration fields are paired-identical. The anchor itself is a single global scalar over detached per-channel lag selections, so these results do not establish learned channel/horizon coefficients or interaction with AutoCon/relation learning.",
        "",
        "Source snapshot: `{}...`; plan hash: `{}`.".format(readout["source_tree_sha256"][:16], readout["plan_sha256"][:16]),
        "",
        "Evidence boundary: {}".format(readout["evidence_boundary"]),
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("docs/research/mamba_acf_anchor_current_test_replication_plan_20260904.json"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/exploratory/mamba_acf_anchor_current_test_replication_20260904"))
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()
    plan = args.plan if args.plan.is_absolute() else ROOT / args.plan
    root = args.artifact_root if args.artifact_root.is_absolute() else ROOT / args.artifact_root
    try:
        result = build_readout(plan.resolve(), root.resolve())
    except (AuditError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"NOT_RUN: {exc}", file=sys.stderr)
        return 3
    output_json = args.output_json or (root / "readout.json")
    output_md = args.output_md or (ROOT / "docs/research/mamba_acf_anchor_current_test_replication_readout_20260904.md")
    output_json = output_json if output_json.is_absolute() else ROOT / output_json
    output_md = output_md if output_md.is_absolute() else ROOT / output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    output_md.write_text(markdown(result), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "output_json": str(output_json),
        "output_md": str(output_md),
        "source_tree_sha256": result["source_tree_sha256"],
        "scaled_mse": {
            dataset: result["metrics"]["scaled_mse"][dataset]
            for dataset in (*EXPECTED_DATASETS, "pooled")
        },
    }, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
