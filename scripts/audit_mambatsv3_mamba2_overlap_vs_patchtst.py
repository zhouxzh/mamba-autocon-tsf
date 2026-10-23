#!/usr/bin/env python3
"""Audit the validation-only Mamba-2 frozen-core overlap-family probe."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "artifacts/exploratory/mambatsv3_pems_mamba2_overlap_probe_20261006"
BASELINE = ROOT / "artifacts/exploratory/official_patchtst_mamba2_patch24_comparator_20261006"
SEEDS = (5870, 5871, 5872)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _one_se_upper(values: np.ndarray) -> float:
    return float(values.mean() + values.std(ddof=1) / math.sqrt(len(values)))


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active", type=Path, default=ACTIVE)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--mamba-headdim", type=int, default=24)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--baseline-stem", default="pems_bay_h192",
                        help="checkpoint filename prefix before _s{seed}_official_patchtst.json")
    args = parser.parse_args()
    active_root = args.active.resolve()
    baseline_root = args.baseline.resolve()
    seeds = tuple(args.seed or SEEDS)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seed values must be unique")
    rows = []
    for seed in seeds:
        active = _load(active_root / f"s{seed}_active_frozen_core.json")
        null = _load(active_root / f"s{seed}_null_frozen_core.json")
        baseline = _load(baseline_root / f"{args.baseline_stem}_s{seed}_official_patchtst.json")
        if any(item.get("test_metrics_produced") for item in (active, null, baseline)):
            raise ValueError("comparison opened test metrics")
        config = active["config"]
        if (config.get("mamba_ts_mixer_type"), int(config.get("mamba_headdim")),
                int(config.get("patch_len")), int(config.get("patch_stride")),
                bool(config.get("freeze_pretrained_core"))) != ("mamba2", args.mamba_headdim, 24, 8, True):
            raise ValueError(f"active checkpoint {seed} violates the registered Mamba-2 transfer contract")
        transfer = active.get("transfer") or {}
        loaded = transfer.get("loaded_keys") or []
        if not any(key.startswith("encoder.") for key in loaded):
            raise ValueError(f"active checkpoint {seed} has no transferred Mamba-2 encoder tensors")
        baseline_config = baseline["config"]
        if (int(baseline_config["patch_len"]), int(baseline_config["patch_stride"])) != (24, 8):
            raise ValueError(f"baseline checkpoint {seed} violates the 24/8 geometry contract")
        value = float(active["best_validation_mse"])
        null_value = float(null["best_validation_mse"])
        reference = float(baseline["best_validation_mse"])
        rows.append({
            "seed": seed,
            "mamba2_active": value,
            "mamba2_null": null_value,
            "official_patchtst": reference,
            "active_vs_null_ratio": value / null_value,
            "active_vs_patchtst_ratio": value / reference,
            "active_beats_null": bool(value < null_value),
            "active_beats_patchtst": bool(value < reference),
        })
    active = np.asarray([row["mamba2_active"] for row in rows])
    null = np.asarray([row["mamba2_null"] for row in rows])
    baseline = np.asarray([row["official_patchtst"] for row in rows])
    patch_ratios = active / baseline
    null_ratios = active / null
    result = {
        "validation_only": True,
        "test_metrics_produced": False,
        "rows": rows,
        "mean_mamba2_active": float(active.mean()),
        "mean_mamba2_null": float(null.mean()),
        "mean_official_patchtst": float(baseline.mean()),
        "active_vs_null_percent": float((active.mean() / null.mean() - 1.0) * 100.0),
        "active_vs_patchtst_percent": float((active.mean() / baseline.mean() - 1.0) * 100.0),
        "active_vs_null_wins": int((active < null).sum()),
        "active_vs_patchtst_wins": int((active < baseline).sum()),
        "active_vs_patchtst_ratio_mean": float(patch_ratios.mean()),
        "active_vs_patchtst_ratio_upper_1se": _one_se_upper(patch_ratios),
        "active_vs_patchtst_noninferior_at_one_se": bool(_one_se_upper(patch_ratios) <= 1.0),
        "mamba2_contract": {
            "mixer": "mamba2", "headdim": args.mamba_headdim, "patch_len": 24, "patch_stride": 8,
            "target_core_frozen": True,
        },
        "scope": "same-seed, validation-only, official-code comparison; no test evaluation",
        "selection_status": "exploratory: two seeds were added after observing seed 5870",
        "submission_ready": False,
        "submission_blocker": "adaptive validation search, one target dataset, and no independently preregistered replication",
    }
    output = args.output or active_root / "official_patchtst_mamba2_validation_audit.json"
    output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
