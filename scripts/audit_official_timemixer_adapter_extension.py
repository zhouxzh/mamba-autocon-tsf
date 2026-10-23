#!/usr/bin/env python3
"""Audit a pinned official TimeMixer adapter extension by horizon."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


COMMIT = "e24610583b36fdd8c76cc17a8df4e65759a5f460"
ACTIVE_ROOT = Path("artifacts/paper_final/mambatsv3_acf_infonce_horizon_confirmatory_20261003")
ROOTS = {
    192: (
        Path("artifacts/exploratory/official_timemixer_adapter_validation_20261003"),
        Path("artifacts/paper_final/mambatsv3_acf_infonce_official_timemixer_h192_extension_20261003"),
        (5410, 5411, 5412),
    ),
    336: (
        Path("artifacts/exploratory/official_timemixer_adapter_h336_validation_20261003"),
        Path("artifacts/paper_final/mambatsv3_acf_infonce_official_timemixer_h336_extension_20261003"),
        (5413, 5414, 5415),
    ),
}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_event(path):
    event = load(path)
    if event.get("status") != "completed" or event.get("trigger") is not None:
        raise ValueError("GPU guard failure: {}".format(path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, choices=sorted(ROOTS), required=True)
    args = parser.parse_args()
    validation_root, test_root, seeds = ROOTS[args.horizon]
    rows = []
    for seed in seeds:
        stem = "pems_bay_h{}_s{}".format(args.horizon, seed)
        validation = load(validation_root / (stem + "_official_timemixer.json"))
        test = load(test_root / (stem + "_official_timemixer.json"))
        active = load(ACTIVE_ROOT / (stem + "_active_test.json"))
        if validation.get("test_metrics_produced"):
            raise ValueError("validation opened test: {}".format(seed))
        if validation.get("official_commit") != COMMIT or test.get("official_commit") != COMMIT:
            raise ValueError("official commit mismatch: {}".format(seed))
        if test.get("test_metrics_produced") is not True or test.get("selection_done_before_test") is not True:
            raise ValueError("test lifecycle mismatch: {}".format(seed))
        if active.get("test_metrics_produced") is not True:
            raise ValueError("candidate test missing: {}".format(seed))
        check_event(validation_root / (stem + "_official_timemixer_event.json"))
        check_event(test_root / (stem + "_official_timemixer_event.json"))
        rows.append({
            "seed": seed,
            "active_test_mse": float(active["test_mse"]),
            "official_timemixer_test_mse": float(test["test_mse"]),
            "official_validation_mse": float(validation["best_validation_mse"]),
            "dataset_sha256": test["dataset_sha256"],
        })
    active = np.asarray([row["active_test_mse"] for row in rows])
    official = np.asarray([row["official_timemixer_test_mse"] for row in rows])
    ratios = active / official
    result = {
        "external_repository": "kwuking/TimeMixer",
        "official_commit": COMMIT,
        "horizon": args.horizon,
        "adapter_contract": "official model with local chronological adapter",
        "rows": rows,
        "summary": {
            "active_mean_mse": float(active.mean()),
            "official_timemixer_mean_mse": float(official.mean()),
            "active_vs_official_timemixer_percent": float((active.mean() / official.mean() - 1.0) * 100.0),
            "active_wins": int((active < official).sum()),
            "active_official_ratio_upper_1se": float(ratios.mean() + ratios.std(ddof=1) / math.sqrt(len(ratios))),
        },
        "audit_valid": True,
        "scope": "post-release external-model-code extension; not part of original frozen primary significance matrix",
    }
    (test_root / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = result["summary"]
    lines = [
        "# Official TimeMixer Adapter H{} Extension Audit".format(args.horizon),
        "",
        "Pinned official network code is evaluated through the local chronological boundary adapter. This is a post-release external-code extension, not a member of the original frozen primary matrix.",
        "",
        "| Active | Official TimeMixer | Active vs official | Wins | One-SE upper ratio |",
        "| ---: | ---: | ---: | ---: | ---: |",
        "| {:.6f} | {:.6f} | {:+.3f}% | {}/3 | {:.4f} |".format(
            summary["active_mean_mse"], summary["official_timemixer_mean_mse"],
            summary["active_vs_official_timemixer_percent"], summary["active_wins"],
            summary["active_official_ratio_upper_1se"],
        ),
        "",
        "Official commit: `{}`".format(COMMIT),
    ]
    (test_root / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
