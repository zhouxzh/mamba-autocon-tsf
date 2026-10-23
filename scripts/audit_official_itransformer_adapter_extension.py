#!/usr/bin/env python3
"""Audit a pinned official iTransformer adapter extension by horizon."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


COORDINATES = {
    ("pems_bay", 192): (Path("artifacts/exploratory/official_itransformer_adapter_validation_20261003"), Path("artifacts/paper_final/mambatsv3_acf_infonce_official_itransformer_h192_extension_20261003"), Path("artifacts/paper_final/mambatsv3_acf_infonce_horizon_confirmatory_20261003"), (5410, 5411, 5412)),
    ("pems_bay", 336): (Path("artifacts/exploratory/official_itransformer_adapter_h336_validation_20261003"), Path("artifacts/paper_final/mambatsv3_acf_infonce_official_itransformer_h336_extension_20261003"), Path("artifacts/paper_final/mambatsv3_acf_infonce_horizon_confirmatory_20261003"), (5413, 5414, 5415)),
    ("metr_la", 192): (Path("artifacts/exploratory/official_itransformer_adapter_metr_la_h192_validation_20261003"), Path("artifacts/paper_final/mambatsv3_acf_infonce_official_itransformer_metr_la_h192_extension_20261003"), Path("artifacts/paper_final/mambatsv3_acf_infonce_metr_la_h192_confirmatory_20261003"), (5440, 5441, 5442)),
}
COMMIT = "c2426e68ca13f74aaec08045c5c724d8ad328124"


def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def guard(path):
    event = load(path)
    if event.get("status") != "completed" or event.get("trigger") is not None:
        raise ValueError("GPU guard failure: {}".format(path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["pems_bay", "metr_la"], required=True)
    parser.add_argument("--horizon", type=int, choices=[192, 336], required=True)
    args = parser.parse_args(); key = (args.target, args.horizon)
    if key not in COORDINATES: raise ValueError("unsupported official iTransformer coordinate: {}".format(key))
    validation_root, test_root, active_root, seeds = COORDINATES[key]
    rows = []
    for seed in seeds:
        stem = "{}_h{}_s{}".format(args.target, args.horizon, seed)
        validation = load(validation_root / (stem + "_official_itransformer.json"))
        test = load(test_root / (stem + "_official_itransformer.json"))
        active = load(active_root / (stem + "_active_test.json"))
        if validation.get("test_metrics_produced"):
            raise ValueError("validation adapter opened test: {}".format(seed))
        if validation.get("official_commit") != COMMIT or test.get("official_commit") != COMMIT:
            raise ValueError("official commit mismatch: {}".format(seed))
        if test.get("test_metrics_produced") is not True or test.get("selection_done_before_test") is not True:
            raise ValueError("test lifecycle mismatch: {}".format(seed))
        if active.get("test_metrics_produced") is not True:
            raise ValueError("missing fixed active test: {}".format(seed))
        guard(validation_root / (stem + "_official_itransformer_event.json"))
        guard(test_root / (stem + "_official_itransformer_event.json"))
        rows.append({"seed": seed, "active_test_mse": float(active["test_mse"]), "official_itransformer_test_mse": float(test["test_mse"]), "official_validation_mse": float(validation["best_validation_mse"]), "dataset_sha256": test["dataset_sha256"]})
    active = np.asarray([row["active_test_mse"] for row in rows]); official = np.asarray([row["official_itransformer_test_mse"] for row in rows]); ratio = active / official
    result = {"external_repository": "thuml/iTransformer", "official_commit": COMMIT, "target": args.target, "horizon": args.horizon, "adapter_contract": "official model with local chronological adapter", "rows": rows, "summary": {"active_mean_mse": float(active.mean()), "official_itransformer_mean_mse": float(official.mean()), "active_vs_official_itransformer_percent": float((active.mean() / official.mean() - 1) * 100), "active_wins": int((active < official).sum()), "active_official_ratio_upper_1se": float(ratio.mean() + ratio.std(ddof=1) / math.sqrt(len(ratio)))}, "audit_valid": True, "scope": "post-release external-model-code extension; not part of original frozen primary significance matrix"}
    (test_root / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# Official iTransformer Adapter {} H{} Extension Audit".format(args.target, args.horizon), "", "Pinned official network code is evaluated through the local chronological boundary adapter. This is a post-release external-code extension, not a member of the original frozen primary matrix.", "", "| Active | Official iTransformer | Active vs official | Wins | One-SE upper ratio |", "| ---: | ---: | ---: | ---: | ---: |", "| {:.6f} | {:.6f} | {:+.3f}% | {}/3 | {:.4f} |".format(result["summary"]["active_mean_mse"], result["summary"]["official_itransformer_mean_mse"], result["summary"]["active_vs_official_itransformer_percent"], result["summary"]["active_wins"], result["summary"]["active_official_ratio_upper_1se"]), "", "Official commit: `{}`".format(COMMIT)]
    (test_root / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
