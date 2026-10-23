#!/usr/bin/env python3
"""Audit the PEMS-BAY 25% official iTransformer extension."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


BASELINE = Path("artifacts/exploratory/official_itransformer_adapter_pems_25_validation_20261003")
TEST = Path("artifacts/paper_final/mambatsv3_acf_infonce_official_itransformer_pems_25_extension_20261003")
ACTIVE = Path("artifacts/exploratory/mambatsv3_acf_infonce_local_cluster_pems_25_confirmation_20261003")
SEEDS = (5540, 5541, 5542)
COMMIT = "c2426e68ca13f74aaec08045c5c724d8ad328124"
HASHES = {"official_model_sha256": "6f34777a5a12c253a293f79e7e1fd5b6f79ac100b443951e051e269e7e2542db", "official_embed_sha256": "2de7e5049a696d320a23a8bdb6e08e823cf7fa3f246a0a49555fcd729c5c8398", "official_attention_sha256": "38ac67428d528b8e145e4944ca059b40629a8446c891142602d5aaadc38dfd9e", "official_encoder_sha256": "985c31f4ae187b08afaa9663dc960a40544b2ec33ab55389ceab3930edc1ce02"}


def load(path: Path): return json.loads(path.read_text(encoding="utf-8"))


def guard(path: Path):
    event = load(path)
    if event.get("status") != "completed" or event.get("trigger") is not None:
        raise ValueError("GPU guard failure: {}".format(path))


def main() -> int:
    rows = []
    for seed in SEEDS:
        stem = "pems_bay_h192_s{}_official_itransformer".format(seed)
        validation = load(BASELINE / (stem + ".json")); official = load(TEST / (stem + ".json")); active = load(ACTIVE / ("s{}_active_test.json".format(seed)))
        if validation.get("test_metrics_produced"):
            raise ValueError("official validation artifact opened test")
        if official.get("test_metrics_produced") is not True or official.get("training_performed") is not False or official.get("selection_done_before_test") is not True:
            raise ValueError("official reader lifecycle failure")
        if active.get("test_metrics_produced") is not True or active.get("training_performed") is not False:
            raise ValueError("candidate reader lifecycle failure")
        if official.get("official_commit") != COMMIT or any(official.get(key) != value for key, value in HASHES.items()):
            raise ValueError("external source lock mismatch")
        guard(BASELINE / (stem + "_event.json")); guard(TEST / (stem + "_event.json"))
        rows.append({"seed": seed, "active_test_mse": float(active["test_mse"]), "official_itransformer_test_mse": float(official["test_mse"]), "official_itransformer_validation_mse": float(validation["best_validation_mse"])})
    active = np.asarray([row["active_test_mse"] for row in rows]); official = np.asarray([row["official_itransformer_test_mse"] for row in rows]); ratio = active / official
    result = {"external_repository": "thuml/iTransformer", "official_commit": COMMIT, **HASHES, "coordinate": "PEMS-BAY H192, 25% labels", "rows": rows, "summary": {"active_mean_mse": float(active.mean()), "official_itransformer_mean_mse": float(official.mean()), "active_vs_official_itransformer_percent": float((active.mean() / official.mean() - 1) * 100), "active_wins": int((active < official).sum()), "active_official_ratio_upper_1se": float(ratio.mean() + ratio.std(ddof=1) / math.sqrt(len(ratio)))}, "audit_valid": True, "scope": "post-confirmation official-code extension; not part of active/null causal matrix"}
    (TEST / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (TEST / "readout.md").write_text("# Official iTransformer PEMS-BAY 25% Extension Audit\n\n| Active | Official iTransformer | Active vs official | Wins | One-SE upper ratio |\n| ---: | ---: | ---: | ---: | ---: |\n| {:.6f} | {:.6f} | {:+.3f}% | {}/3 | {:.4f} |\n".format(result["summary"]["active_mean_mse"], result["summary"]["official_itransformer_mean_mse"], result["summary"]["active_vs_official_itransformer_percent"], result["summary"]["active_wins"], result["summary"]["active_official_ratio_upper_1se"]), encoding="utf-8")
    print(json.dumps(result, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
