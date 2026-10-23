#!/usr/bin/env python3
"""Audit the METR-LA 25% official PatchTST extension."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


BASELINE = Path("artifacts/exploratory/official_patchtst_adapter_metr_25_validation_20261003")
TEST = Path("artifacts/paper_final/mambatsv3_acf_infonce_official_patchtst_metr_25_extension_20261003")
ACTIVE = Path("artifacts/exploratory/mambatsv3_acf_infonce_local_cluster_metr_25_confirmation_20261003")
SEEDS = (5580, 5581, 5582)
COMMIT = "204c21efe0b39603ad6e2ca640ef5896646ab1a9"
MODEL_SHA = "49d8bb865e1226d6338842f4b33c4bc6992cefb048dc2a4a5a8c410ca708586b"
BACKBONE_SHA = "df67173153787c2356bdfb6491159cd754332ef7382986efe879e1fbea8ebf26"


def load(path: Path): return json.loads(path.read_text(encoding="utf-8"))


def guard(path: Path):
    event = load(path)
    if event.get("status") != "completed" or event.get("trigger") is not None:
        raise ValueError("GPU guard failure: {}".format(path))


def main() -> int:
    rows = []
    for seed in SEEDS:
        stem = "metr_la_h192_s{}_official_patchtst".format(seed)
        validation = load(BASELINE / (stem + ".json"))
        official = load(TEST / (stem + ".json"))
        active = load(ACTIVE / ("s{}_active_test.json".format(seed)))
        if validation.get("test_metrics_produced"):
            raise ValueError("official validation artifact opened test")
        if official.get("test_metrics_produced") is not True or official.get("training_performed") is not False or official.get("selection_done_before_test") is not True:
            raise ValueError("official reader lifecycle failure")
        if active.get("test_metrics_produced") is not True or active.get("training_performed") is not False:
            raise ValueError("candidate reader lifecycle failure")
        if (official["official_commit"], official["official_model_sha256"], official["official_backbone_sha256"]) != (COMMIT, MODEL_SHA, BACKBONE_SHA):
            raise ValueError("external source lock mismatch")
        guard(BASELINE / (stem + "_event.json")); guard(TEST / (stem + "_event.json"))
        rows.append({"seed": seed, "active_test_mse": float(active["test_mse"]), "official_patchtst_test_mse": float(official["test_mse"]), "official_patchtst_validation_mse": float(validation["best_validation_mse"])})
    active = np.asarray([row["active_test_mse"] for row in rows]); official = np.asarray([row["official_patchtst_test_mse"] for row in rows]); ratio = active / official
    result = {"external_repository": "yuqinie98/PatchTST", "official_commit": COMMIT, "official_model_sha256": MODEL_SHA, "official_backbone_sha256": BACKBONE_SHA, "coordinate": "METR-LA H192, 25% labels", "rows": rows, "summary": {"active_mean_mse": float(active.mean()), "official_patchtst_mean_mse": float(official.mean()), "active_vs_official_patchtst_percent": float((active.mean() / official.mean() - 1) * 100), "active_wins": int((active < official).sum()), "active_official_ratio_upper_1se": float(ratio.mean() + ratio.std(ddof=1) / math.sqrt(len(ratio)))}, "audit_valid": True, "scope": "post-confirmation official-code extension; not part of active/null causal matrix"}
    (TEST / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (TEST / "readout.md").write_text("# Official PatchTST METR-LA 25% Extension Audit\n\n| Active | Official PatchTST | Active vs official | Wins | One-SE upper ratio |\n| ---: | ---: | ---: | ---: | ---: |\n| {:.6f} | {:.6f} | {:+.3f}% | {}/3 | {:.4f} |\n".format(result["summary"]["active_mean_mse"], result["summary"]["official_patchtst_mean_mse"], result["summary"]["active_vs_official_patchtst_percent"], result["summary"]["active_wins"], result["summary"]["active_official_ratio_upper_1se"]), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
