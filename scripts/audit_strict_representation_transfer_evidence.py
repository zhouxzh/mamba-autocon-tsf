#!/usr/bin/env python3
"""Audit the complete strict representation-transfer evidence boundary."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = {
    "pems_10": ROOT / "artifacts/exploratory/mambatsv3_pems_frozen_core_probe_20261003/audit.json",
    "pems_25": ROOT / "artifacts/exploratory/mambatsv3_pems_frozen_core_25_validation_20261003/audit.json",
    "metr_10": ROOT / "artifacts/exploratory/mambatsv3_metr_frozen_core_validation_20261003/audit.json",
    "weather_10": ROOT / "artifacts/exploratory/mambatsv3_weather_frozen_core_validation_20261003/audit.json",
}
ROBUSTNESS = {
    "pems_10": ROOT / "artifacts/exploratory/mambatsv3_pems_frozen_core_robustness_20261003/audit.json",
    "weather_10": ROOT / "artifacts/exploratory/mambatsv3_weather_frozen_core_robustness_20261003/audit.json",
}
POSTVALIDATION = ROOT / "artifacts/exploratory/mambatsv3_strict_frozen_core_postvalidation_test_20261003/audit.json"
NEGATIVE_BOUNDARY = ROOT / "docs/research/mambatsv3_acf_infonce_highdim_mechanism_negative_readout_20261003.md"
READOUT = ROOT / "docs/research/mambatsv3_strict_representation_transfer_readout_20261003.md"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    validation = {name: _load(path) for name, path in VALIDATION.items()}
    robustness = {name: _load(path) for name, path in ROBUSTNESS.items()}
    postvalidation = _load(POSTVALIDATION)
    checks = {
        "four_validation_matrices": len(validation) == 4,
        "validation_all_seed_wins": all(item["all_seed_wins"] for item in validation.values()),
        "validation_test_closed": all(item["test_metrics_produced"] is False for item in validation.values()),
        "robustness_all_conditions_all_seed_wins": all(item["all_conditions_all_seed_wins"] for item in robustness.values()),
        "robustness_test_closed": all(item["test_metrics_produced"] is False for item in robustness.values()),
        "postvalidation_is_read_only": bool(postvalidation["read_only"]),
        "postvalidation_scope_discloses_prior_test_access": "not globally pristine" in postvalidation["scope"],
        "postvalidation_all_coordinates_win": all(item["active_wins"] == 3 for item in postvalidation["matrices"].values()),
        "negative_boundary_retained": NEGATIVE_BOUNDARY.is_file() and "negative transfer" in NEGATIVE_BOUNDARY.read_text(encoding="utf-8"),
        "conditional_scope_visible": READOUT.is_file() and "do not support a universal claim" in READOUT.read_text(encoding="utf-8"),
        "single_scan_boundary_visible": READOUT.is_file() and "one MambaTSv3 scan" in READOUT.read_text(encoding="utf-8"),
    }
    result = {"valid": bool(all(checks.values())), "checks": checks,
              "submission_ready": False,
              "human_gates": ["independent-machine replication", "author claim review", "venue and declaration review"],
              "scope": "strict frozen-core evidence is positive but conditional; high-dimensional negative domains remain retained"}
    output = ROOT / "artifacts/exploratory/mambatsv3_strict_representation_transfer_evidence_20261003/audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
