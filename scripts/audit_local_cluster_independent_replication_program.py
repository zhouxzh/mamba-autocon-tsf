#!/usr/bin/env python3
"""Summarize independently audited local-cluster replications without pooling domains."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/reproducibility/mambatsv3_local_cluster_independent_replication_program_20261008"
REPLICATIONS = {
    "pems_bay_h192_10": {
        "target": "PEMS-BAY", "horizon": 192, "few_shot_fraction": .1,
        "path": ROOT / "artifacts/reproducibility/mambatsv3_local_cluster_independent_machine_20261008/independent_audit.json",
    },
    "metr_la_h192_10": {
        "target": "METR-LA", "horizon": 192, "few_shot_fraction": .1,
        "path": ROOT / "artifacts/reproducibility/mambatsv3_local_cluster_metr_independent_machine_20261008/independent_audit.json",
    },
    "weather_h192_10": {
        "target": "Weather", "horizon": 192, "few_shot_fraction": .1,
        "path": ROOT / "artifacts/reproducibility/mambatsv3_local_cluster_weather_independent_machine_20261008/independent_audit.json",
    },
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    rows = []
    host_fingerprints = set()
    for key, spec in REPLICATIONS.items():
        payload = load(spec["path"])
        summary = payload.get("active_vs_null", {})
        host = payload.get("host_evidence_verified", {})
        if (
            payload.get("audit_valid") is not True
            or payload.get("independent_machine") is not True
            or payload.get("test_metrics_produced") is not True
            or host.get("different_from_audit_host") is not True
            or summary.get("wins") != 3
            or summary.get("percent", 0.0) >= 0.0
            or summary.get("ratio_upper_1se", float("inf")) >= 1.0
        ):
            raise ValueError("independent replication does not pass program contract: {}".format(key))
        host_fingerprints.add((host.get("hostname"), host.get("gpu"), host.get("torch"), host.get("mamba_ssm")))
        rows.append({
            "key": key,
            "target": spec["target"],
            "horizon": spec["horizon"],
            "few_shot_fraction": spec["few_shot_fraction"],
            "active_mean_test_mse": float(summary["active_mean_test_mse"]),
            "null_mean_test_mse": float(summary["null_mean_test_mse"]),
            "active_vs_null_percent": float(summary["percent"]),
            "wins": int(summary["wins"]),
            "ratio_upper_1se": float(summary["ratio_upper_1se"]),
            "path": str(spec["path"]),
        })
    result = {
        "program": "independent-host local-cluster ACF-InfoNCE replication",
        "independent_machine": True,
        "test_metrics_produced": True,
        "replication_count": len(rows),
        "target_count": len({row["target"] for row in rows}),
        "host_fingerprint_count": len(host_fingerprints),
        "rows": rows,
        "all_domains_improve": all(row["active_vs_null_percent"] < 0.0 for row in rows),
        "all_domains_three_seed_wins": all(row["wins"] == 3 for row in rows),
        "all_domains_one_se_noninferior": all(row["ratio_upper_1se"] < 1.0 for row in rows),
        "pooled_effect_reported": False,
        "pooling_boundary": "Cross-domain effect sizes are intentionally not pooled because targets have different data-generating processes and metric scales.",
        "audit_valid": True,
        "submission_ready": False,
        "human_gates": ["author review of every claim and evidence artifact", "venue policy and declaration review"],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Independent Replication Program Audit", "",
        "This read-only summary verifies three independently audited target-domain replications. It intentionally does not pool their effect sizes.", "",
        "| Target | Horizon | Labels | Active test MSE | Null test MSE | Active vs null | Wins | One-SE upper |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append("| {target} | {horizon} | {few_shot_fraction:.0%} | {active_mean_test_mse:.6f} | {null_mean_test_mse:.6f} | {active_vs_null_percent:+.3f}% | {wins}/3 | {ratio_upper_1se:.5f} |".format(**row))
    lines += [
        "", "## Boundary", "",
        "The three target domains share one independently reviewed host but have different datasets and scales. Their results support cross-domain reproducibility of direction, not a pooled universal effect estimate.",
    ]
    (OUTPUT / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
