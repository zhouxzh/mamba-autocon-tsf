#!/usr/bin/env python3
"""Fail closed on the audited backbone decision for the local-cluster release.

This is a model-selection provenance audit, not a new evaluation.  It guards
against silently replacing the released Mamba-1 backbone with a validation-only
Mamba-2 candidate or a resource-stopped architecture.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGE = ROOT / "artifacts/paper_final/mambatsv3_local_cluster_acf_evidence_package_20261003"
MAMBA1_RELEASE = ROOT / "artifacts/paper_final/mambatsv3_mamba1_acf_frozen_test_release_20261006/bundle_audit.json"
MAMBA2_SCAN = ROOT / "artifacts/exploratory/mambatsv3_mamba2_10pct_backbone_scan_20261006/audit.json"
V13 = ROOT / "artifacts/exploratory/mambats_backbone_family_v13_screen_20261006/audit.json"
MAMBA3 = ROOT / "artifacts/exploratory/mamba3_backbone_preflight_20261008/audit.json"
OUTPUT = LOCAL_PACKAGE / "backbone_selection_audit_20261009.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    package = load(LOCAL_PACKAGE / "package_audit.json")
    bundle = load(LOCAL_PACKAGE / "bundle.json")
    mamba1 = load(MAMBA1_RELEASE)
    mamba2 = load(MAMBA2_SCAN)
    v13 = load(V13)
    mamba3 = load(MAMBA3)
    primary = mamba1.get("patchtst_test", {})
    causal = mamba1.get("paired_null_test", {})
    mamba2_targets = mamba2.get("targets", {})
    v13_pems = v13.get("targets", {}).get("pems_bay", {})
    checks = {
        "local_cluster_evidence_package_valid": package.get("valid") is True,
        "mamba1_release_audit_valid": mamba1.get("audit_valid") is True,
        "mamba1_primary_external_three_domain_wins": (
            set(primary) == {"pems_bay", "metr_la", "weather"}
            and all(row.get("active_wins") == 3 and row.get("noninferior_at_one_se") is True for row in primary.values())
        ),
        "mamba1_causal_three_domain_wins": (
            set(causal) == {"pems_bay", "metr_la", "weather"}
            and all(row.get("active_wins") == 3 and row.get("noninferior_at_one_se") is True for row in causal.values())
        ),
        "independent_active_null_program_valid": (
            bundle.get("independent_machine_replication_program", {}).get("summary", {}).get("audit_valid") is True
            and bundle.get("independent_machine_replication_program", {}).get("summary", {}).get("target_count") == 3
            and bundle.get("independent_machine_replication_program", {}).get("summary", {}).get("pooled_effect_reported") is False
        ),
        "independent_official_patchtst_extension_valid": (
            bundle.get("independent_machine_official_patchtst_extension", {}).get("summary", {}).get("audit_valid") is True
            and bundle.get("independent_machine_official_patchtst_extension", {}).get("summary", {}).get("pooled_effect_reported") is False
        ),
        "mamba2_remains_validation_only": (
            mamba2.get("validation_only") is True
            and mamba2.get("test_metrics_produced") is False
            and set(mamba2_targets) == {"pems_bay", "metr_la"}
        ),
        "v13_did_not_clear_promotion_rule": (
            v13.get("validation_only") is True
            and v13.get("test_metrics_produced") is False
            and v13_pems.get("v13_active_vs_v3_active", {}).get("wins") < 2
            and v13_pems.get("v13_active_vs_v3_active", {}).get("ratio_upper_1se", 0.0) > 1.0
        ),
        "mamba3_resource_gate_rejected": (
            mamba3.get("training_performed") is False
            and mamba3.get("resource_gate_passes") is False
            and mamba3.get("decision") == "stopped_before_forecasting_training"
        ),
    }
    result = {
        "schema_version": 1,
        "selected_backbone": {
            "name": "MambaTSv3 local-cluster, Mamba-1 mixer",
            "mixer": "mamba1",
            "inference_contract": "one scan; no AutoCon representation returned; no relation loss",
            "selection_scope": "audited three-domain low-label local-cluster release",
        },
        "secondary_ablation": {
            "name": "Mamba-2 two-head shared scan",
            "status": "validation-only architecture/accuracy ablation; not a released replacement",
            "reason": "the scanned results have no held-out test release and do not supersede the released Mamba-1 evidence line",
        },
        "stopped_families": {
            "v13": "failed its registered promotion condition in the available paired screen",
            "mamba3": "official resource preflight exceeded the latency gate before forecasting training",
            "v14_v16": "previous resource preflights are retained as stopped variants; see backbone decision memo",
        },
        "checks": checks,
        "audit_valid": all(checks.values()),
        "submission_ready": False,
        "boundary": "This audit fixes the backbone decision. It does not remove historical public-test-access caveats or replace accountable author review.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit_valid": result["audit_valid"], "checks": checks}, sort_keys=True))
    return 0 if result["audit_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
