#!/usr/bin/env python3
"""Fail closed when the local-cluster ACF paper evidence package is inconsistent."""

import json
from pathlib import Path


ROOT = Path("artifacts/paper_final/mambatsv3_local_cluster_acf_evidence_package_20261003")
CLAIM_BINDINGS = ROOT / "claim_binding_audit.json"
NUMERIC_BINDINGS = ROOT / "numeric_claim_binding_audit.json"
EVIDENCE_REGISTRY = Path("docs/research/mambatsv3_acf_infonce_local_cluster_evidence_registry_20261003.md")
HUMAN_STATUS = ROOT / "human_verification_status.json"


def main() -> int:
    bundle = json.loads((ROOT / "bundle.json").read_text(encoding="utf-8"))
    claim_bindings = json.loads(CLAIM_BINDINGS.read_text(encoding="utf-8"))
    numeric_bindings = json.loads(NUMERIC_BINDINGS.read_text(encoding="utf-8"))
    evidence_registry = EVIDENCE_REGISTRY.read_text(encoding="utf-8")
    human_status = json.loads(HUMAN_STATUS.read_text(encoding="utf-8"))
    rows = bundle.get("confirmations", [])
    checks = {
        "six_heldout_confirmations": len(rows) == 6,
        "confirmation_wins": all(row["wins"] == "3/3" and row["active_vs_null_percent"] < 0 for row in rows),
        "failed_gates_closed": set(bundle.get("failed_gates", {})) == {"weather_h192_25", "pems03_h192_10"},
        "four_robustness_extensions": len(bundle.get("robustness_extensions", {})) == 4,
        "three_official_extensions": len(bundle.get("official_patchtst_extensions", {})) == 3,
        "iTransformer_extension_retained": len(bundle.get("official_itransformer_extensions", {})) == 1,
        "one_scan_inference": bundle["inference_boundary"].get("single_scan_inference_supported") is True,
        "latency_boundary_retained": bundle["inference_boundary"].get("speed_advantage_supported") is False,
        "same_machine_replication_retained": bundle.get("same_machine_independent_interpreter_replication", {}).get("summary", {}).get("audit_valid") is True,
        "clean_install_replication_retained": bundle.get("same_machine_clean_install_replication", {}).get("summary", {}).get("audit_valid") is True,
        "launcher_rehearsal_retained": bundle.get("same_machine_independent_launcher_rehearsal", {}).get("summary", {}).get("audit_valid") is True,
        "independent_machine_replications_valid": (
            set(bundle.get("independent_machine_replications", {})) == {"pems_bay_h192_10", "metr_la_h192_10", "weather_h192_10"}
            and all(
                item.get("summary", {}).get("audit_valid") is True
                and item.get("summary", {}).get("independent_machine") is True
                and item.get("summary", {}).get("host_evidence_verified", {}).get("different_from_audit_host") is True
                and item.get("summary", {}).get("active_vs_null", {}).get("wins") == 3
                for item in bundle.get("independent_machine_replications", {}).values()
            )
        ),
        "independent_replication_program_valid": (
            bundle.get("independent_machine_replication_program", {}).get("summary", {}).get("audit_valid") is True
            and bundle.get("independent_machine_replication_program", {}).get("summary", {}).get("replication_count") == 3
            and bundle.get("independent_machine_replication_program", {}).get("summary", {}).get("target_count") == 3
            and bundle.get("independent_machine_replication_program", {}).get("summary", {}).get("pooled_effect_reported") is False
        ),
        "independent_official_patchtst_extension_valid": (
            bundle.get("independent_machine_official_patchtst_extension", {}).get("summary", {}).get("audit_valid") is True
            and bundle.get("independent_machine_official_patchtst_extension", {}).get("summary", {}).get("independent_machine") is True
            and bundle.get("independent_machine_official_patchtst_extension", {}).get("summary", {}).get("pooled_effect_reported") is False
            and set(bundle.get("independent_machine_official_patchtst_extension", {}).get("summary", {}).get("targets", {})) == {"pems_bay_h192_10", "metr_la_h192_10", "weather_h192_10"}
            and all(
                item.get("summary", {}).get("active_wins") == 3 and item.get("summary", {}).get("ratio_upper_1se", float("inf")) < 1
                for item in bundle.get("independent_machine_official_patchtst_extension", {}).get("summary", {}).get("targets", {}).values()
            )
        ),
        "claim_ids_and_files_bound": claim_bindings.get("passed") is True,
        "numeric_prose_bound": numeric_bindings.get("status") == "bound",
        "human_evidence_still_unverified": "unverified" in evidence_registry,
        "human_review_gate_visible": human_status.get("human_verification_complete") is False and len(human_status.get("pending_claim_ids", [])) == 9,
        "conditional_scope": "not universal" in bundle.get("publication_scope", ""),
    }
    result = {"checks": checks, "valid": all(checks.values()), "submission_ready": False, "human_gates": ["author review of every claim and evidence artifact", "venue policy and declaration review"]}
    (ROOT / "package_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
