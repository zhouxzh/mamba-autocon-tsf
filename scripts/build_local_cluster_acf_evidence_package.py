#!/usr/bin/env python3
"""Build a read-only paper evidence package for local-cluster ACF-InfoNCE."""

from __future__ import annotations

import json
from pathlib import Path


OUTPUT = Path("artifacts/paper_final/mambatsv3_local_cluster_acf_evidence_package_20261003")
CONFIRMATIONS = {
    "pems_h192_10": ("PEMS-BAY", 192, .10, Path("artifacts/exploratory/mambatsv3_acf_infonce_local_cluster_pems_confirmation_20261003/audit.json")),
    "pems_h192_25": ("PEMS-BAY", 192, .25, Path("artifacts/exploratory/mambatsv3_acf_infonce_local_cluster_pems_25_confirmation_20261003/audit.json")),
    "pems_h336_10": ("PEMS-BAY", 336, .10, Path("artifacts/exploratory/mambatsv3_acf_infonce_local_cluster_pems_h336_confirmation_20261003/audit.json")),
    "metr_h192_10": ("METR-LA", 192, .10, Path("artifacts/exploratory/mambatsv3_acf_infonce_local_cluster_metr_confirmation_20261003/audit.json")),
    "metr_h192_25": ("METR-LA", 192, .25, Path("artifacts/exploratory/mambatsv3_acf_infonce_local_cluster_metr_25_confirmation_20261003/audit.json")),
    "weather_h192_10": ("Weather", 192, .10, Path("artifacts/exploratory/mambatsv3_acf_infonce_local_cluster_weather_confirmation_20261003/audit.json")),
}
FAILED_GATES = {
    "weather_h192_25": Path("artifacts/exploratory/mambatsv3_acf_infonce_local_cluster_weather_25_confirmation_20261003/gate_audit.json"),
    "pems03_h192_10": Path("artifacts/exploratory/mambatsv3_acf_infonce_local_cluster_pems03_external_confirmation_20261003/gate_audit.json"),
}
ROBUSTNESS = {
    "pems_h192_10": Path("artifacts/exploratory/mambatsv3_acf_infonce_local_cluster_robustness_extension_20261003/pems_bay_robustness_audit.json"),
    "metr_h192_10": Path("artifacts/exploratory/mambatsv3_acf_infonce_local_cluster_robustness_extension_20261003/metr_la_robustness_audit.json"),
    "weather_h192_10": Path("artifacts/exploratory/mambatsv3_acf_infonce_local_cluster_robustness_extension_20261003/weather_robustness_audit.json"),
    "pems_h336_10": Path("artifacts/exploratory/mambatsv3_acf_infonce_local_cluster_pems_h336_confirmation_20261003/pems_bay_h336_robustness_audit.json"),
}
OFFICIAL = {
    "pems_h192_25": Path("artifacts/paper_final/mambatsv3_acf_infonce_official_patchtst_pems_25_extension_20261003/audit.json"),
    "metr_h192_25": Path("artifacts/paper_final/mambatsv3_acf_infonce_official_patchtst_metr_25_extension_20261003/audit.json"),
    "pems_h336_10": Path("artifacts/paper_final/mambatsv3_acf_infonce_official_patchtst_pems_h336_extension_20261003/audit.json"),
}
OFFICIAL_ITRANSFORMER = {
    "pems_h192_25": Path("artifacts/paper_final/mambatsv3_acf_infonce_official_itransformer_pems_25_extension_20261003/audit.json"),
}
INFERENCE_BOUNDARY = Path("artifacts/paper_final/mambatsv3_acf_infonce_official_patchtst_pems_h336_extension_20261003/inference_boundary.json")
REPLICATION = Path("artifacts/exploratory/mambatsv3_acf_infonce_clean_environment_replication_20261003/audit.json")
CLEAN_INSTALL_REPLICATION = Path("artifacts/exploratory/mambatsv3_acf_infonce_clean_install_replication_20261003/audit.json")
LAUNCHER_REHEARSAL = Path("artifacts/reproducibility/independent_machine_replication_20261003/rehearsal_audit.json")
INDEPENDENT_MACHINE = {
    "pems_bay_h192_10": Path("artifacts/reproducibility/mambatsv3_local_cluster_independent_machine_20261008/independent_audit.json"),
    "metr_la_h192_10": Path("artifacts/reproducibility/mambatsv3_local_cluster_metr_independent_machine_20261008/independent_audit.json"),
    "weather_h192_10": Path("artifacts/reproducibility/mambatsv3_local_cluster_weather_independent_machine_20261008/independent_audit.json"),
}
INDEPENDENT_PROGRAM = Path("artifacts/reproducibility/mambatsv3_local_cluster_independent_replication_program_20261008/audit.json")
INDEPENDENT_OFFICIAL_PATCHTST = Path("artifacts/reproducibility/mambatsv3_local_cluster_independent_official_patchtst_20261008/independent_patchtst_audit.json")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    rows, sources = [], {}
    for key, (target, horizon, fraction, path) in CONFIRMATIONS.items():
        payload = load(path)
        if payload.get("audit_valid") is not True or payload.get("test_metrics_produced") is not True:
            raise ValueError("invalid confirmation: {}".format(path))
        summary = payload["summary"]
        rows.append({
            "key": key, "target": target, "horizon": horizon,
            "few_shot_fraction": fraction,
            "active_mse": summary["active_mean_test_mse"],
            "null_mse": summary["null_mean_test_mse"],
            "active_vs_null_percent": summary["active_vs_null_percent"],
            "wins": "{}/3".format(summary["active_wins"]),
            "path": str(path),
        })
        sources[key] = str(path)
    boundaries = {}
    for key, path in FAILED_GATES.items():
        payload = load(path)
        if payload.get("audit_valid") is not True or payload.get("test_metrics_produced") is not False or payload.get("all_seed_gate_passed") is not False:
            raise ValueError("invalid failed gate: {}".format(path))
        boundaries[key] = {"path": str(path), "rows": payload["rows"]}
    robustness = {}
    for key, path in ROBUSTNESS.items():
        payload = load(path)
        if payload.get("audit_valid") is not True or payload.get("test_metrics_produced") is not False:
            raise ValueError("invalid robustness extension: {}".format(path))
        if not all(item["active_wins"] == 3 for item in payload["summary"].values()):
            raise ValueError("robustness direction is not uniformly positive: {}".format(path))
        robustness[key] = {"path": str(path), "summary": payload["summary"]}
    official = {}
    for key, path in OFFICIAL.items():
        payload = load(path)
        if payload.get("audit_valid") is not True or payload["summary"]["active_wins"] != 3:
            raise ValueError("invalid official extension: {}".format(path))
        official[key] = {"path": str(path), "summary": payload["summary"]}
    official_itransformer = {}
    for key, path in OFFICIAL_ITRANSFORMER.items():
        payload = load(path)
        if payload.get("audit_valid") is not True or payload["summary"]["active_wins"] != 3:
            raise ValueError("invalid official iTransformer extension: {}".format(path))
        official_itransformer[key] = {"path": str(path), "summary": payload["summary"]}
    inference = load(INFERENCE_BOUNDARY)
    if inference.get("audit_valid") is not True or inference.get("single_scan_inference_supported") is not True:
        raise ValueError("invalid inference boundary")
    replication = load(REPLICATION)
    if replication.get("audit_valid") is not True or replication.get("same_machine_independent_interpreter") is not True or replication.get("independent_machine") is not False:
        raise ValueError("invalid same-machine replication")
    clean_install = load(CLEAN_INSTALL_REPLICATION)
    if clean_install.get("audit_valid") is not True or clean_install.get("clean_install") is not True or clean_install.get("independent_machine") is not False:
        raise ValueError("invalid clean-install replication")
    launcher_rehearsal = load(LAUNCHER_REHEARSAL)
    if launcher_rehearsal.get("audit_valid") is not True or launcher_rehearsal.get("launcher_rehearsal") is not True or launcher_rehearsal.get("independent_machine") is not False:
        raise ValueError("invalid launcher rehearsal")
    independent_machine = {}
    for key, path in INDEPENDENT_MACHINE.items():
        payload = load(path)
        summary = payload.get("active_vs_null", {})
        if (
            payload.get("audit_valid") is not True
            or payload.get("independent_machine") is not True
            or payload.get("test_metrics_produced") is not True
            or summary.get("wins") != 3
            or summary.get("percent", 0.0) >= 0.0
            or payload.get("host_evidence_verified", {}).get("different_from_audit_host") is not True
        ):
            raise ValueError("invalid independent-machine replication: {}".format(key))
        independent_machine[key] = {"path": str(path), "summary": payload}
    independent_program = load(INDEPENDENT_PROGRAM)
    if (
        independent_program.get("audit_valid") is not True
        or independent_program.get("replication_count") != 3
        or independent_program.get("target_count") != 3
        or independent_program.get("all_domains_improve") is not True
        or independent_program.get("all_domains_three_seed_wins") is not True
        or independent_program.get("all_domains_one_se_noninferior") is not True
        or independent_program.get("pooled_effect_reported") is not False
    ):
        raise ValueError("invalid independent replication program audit")
    independent_official_patchtst = load(INDEPENDENT_OFFICIAL_PATCHTST)
    official_targets = independent_official_patchtst.get("targets", {})
    if (
        independent_official_patchtst.get("audit_valid") is not True
        or independent_official_patchtst.get("independent_machine") is not True
        or independent_official_patchtst.get("pooled_effect_reported") is not False
        or independent_official_patchtst.get("official_repository") != "yuqinie98/PatchTST"
        or set(official_targets) != {"pems_bay_h192_10", "metr_la_h192_10", "weather_h192_10"}
        or any(item.get("summary", {}).get("active_wins") != 3 or item["summary"].get("ratio_upper_1se", float("inf")) >= 1 for item in official_targets.values())
    ):
        raise ValueError("invalid independent-host official PatchTST extension")
    result = {
        "bundle_version": 1,
        "method": "channel-count-agnostic one-scan MambaTSv3 residual pretraining with signed forecast-level ACF InfoNCE and local contrastive clusters",
        "confirmations": rows,
        "failed_gates": boundaries,
        "robustness_extensions": robustness,
        "official_patchtst_extensions": official,
        "official_itransformer_extensions": official_itransformer,
        "inference_boundary": inference,
        "same_machine_independent_interpreter_replication": {"path": str(REPLICATION), "summary": replication},
        "same_machine_clean_install_replication": {"path": str(CLEAN_INSTALL_REPLICATION), "summary": clean_install},
        "same_machine_independent_launcher_rehearsal": {"path": str(LAUNCHER_REHEARSAL), "summary": launcher_rehearsal},
        "independent_machine_replications": independent_machine,
        "independent_machine_replication_program": {"path": str(INDEPENDENT_PROGRAM), "summary": independent_program},
        "independent_machine_official_patchtst_extension": {"path": str(INDEPENDENT_OFFICIAL_PATCHTST), "summary": independent_official_patchtst},
        "publication_scope": "conditional low-label transfer for named periodic sensor targets; not universal multivariate forecasting",
        "submission_ready": False,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "bundle.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Local-Cluster ACF Evidence Package", "",
        "This read-only package separates held-out confirmation, validation-only robustness, failed gates, external-code extensions, and inference boundaries.", "",
        "## Held-Out Confirmations", "",
        "| Target | H | Labels | Active | Exact-null | Active vs null | Wins |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append("| {target} | {horizon} | {few_shot_fraction:.0%} | {active_mse:.6f} | {null_mse:.6f} | {active_vs_null_percent:+.3f}% | {wins} |".format(**row))
    lines += ["", "## Failed Gates", ""]
    for key, value in boundaries.items():
        lines.append("- `{}`: all-seed validation gate failed; test closed (`{}`).".format(key, value["path"]))
    lines += ["", "## Replication", "", "- Same-machine independent-interpreter PEMS-BAY H192/10% replication: {:+.3f}% active versus null; it is not a second-machine replication and is excluded from the primary effect estimate.".format(replication["active_vs_null_percent"]), "- Same-machine clean-install PEMS-BAY H192/10% replication with locked local cp39 binary artifacts: {:+.3f}% active versus null; it is not a second-machine or from-source binary-build replication and is excluded from the primary effect estimate.".format(clean_install["active_vs_null_percent"]), "- Same-machine execution rehearsal of the portable independent-machine launcher: {:+.3f}% active versus null; all guard stages passed, but it remains same-machine support evidence.".format(launcher_rehearsal["active_vs_null_percent"])]
    for key, value in independent_machine.items():
        summary = value["summary"]["active_vs_null"]
        lines.append("- Independent-machine {} replication: {:+.3f}% active versus null, {}/3 wins, one-SE upper ratio {:.5f}; a reviewer-verified distinct RTX 3060 Ti host completed all guarded stages.".format(key, summary["percent"], summary["wins"], summary["ratio_upper_1se"]))
    lines += ["", "## Independent-Host Official PatchTST Extension", ""]
    for value in official_targets.values():
        summary = value["summary"]
        lines.append("- {} H192/10%: active versus official PatchTST {:+.3f}%, {}/3 wins, one-SE upper ratio {:.5f}. This is a post-release external-code extension and is reported separately by target.".format(value["target"], summary["active_vs_official_patchtst_percent"], summary["active_wins"], summary["ratio_upper_1se"]))
    lines += ["", "## Scope", "", "- Positive claims are conditional on the named held-out coordinates.", "- Robustness extensions are validation-only and post-confirmation.", "- Official PatchTST and iTransformer comparisons are post-confirmation external-code extensions.", "- H336 has a clean one-scan inference path but is slower than official PatchTST on the matched latency shape; it uses less peak allocated memory."]
    (OUTPUT / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
