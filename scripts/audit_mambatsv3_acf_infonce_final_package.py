#!/usr/bin/env python3
"""Fail closed when the ACF-InfoNCE evidence package becomes inconsistent."""

import json
from pathlib import Path


ROOT = Path("artifacts/paper_final/mambatsv3_acf_infonce_core_bundle_20261003")
PRIMARY = {
    "pems_bay": Path("artifacts/paper_final/mambatsv3_acf_infonce_horizon_confirmatory_20261003/audit.json"),
    "metr_la": Path("artifacts/paper_final/mambatsv3_acf_infonce_metr_la_h192_confirmatory_20261003/audit.json"),
    "weather": Path("artifacts/paper_final/mambatsv3_acf_infonce_weather_h192_confirmatory_20261003/audit.json"),
}
LOCK = Path("docs/research/mambatsv3_acf_infonce_external_code_locks_20261003.json")
OUTLINE = Path("docs/research/mambatsv3_acf_infonce_manuscript_outline_20261003.md")
CLAIMS = Path("docs/research/mambatsv3_acf_infonce_manuscript_claims_20261003.csv")
SOURCES = Path("docs/research/mambatsv3_acf_infonce_source_manifest_20261003.json")


def load(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    checks, primary = {}, {}
    for name, path in PRIMARY.items():
        payload = load(path)
        valid = payload.get("audit_valid") is True and payload.get("test_metrics_produced") is True
        checks["primary_{}_audit".format(name)] = valid
        primary[name] = {"path": str(path), "valid": valid}
    bundle = load(ROOT / "bundle.json")
    audited_keys = set(bundle.get("audited_releases", {}))
    checks["bundle_has_primary"] = audited_keys == {"pems_bay_h192_h336", "metr_la_h192", "weather_h192"}
    expected = {"electricity_h96", "traffic_h96_25_percent", "weather_h192_25_percent", "forecast_affine_loss", "acf_source_ranking_h192"}
    checks["negative_boundaries_retained"] = expected.issubset(bundle.get("negative_boundaries", {}))
    checks["one_scan_contract"] = bundle.get("inference_contract") == {"mamba_scan_count": 1, "relation_loss_included": False, "autocon_pairing_included": False}
    official = load(ROOT / "official_model_extensions.json")
    rows = official.get("rows", [])
    checks["official_extension_count"] = len(rows) == 11
    checks["official_extension_wins"] = all(row.get("wins") == "3/3" and float(row.get("upper_ratio", 2)) < 1.02 for row in rows)
    official_audits = []
    for row in rows:
        payload = load(Path(row["audit"]))
        valid = payload.get("audit_valid") is True and "post-release" in payload.get("scope", "")
        official_audits.append({"path": row["audit"], "valid": valid})
    checks["official_extension_audits"] = all(item["valid"] for item in official_audits)
    source_locks = load(LOCK).get("sources", {})
    checks["external_source_locks"] = set(source_locks) == {"patchtst", "itransformer", "timesnet", "timemixer"} and all(len(record.get("commit", "")) == 40 and len(record.get("model_sha256", "")) == 64 and len(record.get("requirements_sha256", "")) == 64 for record in source_locks.values())
    checks["manuscript_is_human_gated"] = "Not for submission" in OUTLINE.read_text(encoding="utf-8") and '"status": "unverified"' in SOURCES.read_text(encoding="utf-8")
    checks["manuscript_binding_files"] = "[claim:C001]" in OUTLINE.read_text(encoding="utf-8") and "claim_id" in CLAIMS.read_text(encoding="utf-8")
    result = {"checks": checks, "primary": primary, "official_extension_audits": official_audits, "valid": all(checks.values()), "submission_ready": False, "human_gates": ["human evidence verification", "author and declaration approval", "target venue policy review", "independent clean-environment rerun"]}
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "final_package_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# Final Evidence Package Audit", "", "This audit checks internal consistency only. It does not make the work submission-ready or replace human evidence verification.", "", "| Check | Status |", "| --- | --- |"]
    lines.extend("| {} | {} |".format(name, "pass" if valid else "fail") for name, valid in checks.items())
    lines += ["", "## Human Gates", ""]
    lines.extend("- {}".format(gate) for gate in result["human_gates"])
    (ROOT / "final_package_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
