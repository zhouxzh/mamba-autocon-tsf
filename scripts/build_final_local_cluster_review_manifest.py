#!/usr/bin/env python3
"""Create a file-level SHA-256 manifest for the final local-cluster review pack.

The manifest is a handoff fingerprint for accountable reviewers.  It is not a
result-producing operation: it never trains, evaluates a model, or loads a
checkpoint tensor.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "artifacts/paper_final/mambatsv3_local_cluster_acf_evidence_package_20261003"
OUTPUT = PACKAGE / "review_manifest_20261009.json"
FILES = (
    "artifacts/paper_final/mambatsv3_local_cluster_acf_evidence_package_20261003/bundle.json",
    "artifacts/paper_final/mambatsv3_local_cluster_acf_evidence_package_20261003/package_audit.json",
    "artifacts/paper_final/mambatsv3_local_cluster_acf_evidence_package_20261003/backbone_selection_audit_20261009.json",
    "artifacts/paper_final/mambatsv3_local_cluster_acf_evidence_package_20261003/submission_readiness_20261008.json",
    "artifacts/paper_final/mambatsv3_local_cluster_acf_evidence_package_20261003/source_drift_audit_20261009.json",
    "artifacts/reproducibility/mambatsv3_local_cluster_independent_replication_program_20261008/audit.json",
    "artifacts/reproducibility/mambatsv3_local_cluster_independent_official_patchtst_20261008/independent_patchtst_audit.json",
    "docs/research/mambatsv3_acf_infonce_local_cluster_results_draft_20261003.md",
    "docs/research/mambatsv3_acf_infonce_local_cluster_claim_registry_20261003.md",
    "docs/research/mambatsv3_acf_infonce_local_cluster_evidence_registry_20261003.md",
    "docs/research/mambatsv3_acf_infonce_local_cluster_human_verification_worksheet_20261003.csv",
    "docs/research/mambatsv3_acf_infonce_local_cluster_human_review_20261003.md",
    "docs/research/final_backbone_selection_20261009.md",
    "docs/research/local_cluster_manuscript_outline_20261009.md",
    "scripts/pretrain_mambatsv3_core_autocon.py",
    "scripts/finetune_mambatsv3_dlinear_hybrid.py",
    "scripts/evaluate_lowdata_checkpoint_test.py",
    "scripts/official_patchtst_adapter.py",
    "scripts/evaluate_official_patchtst_checkpoint_test.py",
    "scripts/audit_final_backbone_selection.py",
    "scripts/audit_local_cluster_acf_evidence_package.py",
    "scripts/audit_local_cluster_submission_readiness.py",
    "scripts/audit_final_local_cluster_source_drift.py",
    "scripts/verify_final_local_cluster_evidence.py",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    records = []
    for raw in FILES:
        path = ROOT / raw
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append({"path": raw, "sha256": digest(path), "bytes": path.stat().st_size})
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    result = {
        "schema_version": 1,
        "purpose": "final local-cluster human-review file fingerprint",
        "file_count": len(records),
        "files": records,
        "manifest_content_sha256": hashlib.sha256(canonical).hexdigest(),
        "result_mutation": "derived_manifest_only",
        "training_performed": False,
        "new_test_reader_opened": False,
        "valid": True,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("valid", "file_count", "manifest_content_sha256")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
