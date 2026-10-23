#!/usr/bin/env python3
"""Compare current local inputs with the independent-replication hash manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "artifacts/paper_final/mambatsv3_local_cluster_acf_evidence_package_20261003"
MANIFESTS = {
    "pems_active_null": ROOT / "artifacts/reproducibility/mambatsv3_local_cluster_independent_machine_20261008/manifest.json",
    "metr_active_null": ROOT / "artifacts/reproducibility/mambatsv3_local_cluster_metr_independent_machine_20261008/manifest.json",
    "weather_active_null": ROOT / "artifacts/reproducibility/mambatsv3_local_cluster_weather_independent_machine_20261008/manifest.json",
    "official_patchtst": ROOT / "artifacts/reproducibility/mambatsv3_local_cluster_independent_official_patchtst_20261008/manifest.json",
}
OUTPUT = PACKAGE / "source_drift_audit_20261009.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    records = {}
    all_match = True
    for name, manifest_path in MANIFESTS.items():
        manifest = load(manifest_path)
        rows = []
        for raw, expected in sorted(manifest.get("source_hashes", {}).items()):
            path = ROOT / raw
            present = path.is_file()
            actual = digest(path) if present else None
            matches = present and actual == expected
            all_match = all_match and matches
            rows.append({"path": raw, "expected_sha256": expected, "actual_sha256": actual, "present": present, "matches": matches})
        if not rows:
            raise ValueError("missing source hashes in {}".format(manifest_path))
        records[name] = {
            "manifest": str(manifest_path.relative_to(ROOT)),
            "target": manifest.get("target"),
            "hash_count": len(rows),
            "rows": rows,
            "all_match": all(row["matches"] for row in rows),
        }
    result = {
        "schema_version": 1,
        "purpose": "current-tree drift check against independent-host execution manifests",
        "replications": records,
        "source_hashes_match": all_match,
        "training_performed": False,
        "new_test_reader_opened": False,
        "audit_valid": all_match,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit_valid": result["audit_valid"], "replications": len(records)}, sort_keys=True))
    return 0 if all_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
