#!/usr/bin/env python3
"""Validate manuscript C/E tags against local claim and evidence registries."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


CLAIM_TAG = re.compile(r"\[claim:([C\d,]+)\]")
EVIDENCE_TAG = re.compile(r"\[evidence:([E\d,]+)\]")
CLAIM_ROW = re.compile(r"^\|\s*(C\d{3})\s*\|")
EVIDENCE_ROW = re.compile(r"^\|\s*(E\d{3})\s*\|")
CODE_PATH = re.compile(r"`([^`]+)`")


def tag_ids(pattern: re.Pattern[str], text: str) -> set[str]:
    result = set()
    for group in pattern.findall(text):
        result.update(item.strip() for item in group.split(",") if item.strip())
    return result


def row_ids(pattern: re.Pattern[str], text: str) -> set[str]:
    return {
        matched.group(1)
        for line in text.splitlines()
        if (matched := pattern.match(line))
    }


def evidence_paths(registry: Path) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for line in registry.read_text(encoding="utf-8").splitlines():
        matched = EVIDENCE_ROW.match(line)
        if not matched:
            continue
        output[matched.group(1)] = CODE_PATH.findall(line)
    return output


def audit(manuscript: Path, claims: Path, evidence_registry: Path, repo_root: Path) -> dict:
    manuscript_text = manuscript.read_text(encoding="utf-8")
    claim_text = claims.read_text(encoding="utf-8")
    evidence_text = evidence_registry.read_text(encoding="utf-8")
    used_claims = tag_ids(CLAIM_TAG, manuscript_text)
    used_evidence = tag_ids(EVIDENCE_TAG, manuscript_text)
    declared_claims = row_ids(CLAIM_ROW, claim_text)
    declared_evidence = row_ids(EVIDENCE_ROW, evidence_text)
    paths = evidence_paths(evidence_registry)
    missing_claims = sorted(used_claims - declared_claims)
    missing_evidence = sorted(used_evidence - declared_evidence)
    missing_files: dict[str, list[str]] = {}
    for evidence_id in sorted(used_evidence & declared_evidence):
        absent = []
        for raw in paths.get(evidence_id, []):
            candidate = repo_root / raw
            if not candidate.exists():
                absent.append(raw)
        if absent:
            missing_files[evidence_id] = absent
    report = {
        "manuscript": str(manuscript),
        "used_claim_ids": sorted(used_claims),
        "used_evidence_ids": sorted(used_evidence),
        "missing_claim_ids": missing_claims,
        "missing_evidence_ids": missing_evidence,
        "missing_evidence_files": missing_files,
        "passed": not missing_claims and not missing_evidence and not missing_files,
        "note": "This checks local identifier and file reachability only. It does not prove that a source supports a claim or that a manuscript is submission-ready.",
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript", type=Path, required=True)
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--evidence-registry", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.manuscript, args.claims, args.evidence_registry, args.repo_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "claims": len(report["used_claim_ids"]), "evidence": len(report["used_evidence_ids"]), "missing_files": len(report["missing_evidence_files"])}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
