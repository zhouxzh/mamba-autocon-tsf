#!/usr/bin/env python3
"""Fail-closed audit for numeric manuscript claims and local evidence bindings."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


NUMERIC = re.compile(r"(?<![A-Za-z])[+-]?(?:\d+(?:\.\d+)?%?|\.\d+%?)(?![A-Za-z])")
EVIDENCE_ID = re.compile(r"\bE\d{3}\b")
LOCAL_LINK = re.compile(r"\]\((?:\.\.?/|/|https?://)[^)]+\)")
CODE_OR_TABLE = re.compile(r"^\s*(?:```|\||#|[-*]\s*$)")
ORDERED_LIST = re.compile(r"^\s*\d+[.)]\s+")


def audit(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    numeric = []
    bound = []
    unresolved = []
    for number, line in enumerate(lines, 1):
        if not NUMERIC.search(line) or CODE_OR_TABLE.match(line) or ORDERED_LIST.match(line):
            continue
        item = {"line": number, "text": line.strip()}
        numeric.append(item)
        has_binding = bool(EVIDENCE_ID.search(line) or LOCAL_LINK.search(line))
        if has_binding:
            bound.append(item)
        else:
            unresolved.append(item)
    return {
        "manuscript": str(path),
        "numeric_claim_lines": len(numeric),
        "bound_lines": len(bound),
        "unresolved_lines": len(unresolved),
        "status": "review_required" if unresolved else "bound",
        "unresolved": unresolved,
        "note": "This audit is conservative: an existing nearby section link may still require human claim-level verification. It never certifies scientific correctness.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.manuscript)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("numeric_claim_lines", "bound_lines", "unresolved_lines", "status")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
