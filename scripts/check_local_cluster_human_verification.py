#!/usr/bin/env python3
"""Report pending human verification fields without changing their status."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worksheet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.worksheet.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"claim_id", "evidence_id", "artifact", "review_task", "verification_status", "verified_by", "verified_on", "review_notes"}
    if not rows or set(rows[0]) != required:
        raise ValueError("worksheet schema mismatch")
    pending = [row["claim_id"] for row in rows if row["verification_status"] != "verified_human"]
    incomplete = [row["claim_id"] for row in rows if row["verification_status"] == "verified_human" and (not row["verified_by"] or not row["verified_on"])]
    result = {
        "worksheet": str(args.worksheet),
        "claim_count": len(rows),
        "pending_claim_ids": pending,
        "incomplete_verified_claim_ids": incomplete,
        "human_verification_complete": not pending and not incomplete,
        "note": "This checker never changes a verification status and does not certify scientific correctness.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
