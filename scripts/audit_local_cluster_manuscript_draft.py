#!/usr/bin/env python3
"""Audit numeric prose in the local-cluster manuscript draft.

This intentionally excludes document metadata, Mermaid source, tables, and
reference footnotes.  Those contain dates, style color codes, and bibliographic
years rather than experimental claims.  Numeric prose in the manuscript body
must carry an inline internal claim or evidence identifier.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


NUMERIC = re.compile(r"(?<![A-Za-z])[+-]?(?:\d+(?:\.\d+)?%?|\.\d+%?)(?![A-Za-z])")
BINDING = re.compile(r"\[(?:claim:C\d{3}|evidence:E\d{3})")
FOOTNOTE = re.compile(r"\[\^\d+\]")
ORDERED_LIST = re.compile(r"^\d+[.)]\s+")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lines = args.manuscript.read_text(encoding="utf-8").splitlines()
    in_code = False
    numeric, bound, unresolved = [], [], []
    for number, line in enumerate(lines, 1):
        if line.startswith("```"):
            in_code = not in_code
            continue
        stripped = line.strip()
        if (
            in_code
            or not stripped
            or stripped.startswith(("#", "_", "|", "[^", "*表", "*图"))
            or stripped.startswith(">")
            or ORDERED_LIST.match(stripped)
        ):
            continue
        # Citation footnote labels are bibliography syntax, not reported data.
        claim_text = FOOTNOTE.sub("", stripped)
        if not NUMERIC.search(claim_text):
            continue
        item = {"line": number, "text": stripped}
        numeric.append(item)
        if BINDING.search(stripped):
            bound.append(item)
        else:
            unresolved.append(item)
    result = {
        "manuscript": str(args.manuscript),
        "scope": "numeric experimental prose; metadata, Mermaid, tables, captions, and references excluded",
        "numeric_prose_lines": len(numeric),
        "bound_lines": len(bound),
        "unresolved_lines": len(unresolved),
        "unresolved": unresolved,
        "status": "bound" if not unresolved else "review_required",
        "note": "This is a local binding check, not a human verification of scientific correctness.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("numeric_prose_lines", "bound_lines", "unresolved_lines", "status")}))
    return 0 if not unresolved else 1


if __name__ == "__main__":
    raise SystemExit(main())
