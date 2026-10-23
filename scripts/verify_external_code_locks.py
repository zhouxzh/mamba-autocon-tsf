#!/usr/bin/env python3
"""Verify pinned external model files against local hash locks."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--checkout", action="append", required=True)
    args = parser.parse_args()
    sources = json.loads(args.lock.read_text(encoding="utf-8"))["sources"]
    checkouts = {}
    for item in args.checkout:
        if "=" not in item:
            raise ValueError("--checkout must use NAME=PATH")
        name, raw_path = item.split("=", 1)
        if not name or name in checkouts:
            raise ValueError("checkout names must be unique and nonempty")
        checkouts[name] = Path(raw_path).resolve()
    if set(checkouts) != set(sources):
        raise ValueError("checkout names must exactly match source locks")
    rows = []
    for name, record in sorted(sources.items()):
        model = checkouts[name] / record["model_path"]
        requirements = checkouts[name] / record["requirements_path"]
        if not model.is_file() or not requirements.is_file():
            raise FileNotFoundError("missing locked file for {}".format(name))
        model_hash, requirements_hash = digest(model), digest(requirements)
        rows.append({"name": name, "model_sha256": model_hash, "requirements_sha256": requirements_hash, "valid": model_hash == record["model_sha256"] and requirements_hash == record["requirements_sha256"]})
    result = {"lock": str(args.lock), "rows": rows, "valid": all(row["valid"] for row in rows)}
    print(json.dumps(result, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
