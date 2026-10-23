#!/usr/bin/env python
"""Validate the immutable data contract registered for high-dimensional runs.

This gate intentionally depends only on the Python standard library.  It does
not import the training data loader or rewrite a dataset/quality report.  The
CSV is scanned for its header and row widths, while SHA256 digests and the
registered quality report are checked independently.  A report-file digest is
always emitted; a contract may optionally pin that digest with
``quality_report_content_sha256`` (or one of the accepted compatibility
aliases).  The content digest ignores volatile ``generated_at_utc`` and
machine-specific ``path`` fields, so re-running the audit does not invalidate
an otherwise identical report.

Examples
--------
    python scripts/validate_highdim_contract.py --strict
    python scripts/validate_highdim_contract.py --strict \
        --output artifacts/paper_final/corrected/highdim_data_quality/contract.json
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_PLAN = REPO_ROOT / "docs/research/highdim_training_plan.json"
_REPORT_HASH_KEYS = (
    "quality_report_sha256",
    "quality_report_file_sha256",
    "report_file_sha256",
)
_REPORT_CONTENT_HASH_KEYS = (
    "quality_report_content_sha256",
    "quality_report_canonical_sha256",
    "quality_report_hash",
    "report_sha256",
    "report_hash",
)


def _sha256(path: Path) -> str:
    """Return the SHA256 digest of *path* without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


sha256_file = _sha256


def _canonical_report_hash(report: dict[str, Any]) -> str:
    """Hash stable report content while excluding run-location metadata."""

    stable = dict(report)
    stable.pop("generated_at_utc", None)
    stable.pop("path", None)
    encoded = json.dumps(
        stable, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_path(value: str | os.PathLike[str], root: Path) -> Path:
    """Resolve a contract path and reject paths escaping the repository root."""

    candidate = Path(value)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    root_resolved = root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"path escapes repository root: {value!r}") from exc
    return resolved


def _first_present(mapping: dict[str, Any], keys: Iterable[str]) -> tuple[str | None, Any]:
    for key in keys:
        if key in mapping:
            return key, mapping[key]
    return None, None


def _hash_check(expected: Any, actual: str, label: str) -> tuple[dict[str, Any], list[str]]:
    expected_text = str(expected).lower() if expected is not None else None
    valid_shape = expected_text is None or (
        len(expected_text) == 64 and all(char in "0123456789abcdef" for char in expected_text)
    )
    errors: list[str] = []
    if expected_text is not None and not valid_shape:
        errors.append(f"{label}_expected_invalid")
    matches = expected_text is not None and valid_shape and actual == expected_text
    if expected_text is not None and valid_shape and not matches:
        errors.append(f"{label}_mismatch")
    return {
        "expected": expected_text,
        "actual": actual,
        "match": matches if expected_text is not None and valid_shape else None,
        "registered": expected_text is not None,
    }, errors


def _scan_csv(path: Path, time_column: str, target: str, expected_channels: Any) -> tuple[dict[str, Any], list[str]]:
    """Read only CSV structure, retaining bounded diagnostics for bad rows."""

    errors: list[str] = []
    expected = expected_channels if isinstance(expected_channels, int) and not isinstance(expected_channels, bool) else None
    header: list[str] = []
    row_count = 0
    empty_rows = 0
    bad_width_count = 0
    bad_width_examples: list[dict[str, int]] = []
    csv_error: str | None = None

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            try:
                header = [str(item) for item in next(reader)]
            except StopIteration:
                errors.append("csv_header_missing")
            except csv.Error as exc:
                csv_error = str(exc)
                errors.append("csv_parse_error")
            if header:
                if not header[0] == time_column:
                    errors.append("csv_time_column_mismatch")
                if target not in header:
                    errors.append("csv_target_missing")
                if any(not item for item in header):
                    errors.append("csv_empty_column_name")
                if len(set(header)) != len(header):
                    errors.append("csv_duplicate_column_names")
                actual_channels = len(header) - 1 if header and header[0] == time_column else max(len(header) - 1, 0)
                if expected is None:
                    errors.append("contract_channels_invalid")
                elif actual_channels != expected:
                    errors.append("csv_channel_count_mismatch")
                for row_number, row in enumerate(reader, start=2):
                    if not row:
                        empty_rows += 1
                        continue
                    row_count += 1
                    if len(row) != len(header):
                        bad_width_count += 1
                        if len(bad_width_examples) < 5:
                            bad_width_examples.append({"row": row_number, "fields": len(row)})
            # ``csv.reader`` can raise while iteration is in progress.
            # The explicit loop above is kept inside the try for that reason.
    except (OSError, UnicodeError, csv.Error) as exc:
        if csv_error is None:
            csv_error = str(exc)
            errors.append("csv_read_error")

    if empty_rows:
        errors.append("csv_empty_rows")
    if bad_width_count:
        errors.append("csv_row_width_mismatch")
    if csv_error and "csv_parse_error" not in errors and "csv_read_error" not in errors:
        errors.append("csv_parse_error")

    actual_channels = len(header) - 1 if header else None
    schema = {
        "time_column": time_column,
        "target": target,
        "columns": header,
        "value_columns": header[1:] if header else [],
        "rows": row_count,
        "channels_including_target": actual_channels,
        "expected_channels_including_target": expected,
        "empty_rows": empty_rows,
        "row_width_errors": bad_width_count,
        "row_width_examples": bad_width_examples,
    }
    return schema, errors


def _quality_report_checks(
    report_path: Path,
    dataset: str,
    contract: dict[str, Any],
    csv_schema: dict[str, Any],
    data_sha256: str,
    data_bytes: int,
) -> tuple[dict[str, Any], list[str]]:
    """Validate a quality report and its binding to the current CSV."""

    errors: list[str] = []
    report_hash: str | None = None
    report_content_hash: str | None = None
    expected_report_key, expected_report_hash = _first_present(contract, _REPORT_HASH_KEYS)
    expected_content_key, expected_content_hash = _first_present(
        contract, _REPORT_CONTENT_HASH_KEYS
    )
    report_info: dict[str, Any] = {
        "path": str(report_path),
        "exists": report_path.is_file(),
        "sha256": None,
        "content_sha256": None,
        "hash": {
            "expected": str(expected_report_hash).lower() if expected_report_hash is not None else None,
            "actual": None,
            "match": None,
            "registered": expected_report_key is not None,
            "field": expected_report_key,
        },
        "content_hash": {
            "expected": str(expected_content_hash).lower() if expected_content_hash is not None else None,
            "actual": None,
            "match": None,
            "registered": expected_content_key is not None,
            "field": expected_content_key,
        },
        "quality_pass": None,
        "dataset_sha256": None,
        "schema_match": None,
    }
    if not report_path.is_file():
        errors.append("quality_report_missing")
        return report_info, errors

    try:
        report_hash = _sha256(report_path)
    except OSError:
        errors.append("quality_report_unreadable")
        return report_info, errors
    report_info["sha256"] = report_hash

    try:
        with report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError):
        errors.append("quality_report_invalid_json")
        return report_info, errors
    if not isinstance(report, dict):
        errors.append("quality_report_not_object")
        return report_info, errors
    report_content_hash = _canonical_report_hash(report)
    report_info["content_sha256"] = report_content_hash
    file_hash_label = expected_report_key or "quality_report_file_sha256"
    hash_check, hash_errors = _hash_check(expected_report_hash, report_hash, file_hash_label)
    hash_check["field"] = expected_report_key
    report_info["hash"] = hash_check
    errors.extend(hash_errors)
    content_check, content_errors = _hash_check(
        expected_content_hash, report_content_hash, "quality_report_content_sha256"
    )
    content_check["field"] = expected_content_key
    report_info["content_hash"] = content_check
    errors.extend(content_errors)

    quality_pass = report.get("quality_pass")
    report_info["quality_pass"] = quality_pass if isinstance(quality_pass, bool) else None
    if quality_pass is not True:
        errors.append("quality_report_quality_pass_false")
    quality_errors = report.get("quality_errors")
    if quality_pass is True and quality_errors not in ([], None):
        errors.append("quality_report_errors_not_empty")

    recorded_data_hash = report.get("sha256")
    report_info["dataset_sha256"] = recorded_data_hash
    if not isinstance(recorded_data_hash, str) or recorded_data_hash.lower() != data_sha256:
        errors.append("quality_report_dataset_sha256_mismatch")
    if report.get("dataset") != dataset:
        errors.append("quality_report_dataset_mismatch")
    if report.get("bytes") is not None and report.get("bytes") != data_bytes:
        errors.append("quality_report_bytes_mismatch")

    report_schema = report.get("schema")
    schema_match = False
    if isinstance(report_schema, dict):
        schema_match = (
            report_schema.get("time_column") == csv_schema["time_column"]
            and report_schema.get("target") == csv_schema["target"]
            and report_schema.get("columns") == csv_schema["columns"]
            and report_schema.get("value_columns") == csv_schema["value_columns"]
            and report_schema.get("rows") == csv_schema["rows"]
            and report_schema.get("channels") == csv_schema["channels_including_target"]
        )
        expected_channels = contract.get("channels_including_target")
        if expected_channels is not None and report_schema.get("expected_channels") not in (None, expected_channels):
            errors.append("quality_report_expected_channels_mismatch")
    else:
        errors.append("quality_report_schema_missing")
    report_info["schema_match"] = schema_match
    if not schema_match:
        errors.append("quality_report_schema_mismatch")

    report_time = report.get("time")
    if isinstance(report_time, dict):
        expected_frequency = contract.get("frequency")
        if expected_frequency is not None and report_time.get("expected_frequency") != expected_frequency:
            errors.append("quality_report_frequency_mismatch")
    return report_info, errors


def validate_dataset_contract(dataset: str, contract: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    """Validate one ``dataset_contract`` entry and return JSON-safe evidence."""

    errors: list[str] = []
    if not isinstance(contract, dict):
        return {
            "dataset": dataset,
            "valid": False,
            "errors": ["contract_entry_not_object"],
            "contract": {"entry_type": type(contract).__name__},
        }
    result: dict[str, Any] = {
        "dataset": dataset,
        "valid": False,
        "errors": errors,
        "contract": {
            "relative_path": contract.get("relative_path"),
            "quality_report": contract.get("quality_report"),
            "channels_including_target": contract.get("channels_including_target"),
            "frequency": contract.get("frequency"),
            "time_column": contract.get("time_column"),
            "target": contract.get("target"),
        },
    }
    required = ("relative_path", "channels_including_target", "frequency", "time_column", "target", "sha256", "quality_report")
    for key in required:
        if key not in contract:
            errors.append(f"contract_{key}_missing")
    if errors:
        return result

    try:
        csv_path = _resolve_path(contract["relative_path"], repo_root)
    except (TypeError, ValueError, OSError) as exc:
        errors.append(f"csv_path_invalid:{exc}")
        return result
    csv_info: dict[str, Any] = {
        "path": str(csv_path),
        "exists": csv_path.is_file(),
        "bytes": None,
        "sha256": None,
        "hash": None,
        "schema": None,
    }
    result["csv"] = csv_info
    if not csv_path.is_file():
        errors.append("csv_missing")
        return result
    try:
        data_bytes = int(csv_path.stat().st_size)
        data_sha256 = _sha256(csv_path)
    except OSError as exc:
        errors.append(f"csv_unreadable:{exc}")
        return result
    csv_info["bytes"] = data_bytes
    csv_info["sha256"] = data_sha256
    hash_check, hash_errors = _hash_check(contract.get("sha256"), data_sha256, "csv_sha256")
    csv_info["hash"] = hash_check
    errors.extend(hash_errors)

    csv_schema, schema_errors = _scan_csv(
        csv_path,
        str(contract.get("time_column")),
        str(contract.get("target")),
        contract.get("channels_including_target"),
    )
    csv_info["schema"] = csv_schema
    errors.extend(schema_errors)

    try:
        report_path = _resolve_path(contract["quality_report"], repo_root)
    except (TypeError, ValueError, OSError) as exc:
        errors.append(f"quality_report_path_invalid:{exc}")
    else:
        report_info, report_errors = _quality_report_checks(
            report_path, dataset, contract, csv_schema, data_sha256, data_bytes
        )
        result["quality_report"] = report_info
        errors.extend(report_errors)

    result["valid"] = not errors
    return result


def validate_plan(plan_path: Path, repo_root: Path, datasets: Iterable[str] | None = None) -> dict[str, Any]:
    """Validate selected entries from a plan and return a complete JSON report."""

    errors: list[str] = []
    plan_path = plan_path.resolve()
    repo_root = repo_root.resolve()
    output: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "plan": {"path": str(plan_path), "sha256": None},
        "repo_root": str(repo_root),
        "valid": False,
        "datasets": {},
        "errors": errors,
    }
    if not plan_path.is_file():
        errors.append("plan_missing")
        return output
    try:
        output["plan"]["sha256"] = _sha256(plan_path)
        with plan_path.open("r", encoding="utf-8") as handle:
            plan = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"plan_invalid:{exc}")
        return output
    if not isinstance(plan, dict):
        errors.append("plan_not_object")
        return output
    contract = plan.get("dataset_contract")
    if not isinstance(contract, dict) or not contract:
        errors.append("dataset_contract_missing_or_empty")
        return output

    requested = list(datasets) if datasets is not None else list(contract)
    unknown = [name for name in requested if name not in contract]
    for name in unknown:
        errors.append(f"unknown_dataset:{name}")
    for name in requested:
        if name in contract:
            result = validate_dataset_contract(name, contract[name], repo_root)
            output["datasets"][name] = result
            errors.extend(f"{name}:{error}" for error in result.get("errors", []))
    output["valid"] = not errors
    output["summary"] = {
        "requested_datasets": requested,
        "dataset_count": len(output["datasets"]),
        "valid_dataset_count": sum(1 for item in output["datasets"].values() if item.get("valid")),
        "error_count": len(errors),
    }
    return output


def _parse_report_hash_overrides(values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected DATASET=SHA256, got {value!r}")
        dataset, digest = value.split("=", 1)
        if not dataset or not digest:
            raise ValueError(f"expected DATASET=SHA256, got {value!r}")
        overrides[dataset] = digest.lower()
    return overrides


def _apply_report_hash_overrides(result: dict[str, Any], overrides: dict[str, str]) -> None:
    """Apply CLI-only expected report hashes to the in-memory result."""

    for dataset, expected in overrides.items():
        item = result.get("datasets", {}).get(dataset)
        if not item:
            result.setdefault("errors", []).append(f"unknown_dataset_hash_override:{dataset}")
            continue
        report = item.get("quality_report")
        if not isinstance(report, dict):
            item.setdefault("errors", []).append("quality_report_hash_override_unusable")
            continue
        actual = report.get("sha256")
        check, check_errors = _hash_check(expected, actual or "", "quality_report_file_sha256")
        check["field"] = "cli_expected_report_sha256"
        report["hash"] = check
        item.setdefault("errors", []).extend(check_errors)
    # Recompute aggregate state after applying overrides.
    errors = result.setdefault("errors", [])
    for dataset, item in result.get("datasets", {}).items():
        item["valid"] = not item.get("errors")
    dataset_names = set(result.get("datasets", {}))
    # ``validate_plan`` already flattened dataset errors.  Rebuild that
    # portion after applying overrides, while retaining plan-level errors.
    errors[:] = [
        error
        for error in errors
        if not any(error.startswith(f"{name}:") for name in dataset_names)
        and not error.startswith("unknown_dataset_hash_override:")
    ]
    for dataset in overrides:
        if dataset not in dataset_names:
            errors.append(f"unknown_dataset_hash_override:{dataset}")
    for dataset, item in result.get("datasets", {}).items():
        errors.extend(f"{dataset}:{error}" for error in item.get("errors", []))
    result["valid"] = not errors
    if isinstance(result.get("summary"), dict):
        result["summary"]["valid_dataset_count"] = sum(
            1 for item in result.get("datasets", {}).values() if item.get("valid")
        )
        result["summary"]["error_count"] = len(errors)


def _write_json(payload: dict[str, Any], output_path: Path | None) -> None:
    encoded = json.dumps(payload, indent=2, ensure_ascii=True) + "\n"
    if output_path is not None:
        if str(output_path) == "-":
            print(encoded, end="")
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--dataset", action="append", dest="datasets", help="validate only this dataset (repeatable)")
    parser.add_argument(
        "--expected-report-sha256",
        action="append",
        default=[],
        metavar="DATASET=SHA256",
        help="pin a quality-report file digest without editing the plan (repeatable)",
    )
    parser.add_argument("--output", type=Path, default=None, help="write the JSON report; omit to print it")
    parser.add_argument("--strict", action="store_true", help="exit 2 when any contract check fails")
    args = parser.parse_args(argv)

    plan_path = args.plan if args.plan.is_absolute() else args.repo_root / args.plan
    try:
        overrides = _parse_report_hash_overrides(args.expected_report_sha256)
        result = validate_plan(plan_path, args.repo_root, args.datasets)
        _apply_report_hash_overrides(result, overrides)
    except (OSError, TypeError, ValueError) as exc:
        result = {
            "schema_version": 1,
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
            "plan": {"path": str(plan_path.resolve()), "sha256": None},
            "repo_root": str(args.repo_root.resolve()),
            "valid": False,
            "datasets": {},
            "errors": [f"validator_error:{exc}"],
        }
    _write_json(result, args.output)
    if args.strict and not result.get("valid", False):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
