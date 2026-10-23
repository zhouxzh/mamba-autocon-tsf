#!/usr/bin/env python
"""Audit dense Solar/PEMS arrays through the repository's real loader.

The audit records source hashes, raw and selected shapes, finite-value checks,
and the split/window contract used by ``Dataset_Array``.  It intentionally
does not infer a graph or fabricate timestamps: the current forecasting
experiments use only the selected dense feature matrix and regular synthetic
calendar marks.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_provider.data_loader import Dataset_Array, inspect_numeric_array_file
from scripts.run_training_plan import DATASETS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _raw_schema(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    data_format = str(spec.get("format", "auto"))
    data_key = spec.get("data_key", "data") or "data"
    if data_format.lower() == "npz":
        with np.load(path, allow_pickle=False) as archive:
            keys = sorted(str(key) for key in archive.files)
            raw = np.asarray(archive[data_key])
    else:
        keys = []
        raw = np.loadtxt(path, dtype=np.float64, ndmin=2)
    return {
        "keys": keys,
        "shape": [int(item) for item in raw.shape],
        "dtype": str(raw.dtype),
        "finite": bool(np.all(np.isfinite(raw))),
        "min": float(np.min(raw)),
        "max": float(np.max(raw)),
        "mean": float(np.mean(raw)),
        "std": float(np.std(raw)),
    }


def audit_dataset(name: str, seq_len: int, label_len: int, pred_len: int) -> dict[str, Any]:
    if name not in DATASETS:
        raise ValueError("unknown dataset {!r}; choose from {}".format(name, sorted(DATASETS)))
    spec = DATASETS[name]
    path = (REPO_ROOT / spec["root"] / spec["path"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    raw_schema = _raw_schema(path, spec)
    selected_schema = inspect_numeric_array_file(
        path,
        data_format=spec.get("format", "auto"),
        data_key=spec.get("data_key", "data") or "data",
        data_feature_index=spec.get("data_feature_index", 0),
    )
    split_rows: dict[str, int] = {}
    sample_shapes: dict[str, list[list[int]]] = {}
    for flag in ("train", "val", "test"):
        dataset = Dataset_Array(
            root_path=str(path.parent),
            data_path=path.name,
            flag=flag,
            size=[seq_len, label_len, pred_len],
            features="M",
            target=spec.get("target", "OT"),
            scale=True,
            timeenc=0,
            freq=spec["freq"],
            data_format=spec.get("format", "auto"),
            data_key=spec.get("data_key", "data") or "data",
            data_feature_index=spec.get("data_feature_index", 0),
            target_index=spec.get("target_index", -1),
            data_interval_minutes=spec.get("data_interval_minutes"),
        )
        split_rows[flag] = len(dataset)
        _, x, y, x_mark, y_mark = dataset[0]
        sample_shapes[flag] = [
            [int(item) for item in x.shape],
            [int(item) for item in y.shape],
            [int(item) for item in x_mark.shape],
            [int(item) for item in y_mark.shape],
        ]
    raw_shape = raw_schema["shape"]
    return _json_value({
        "dataset": name,
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
        "source_hint": spec.get("source_hint"),
        "declared": {
            "freq": spec["freq"],
            "interval_minutes": spec.get("data_interval_minutes"),
            "data_format": spec.get("format", "auto"),
            "data_key": spec.get("data_key", "data"),
            "data_feature_index": spec.get("data_feature_index", 0),
            "target_index": spec.get("target_index", -1),
            "declared_channels": spec.get("channels"),
            "graph_adjacency_used": False,
        },
        "raw": raw_schema,
        "selected": selected_schema,
        "windows": {
            "size": [seq_len, label_len, pred_len],
            "split_window_counts": split_rows,
            "sample_shapes": sample_shapes,
            "finite_loader_sample": True,
        },
        "quality_pass": bool(
            raw_schema["finite"]
            and selected_schema["shape"][1] == spec.get("channels")
            and all(value > 0 for value in split_rows.values())
        ),
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["pems03", "pems04", "pems07", "pems08"])
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--label-len", type=int, default=48)
    parser.add_argument("--pred-len", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.seq_len, args.label_len, args.pred_len) <= 0:
        raise SystemExit("sequence lengths must be positive")
    try:
        rows = [audit_dataset(name, args.seq_len, args.label_len, args.pred_len) for name in args.datasets]
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit("array audit failed: {}".format(exc)) from exc
    result = {
        "schema_version": 1,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "window_contract": [args.seq_len, args.label_len, args.pred_len],
        "datasets": rows,
        "all_quality_pass": all(row["quality_pass"] for row in rows),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "datasets": args.datasets,
        "all_quality_pass": result["all_quality_pass"],
        "output": str(output),
    }, indent=2))
    return 0 if result["all_quality_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
