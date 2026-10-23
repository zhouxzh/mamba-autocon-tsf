#!/usr/bin/env python3
"""Convert official DCRNN adjacency pickles into auditable safe NPZ contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "metr_la": {
        "csv": "dataset/METR-LA/METR-LA.csv",
        "pickle": "dataset/METR-LA/adj_mx.pkl",
        "output": "dataset/METR-LA/adjacency.npz",
        "contract": "dataset/METR-LA/adjacency_contract.json",
    },
    "pems_bay": {
        "csv": "dataset/PEMS-BAY/PEMS-BAY.csv",
        "pickle": "dataset/PEMS-BAY/adj_mx_bay.pkl",
        "output": "dataset/PEMS-BAY/adjacency.npz",
        "contract": "dataset/PEMS-BAY/adjacency_contract.json",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_official_adjacency(path: Path):
    with path.open("rb") as handle:
        sensor_ids, _, adjacency = pickle.load(handle, encoding="latin1")
    ids = [str(value) for value in sensor_ids]
    adjacency = np.asarray(adjacency, dtype=np.float32)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("official adjacency must be square")
    if adjacency.shape[0] != len(ids) or len(set(ids)) != len(ids):
        raise ValueError("official adjacency IDs do not form a unique square graph")
    if not np.isfinite(adjacency).all() or np.any(adjacency < 0):
        raise ValueError("official adjacency must be finite and non-negative")
    return ids, adjacency


def build(dataset: str, *, overwrite: bool) -> dict:
    spec = DATASETS[dataset]
    csv_path = REPO_ROOT / spec["csv"]
    pickle_path = REPO_ROOT / spec["pickle"]
    output_path = REPO_ROOT / spec["output"]
    contract_path = REPO_ROOT / spec["contract"]
    if not csv_path.is_file() or not pickle_path.is_file():
        raise FileNotFoundError("traffic CSV or official adjacency pickle is missing")
    csv_ids = pd.read_csv(csv_path, nrows=1).columns[1:].astype(str).tolist()
    sensor_ids, adjacency = _load_official_adjacency(pickle_path)
    if csv_ids != sensor_ids:
        raise ValueError("CSV sensor-column order does not match official graph IDs")
    row_sum = adjacency.sum(axis=1, keepdims=True)
    normalized = adjacency / np.maximum(row_sum, np.float32(1e-12))
    target, source = np.nonzero(normalized)
    weights = normalized[target, source].astype(np.float32, copy=False)
    if output_path.exists() and not overwrite:
        raise FileExistsError("output already exists: {}".format(output_path))
    np.savez_compressed(
        output_path,
        schema_version=np.asarray(1, dtype=np.int64),
        sensor_ids=np.asarray(sensor_ids, dtype="U"),
        target=np.asarray(target, dtype=np.int64),
        source=np.asarray(source, dtype=np.int64),
        weight=weights,
        node_count=np.asarray(len(sensor_ids), dtype=np.int64),
    )
    payload = {
        "schema_version": 1,
        "dataset": dataset,
        "source_pickle": spec["pickle"],
        "source_pickle_sha256": _sha256(pickle_path),
        "speed_csv": spec["csv"],
        "speed_csv_sha256": _sha256(csv_path),
        "adjacency_npz": spec["output"],
        "adjacency_npz_sha256": _sha256(output_path),
        "node_count": len(sensor_ids),
        "directed_edge_count": int(weights.size),
        "csv_sensor_order_matches_official_graph": True,
        "normalization": "row_normalized_official_directed_adjacency",
    }
    contract_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), action="append")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for dataset in args.dataset or sorted(DATASETS):
        print(json.dumps(build(dataset, overwrite=args.overwrite), ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
