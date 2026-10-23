"""Measure the scalar-transfer overhead of ACF telemetry snapshots.

This is an engineering micro-benchmark only.  It does not run a forecasting
model and must not be used as an end-to-end training-speed claim.
"""

import argparse
import json
import pathlib
import sys
import time
from types import SimpleNamespace

import torch

# Allow the script to be launched directly from the repository root or from
# another working directory without requiring an external PYTHONPATH.
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.exp_long_term_forecasting_with_AutoCon import (
    _read_detached_scalar_fields,
)


def _old_snapshot(source, specs):
    return {
        key: float(getattr(source, attribute).detach().cpu())
        for key, attribute in specs
    }


def _measure(callback, iterations):
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        callback()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0 / iterations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print(json.dumps({"status": "cuda_unavailable"}, sort_keys=True))
        return 0
    if args.fields <= 0 or args.warmup < 0 or args.iterations <= 0:
        parser.error("fields must be positive; warmup must be non-negative; iterations must be positive")

    names = ["field_{}".format(index) for index in range(args.fields)]
    source = SimpleNamespace(
        **{name: torch.rand((), device="cuda") for name in names}
    )
    specs = [("key_{}".format(index), name) for index, name in enumerate(names)]
    old = lambda: _old_snapshot(source, specs)
    batched = lambda: _read_detached_scalar_fields(source, specs)
    for _ in range(args.warmup):
        old()
        batched()
    old_ms = _measure(old, args.iterations)
    batched_ms = _measure(batched, args.iterations)
    print(json.dumps({
        "status": "ok",
        "fields": args.fields,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "old_ms_per_snapshot": old_ms,
        "batched_ms_per_snapshot": batched_ms,
        "batched_over_old_ratio": batched_ms / old_ms if old_ms else None,
        "scope": "scalar telemetry transfer only",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
