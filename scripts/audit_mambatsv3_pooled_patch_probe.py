#!/usr/bin/env python3
"""Read the validation-only pooled-patch representation probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MATRICES = {"electricity": 5750, "traffic": 5760}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = _load(args.root / "plan.json")
    if not plan.get("validation_only") or plan.get("test_metrics_produced"):
        raise ValueError("probe must be validation-only")
    rows = []
    for target, seed in MATRICES.items():
        active = _load(args.root / "{}_s{}_pooled_patch_active_hybrid.json".format(target, seed))
        null = _load(args.root / "{}_s{}_pooled_patch_null_hybrid.json".format(target, seed))
        pretrain = _load(args.root / "{}_s{}_pooled_patch_active_pretrain.json".format(target, seed))
        if any(payload.get("test_metrics_produced") for payload in (active, null, pretrain)):
            raise ValueError("probe opened test metrics")
        if pretrain["config"].get("relation_view") != "pooled_patch_contrastive":
            raise ValueError("wrong relation view")
        if "pretrain_pooled_patch_projector" not in pretrain:
            raise ValueError("pooled patch projector manifest is missing")
        value, reference = float(active["best_validation_mse"]), float(null["best_validation_mse"])
        rows.append({"target": target, "seed": seed, "active": value, "exact_null": reference,
                     "active_vs_null_percent": (value / reference - 1.0) * 100.0,
                     "active_better": value < reference})
    result = {"validation_only": True, "test_metrics_produced": False, "rows": rows,
              "all_targets_improved": bool(all(row["active_better"] for row in rows)),
              "next_step": "No test protocol is opened by this probe."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
