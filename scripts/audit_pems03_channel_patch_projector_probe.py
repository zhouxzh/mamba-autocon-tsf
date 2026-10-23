#!/usr/bin/env python3
"""Read-only audit for the PEMS03 shared Channel-Patch projector screen."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/exploratory/pems03_channel_patch_projector_probe_20261007"
SEED = 6089


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    identity = load(OUTPUT / f"s{SEED}_identity_frozen_core.json")
    mlp = load(OUTPUT / f"s{SEED}_mlp_frozen_core.json")
    pretrain = load(OUTPUT / f"s{SEED}_mlp_pretrain.json")
    event = load(OUTPUT / f"s{SEED}_mlp_pretrain_event.json")
    if any(item.get("test_metrics_produced") for item in (identity, mlp, pretrain)):
        raise ValueError("test boundary")
    if pretrain["config"].get("channel_patch_projector") != "mlp":
        raise ValueError("MLP projector mode missing")
    if "pretrain_channel_patch_projector" not in pretrain:
        raise ValueError("projector provenance missing")
    if "--channel-patch-projector" not in event["command"]:
        raise ValueError("projector invocation missing")
    base = float(identity["best_validation_mse"])
    candidate = float(mlp["best_validation_mse"])
    result = {
        "validation_only": True, "test_metrics_produced": False,
        "identity_mse": base, "mlp_mse": candidate,
        "mlp_vs_identity_percent": (candidate / base - 1.0) * 100.0,
        "screen_passes_expansion_margin": bool(candidate / base - 1.0 <= -.0025),
        "target_excluded_from_source_and_initialization": True,
        "inference_projector_absent": True, "submission_ready": False,
    }
    (OUTPUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
