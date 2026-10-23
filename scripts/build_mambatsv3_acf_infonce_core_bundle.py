#!/usr/bin/env python3
"""Build a read-only index over the audited ACF-InfoNCE evidence bundle."""

from __future__ import annotations

import json
from pathlib import Path


OUTPUT = Path("artifacts/paper_final/mambatsv3_acf_infonce_core_bundle_20261003")
RELEASES = {
    "pems_bay_h192_h336": Path("artifacts/paper_final/mambatsv3_acf_infonce_horizon_confirmatory_20261003/audit.json"),
    "metr_la_h192": Path("artifacts/paper_final/mambatsv3_acf_infonce_metr_la_h192_confirmatory_20261003/audit.json"),
    "weather_h192": Path("artifacts/paper_final/mambatsv3_acf_infonce_weather_h192_confirmatory_20261003/audit.json"),
}
FULL_PREFIX = Path("artifacts/exploratory/mambatsv3_acf_infonce_h192_full_validation_20261003")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def full_prefix_summary():
    seeds = (5410, 5411, 5412)
    active, null = [], []
    for seed in seeds:
        active.append(load(FULL_PREFIX / ("pems_bay_h192_s{}_active_hybrid.json".format(seed)))["best_validation_mse"])
        null.append(load(FULL_PREFIX / ("pems_bay_h192_s{}_null_hybrid.json".format(seed)))["best_validation_mse"])
    active_mean = sum(active) / len(active)
    null_mean = sum(null) / len(null)
    return {
        "target": "PEMS-BAY", "horizon": 192, "target_prefix_fraction": 1.0,
        "max_train_windows": 128, "active_mean_mse": active_mean,
        "null_mean_mse": null_mean,
        "active_vs_null_percent": (active_mean / null_mean - 1.0) * 100.0,
        "paired_wins": sum(left < right for left, right in zip(active, null)),
        "test_metrics_produced": False,
        "interpretation": "full target-prefix validation reference, not exhaustive convergence training",
    }


def main() -> int:
    release_payloads = {}
    for name, path in RELEASES.items():
        payload = load(path)
        if payload.get("audit_valid") is not True or payload.get("test_metrics_produced") is not True:
            raise ValueError("invalid audited release: {}".format(path))
        release_payloads[name] = {"path": str(path), "summary": payload["summary"]}
    full = full_prefix_summary()
    if full["test_metrics_produced"]:
        raise ValueError("full-prefix validation unexpectedly opened test")
    result = {
        "bundle_version": 1,
        "method": "shared channel-agnostic MambaTSv3 residual pretraining with signed forecast-level ACF InfoNCE",
        "inference_contract": {
            "mamba_scan_count": 1,
            "relation_loss_included": False,
            "autocon_pairing_included": False,
        },
        "audited_releases": release_payloads,
        "full_target_prefix_validation_reference": full,
        "negative_boundaries": {
            "electricity_h96": "legacy active/null +0.316%, 0/3; round-robin coverage repair +0.548%, 0/3",
            "traffic_h96_25_percent": "active/null +0.216%, 0/3 wins",
            "weather_h192_25_percent": "active/null -0.108%, 1/3 wins",
            "forecast_affine_loss": "stopped after active/null failures",
            "acf_source_ranking_h192": "selected top-3 +2.032% vs heterogeneous counterfactual, 0/3",
        },
        "post_release_extensions": {
            "pems_bay_timesnet_timemixer_rlinear": "descriptive only; excluded from frozen primary significance claims",
            "pems_bay_official_patchtst_h192_h336": "external-model-code adapter extension; excluded from frozen primary significance claims",
            "pems_bay_official_itransformer_h192_h336": "external-model-code adapter extension; excluded from frozen primary significance claims",
            "pems_bay_official_timesnet_h192_h336": "external-model-code adapter extension; excluded from frozen primary significance claims",
            "pems_bay_official_timemixer_h192_h336": "external-model-code adapter extension; excluded from frozen primary significance claims",
        },
        "publication_scope": "ACF-compatible low-data periodic traffic and sensor forecasting; not universal multivariate forecasting",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "bundle.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# ACF-InfoNCE Core Evidence Bundle", "", "This index is read-only. It binds the three audited held-out releases and explicitly retains their limits.", "", "| Release | Scope | Audit |", "| --- | --- | --- |"]
    for name, value in release_payloads.items():
        lines.append("| {} | audited held-out | `{}` |".format(name, value["path"]))
    lines += ["", "## Full-Prefix Validation Reference", "", "PEMS-BAY H192 uses 100% target training prefix with 128 sampled windows: active vs null {:+.3f}%, {}/3 wins. This is not exhaustive full-data convergence training.".format(full["active_vs_null_percent"], full["paired_wins"]), "", "## Mandatory Limits", "", "- Electricity and higher-coverage Traffic do not support a universal positive claim.", "- Weather H192 has held-out accuracy but unstable 25% mechanism evidence.", "- PEMS-BAY TimesNet/TimeMixer/RLinear comparisons are post-release descriptive extensions.", "- Official PatchTST H192/H336 use pinned external model code with a local chronological adapter and are post-release.", "- Official iTransformer H192/H336 use pinned external model code with a local chronological adapter and are post-release.", "- Official TimesNet H192/H336 use pinned external model code with a local chronological adapter and are post-release.", "- Official TimeMixer H192/H336 use pinned external model code with a local chronological adapter and are post-release.", "- RLinear is faster than the candidate; speed claims are limited to nonlinear measured baselines."]
    (OUTPUT / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
