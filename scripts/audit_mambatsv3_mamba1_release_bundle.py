#!/usr/bin/env python3
"""Compose the bounded Mamba-1 ACF release evidence into one audited readout."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "artifacts/paper_final/mambatsv3_mamba1_acf_frozen_test_release_20261006"
PEMS10_RELEASE = ROOT / "artifacts/paper_final/mambatsv3_mamba1_pems10_frozen_test_release_20261006"
METR10_RELEASE = ROOT / "artifacts/paper_final/mambatsv3_mamba1_metr10_frozen_test_release_20261006"
ELECTRICITY_STRONG = ROOT / "artifacts/exploratory/mambatsv3_electricity_mamba1_strong_calibration_relation_probe_20261006"
TARGETS = ("pems_bay", "metr_la", "weather")
HIGH_DIM = {
    "electricity_source_only": ROOT / "artifacts/exploratory/mambatsv3_electricity_mamba1_overlap_probe_20261006",
    "traffic_source_only": ROOT / "artifacts/exploratory/mambatsv3_traffic_mamba1_overlap_probe_20261006",
    "electricity_target_calibration": ROOT / "artifacts/exploratory/mambatsv3_electricity_mamba1_target_calibration_probe_20261006",
    "electricity_channel_adapter": ROOT / "artifacts/exploratory/mambatsv3_electricity_mamba1_adapter_probe_20261006",
    "electricity_target_mlp_adapter": ROOT / "artifacts/exploratory/mambatsv3_electricity_mamba1_target_mlp_probe_20261006",
    "electricity_channelwise_representation_fusion": ROOT / "artifacts/exploratory/mambatsv3_electricity_mamba1_channelwise_fusion_probe_20261006",
    "electricity_shared_summary_transfer": ROOT / "artifacts/exploratory/mambatsv3_electricity_mamba1_shared_summary_transfer_probe_20261006",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _highdim_value(root: Path, active_name: str, null_name: str) -> dict:
    active = _load(root / active_name)
    null = _load(root / null_name)
    if active.get("test_metrics_produced") or null.get("test_metrics_produced"):
        raise ValueError(f"high-dimensional probe opened test: {root}")
    value, reference = float(active["best_validation_mse"]), float(null["best_validation_mse"])
    return {"active_validation_mse": value, "null_validation_mse": reference,
            "active_vs_null_percent": float((value / reference - 1) * 100),
            "active_better": bool(value < reference), "test_metrics_produced": False}


def main() -> int:
    patch = _load(RELEASE / "audit.json")
    null = _load(RELEASE / "paired_null_readout/audit.json")
    itransformer = _load(RELEASE / "official_itransformer_extension/audit.json")
    timemixer = _load(RELEASE / "official_timemixer_extension/audit.json")
    pems10 = _load(PEMS10_RELEASE / "audit.json")
    metr10 = _load(METR10_RELEASE / "audit.json")
    electricity_strong = _load(ELECTRICITY_STRONG / "audit.json")
    electricity_robust = _load(ELECTRICITY_STRONG / "robustness_validation/audit.json")
    robust = _load(ROOT / "artifacts/exploratory/mambatsv3_mamba1_crossdomain_replication_20261006/robustness_validation/audit.json")
    inference = _load(RELEASE / "inference_comparison_pems_bay/audit.json")
    if not (patch.get("audit_valid") and null.get("audit_valid") and itransformer.get("audit_valid") and timemixer.get("audit_valid")):
        raise ValueError("a required release audit is invalid")
    if not (pems10.get("audit_valid") and metr10.get("audit_valid")) or electricity_strong.get("test_metrics_produced"):
        raise ValueError("PEMS10 or Electricity extension contract failure")
    if not (inference["mamba1"]["mamba_scan_count"] == 1
            and not inference["mamba1"]["relation_loss_included"]
            and not inference["mamba1"]["autocon_representation_returned"]):
        raise ValueError("inference contract is not one scan and loss-free")
    for target in TARGETS:
        if not (patch["summary"][target]["active_wins"] == 3
                and null["summary"][target]["active_wins"] == 3
                and itransformer["summary"][target]["active_wins"] == 3
                and timemixer["summary"][target]["noninferior_at_one_se"]):
            raise ValueError(f"expected held-out paired wins missing for {target}")
        for condition, value in robust["summary"][target].items():
            if value["active_wins"] != 3 or not value["noninferior_at_one_se"]:
                raise ValueError(f"robustness failure: {target}/{condition}")
    if not (pems10["active_wins"] == 3 and pems10["noninferior_at_one_se"]):
        raise ValueError("PEMS-BAY 10% frozen test extension did not pass")
    if not (metr10["active_wins"] == 3 and metr10["noninferior_at_one_se"]):
        raise ValueError("METR-LA 10% frozen test extension did not pass")
    for condition, value in electricity_robust["summary"].items():
        if value["active_wins"] != 3 or not value["noninferior_at_one_se"]:
            raise ValueError(f"Electricity strong-calibration stress failure: {condition}")
    highdim = {
        "electricity_source_only": _highdim_value(HIGH_DIM["electricity_source_only"], "s5940_active_frozen_core.json", "s5940_null_frozen_core.json"),
        "traffic_source_only": _highdim_value(HIGH_DIM["traffic_source_only"], "s5941_active_frozen_core.json", "s5941_null_frozen_core.json"),
        "electricity_target_calibration": _highdim_value(HIGH_DIM["electricity_target_calibration"], "s5942_active_frozen_core.json", "s5942_null_frozen_core.json"),
        "electricity_channel_adapter": _highdim_value(HIGH_DIM["electricity_channel_adapter"], "s5943_active_adapter.json", "s5943_null_adapter.json"),
        "electricity_target_mlp_adapter": _highdim_value(HIGH_DIM["electricity_target_mlp_adapter"], "s5944_active_target_mlp.json", "s5944_null_target_mlp.json"),
        "electricity_channelwise_representation_fusion": _highdim_value(HIGH_DIM["electricity_channelwise_representation_fusion"], "s5945_active_channelwise_fusion.json", "s5945_null_channelwise_fusion.json"),
        "electricity_shared_summary_transfer": _highdim_value(HIGH_DIM["electricity_shared_summary_transfer"], "s5947_active_shared_summary.json", "s5947_null_shared_summary.json"),
    }
    transferred = _load(HIGH_DIM["electricity_shared_summary_transfer"] / "s5947_active_shared_summary.json")
    expected_summary_keys = {
        "ts_shared_channel_summary_projection.0.weight",
        "ts_shared_channel_summary_projection.2.weight",
        "ts_shared_channel_summary_scale",
    }
    if not expected_summary_keys.issubset(set(transferred["transfer"]["loaded_keys"])):
        raise ValueError("shared channel summary was not transferred")
    if not all(row["mamba_core_frozen"] for row in transferred["history"]):
        raise ValueError("shared channel summary was not frozen with target core")
    highdim["electricity_shared_summary_transfer"]["transferred_and_frozen"] = True
    if any(value["active_better"] for value in highdim.values()):
        raise ValueError("high-dimensional negative boundary unexpectedly passed")
    result = {
        "schema_version": 1,
        "release_scope": "three compatible target domains; low-label chronological transfer",
        "patchtst_test": patch["summary"],
        "paired_null_test": null["summary"],
        "itransformer_test_extension": itransformer["summary"],
        "timemixer_test_extension": timemixer["summary"],
        "pems_bay_10pct_patchtst_test_extension": {
            key: pems10[key] for key in (
                "active_mean_test_mse", "official_patchtst_mean_test_mse",
                "active_vs_patchtst_percent", "active_wins", "ratio_upper_1se",
                "noninferior_at_one_se",
            )
        },
        "metr_la_10pct_patchtst_test_extension": {
            key: metr10[key] for key in (
                "active_mean_test_mse", "official_patchtst_mean_test_mse",
                "active_vs_patchtst_percent", "active_wins", "ratio_upper_1se",
                "noninferior_at_one_se",
            )
        },
        "electricity_strong_calibration_validation": {
            "causal": {
                key: electricity_strong[key] for key in (
                    "active_mean_validation_mse", "null_mean_validation_mse",
                    "active_vs_null_percent", "active_wins", "ratio_upper_1se",
                    "noninferior_at_one_se",
                )
            },
            "robustness": electricity_robust["summary"],
            "external_patchtst_validation_boundary": "official PatchTST validation mean is lower than the strong-calibration candidate",
        },
        "robustness_validation": robust["summary"],
        "inference_comparison": inference,
        "highdim_negative_boundary": highdim,
        "historical_public_test_access_caveat": bool(patch.get("historical_public_test_access_caveat")),
        "audit_valid": True,
        "submission_ready": False,
        "limitations": [
            "historical prior public-test access prevents a pristine first-open test claim",
            "Traffic remains a negative active/null boundary; Electricity is positive only under a stronger target-calibration relation schedule and still loses the matched PatchTST validation comparator",
            "candidate is slower than named PatchTST and iTransformer microbenchmarks",
            "iTransformer is a post-release external baseline extension",
        ],
    }
    (RELEASE / "bundle_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# Mamba-1 ACF Release Bundle", "", "Audit: valid; scope remains bounded.", "",
             "| Target | vs null | vs PatchTST | vs iTransformer | vs TimeMixer |", "| --- | ---: | ---: | ---: | ---: |"]
    for target in TARGETS:
        lines.append("| {} | {:+.3f}% | {:+.3f}% | {:+.3f}% | {:+.3f}% |".format(
            target, null["summary"][target]["active_vs_null_percent"],
            patch["summary"][target]["active_vs_patchtst_percent"],
            itransformer["summary"][target]["active_vs_itransformer_percent"],
            timemixer["summary"][target]["active_vs_timemixer_percent"],
        ))
    lines += ["", "High-dimensional Electricity/Traffic probes remain negative; see `bundle_audit.json`."]
    (RELEASE / "bundle_readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
