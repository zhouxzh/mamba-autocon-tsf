#!/usr/bin/env python3
"""Conservative end-to-end evidence audit for channel-patch ACF SSL."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/paper_final/channel_patch_mechanism_evidence_20261006"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    low10 = load(ROOT / "artifacts/exploratory/mambats_channel_patch_replication_20261006/audit.json")
    robust10 = load(ROOT / "artifacts/exploratory/mambats_channel_patch_replication_20261006/robustness_validation/audit.json")
    weather = load(ROOT / "artifacts/exploratory/weather_channel_patch_replication_20261006/audit.json")
    weather_robust = load(ROOT / "artifacts/exploratory/weather_channel_patch_replication_20261006/robustness_validation/audit.json")
    low25 = load(ROOT / "artifacts/exploratory/channel_patch_25pct_replication_20261006/audit.json")
    robust25 = load(ROOT / "artifacts/exploratory/channel_patch_25pct_replication_20261006/robustness_validation/audit.json")
    h336 = load(ROOT / "artifacts/exploratory/pems_h336_channel_patch_replication_20261006/audit.json")
    h336_robust = load(ROOT / "artifacts/exploratory/pems_h336_channel_patch_replication_20261006/robustness_validation/audit.json")
    test = load(ROOT / "artifacts/paper_final/channel_patch_frozen_test_extension_20261006/audit.json")
    electricity = load(ROOT / "artifacts/exploratory/electricity_channel_patch_strong_calibration_replication_20261006/external_patchtst_audit.json")
    d32 = load(ROOT / "artifacts/exploratory/electricity_d32_channel_patch_replication_20261006/audit.json")
    channel_gate = load(ROOT / "artifacts/exploratory/electricity_d32_channelwise_gate_expansion_20261006/five_seed_audit.json")
    clean = load(ROOT / "artifacts/reproducibility/channel_patch_clean_interpreter_validation_20261006/audit.json")
    independent = load(ROOT / "artifacts/reproducibility/pems03_channel_patch_independent_machine_20261007/independent_audit.json")
    pems_gate = load(ROOT / "artifacts/exploratory/pems_channelwise_residual_gate_probe_20261006/audit.json")
    traffic_gate = load(ROOT / "artifacts/exploratory/traffic_d32_channelwise_residual_gate_probe_20261006/audit.json")
    inference = load(ROOT / "artifacts/exploratory/channel_patch_25pct_replication_20261006/inference_benchmark/mamba.json")
    geometry = load(ROOT / "artifacts/exploratory/channel_patch_representation_geometry_20261007/audit.json")
    pems03 = load(ROOT / "artifacts/exploratory/pems03_channel_patch_external_replication_rerun_20261007/audit.json")
    pems03_robust = load(ROOT / "artifacts/exploratory/pems03_channel_patch_external_replication_rerun_20261007/robustness_validation/audit.json")
    pems03_25 = load(ROOT / "artifacts/exploratory/pems03_channel_patch_25pct_external_replication_20261007/audit.json")
    pems03_25_robust = load(ROOT / "artifacts/exploratory/pems03_channel_patch_25pct_external_replication_20261007/robustness_validation/audit.json")
    pems03_inference = load(ROOT / "artifacts/exploratory/pems03_channel_patch_25pct_external_replication_20261007/inference_benchmark/audit.json")
    pems03_itransformer = load(ROOT / "artifacts/exploratory/official_itransformer_pems03_channel_patch_25pct_20261007/audit.json")
    pems03_timemixer = load(ROOT / "artifacts/exploratory/pems03_channel_patch_timemixer_precision_expansion_20261007/audit.json")
    pems03_timemixer_update_matched = load(ROOT / "artifacts/exploratory/pems03_channel_patch_update_matched_timemixer_20261007/audit.json")
    pems03_timemixer_final = load(ROOT / "artifacts/exploratory/pems03_channel_patch_batch2_timemixer_final_expansion_20261007/audit.json")
    pems03_positionwise = load(ROOT / "artifacts/exploratory/pems03_channel_patch_positionwise_probe_20261007/audit.json")
    pems03_projector = load(ROOT / "artifacts/exploratory/pems03_channel_patch_projector_probe_20261007/audit.json")
    pems03_cuda_graph = load(ROOT / "artifacts/exploratory/pems03_channel_patch_25pct_external_replication_20261007/inference_benchmark/cuda_graph_audit.json")
    pems03_cuda_grid = load(ROOT / "artifacts/exploratory/pems03_channel_patch_25pct_external_replication_20261007/inference_benchmark/cuda_graph_batch_grid/audit.json")
    weather_calibrated = load(ROOT / "artifacts/exploratory/weather_channel_patch_calibration_replication_20261007/audit.json")
    weather_calibrated_patch = load(ROOT / "artifacts/exploratory/official_patchtst_weather_channel_patch_calibration_20261007/audit.json")
    weather_calibrated_robust = load(ROOT / "artifacts/exploratory/weather_channel_patch_calibration_replication_20261007/robustness_validation/audit.json")
    weather_calibrated_itransformer = load(ROOT / "artifacts/exploratory/official_itransformer_weather_channel_patch_calibration_20261007/audit.json")
    weather_calibrated_timemixer = load(ROOT / "artifacts/exploratory/official_timemixer_weather_channel_patch_calibration_20261007/audit.json")
    weather_calibrated_25 = load(ROOT / "artifacts/exploratory/weather_channel_patch_calibration_25pct_replication_20261007/audit.json")
    weather_calibrated_gate = load(ROOT / "artifacts/exploratory/weather_calibrated_channelwise_gate_probe_20261007/audit.json")
    electricity_gate_final = load(ROOT / "artifacts/exploratory/electricity_d32_channelwise_gate_final_expansion_20261007/audit.json")
    traffic_channel_patch = load(ROOT / "artifacts/exploratory/traffic_d32_channel_patch_probe_20261007/audit.json")
    traffic_channel_weighting = load(ROOT / "artifacts/exploratory/traffic_d32_channel_patch_weighting_probe_20261007/audit.json")
    traffic_channel_locality = load(ROOT / "artifacts/exploratory/traffic_d32_channel_patch_locality_probe_20261007/audit.json")
    traffic_channel_last_patch = load(ROOT / "artifacts/exploratory/traffic_d32_channel_patch_last_patch_probe_20261007/audit.json")
    traffic_grouped_patch = load(ROOT / "artifacts/exploratory/traffic_d32_grouped_patch_probe_20261007/audit.json")
    traffic_acf_descriptor = load(ROOT / "artifacts/exploratory/traffic_d32_acf_descriptor_probe_20261007/audit.json")
    traffic_acf_descriptor_scale = load(ROOT / "artifacts/exploratory/traffic_d32_acf_descriptor_scale_probe_20261007/audit.json")
    traffic = ("pems_bay", "metr_la")
    traffic10 = all(low10["targets"][name]["channel_patch_vs_forecast"]["wins"] == 3 for name in traffic)
    traffic25 = all(low25["targets"][name]["channel_patch_vs_forecast"]["wins"] == 3 for name in traffic)
    traffic_test = all(test["targets"][name]["channel_patch_vs_forecast"]["wins"] == 3 for name in traffic)
    one_scan = (inference["mamba_scan_count"] == 1 and not inference["relation_loss_included"] and not inference["autocon_representation_returned"] and inference["mamba_encoder_block_calls"] == [inference["warmup"] + inference["iterations"]])
    result = {
        "objective": "channel-count-agnostic one-scan MambaTSv3 with train-only ACF representation pretraining",
        "mechanism": {
            "low_label_10pct_traffic_replication": traffic10,
            "low_label_25pct_traffic_replication": traffic25,
            "validation_external_robustness_10pct": bool(robust10["all_external_noninferior"]),
            "validation_external_robustness_25pct": bool(robust25["all_external_noninferior"]),
            "long_horizon_h336_replication": bool(h336["channel_patch_vs_forecast"]["wins"] == 3 and h336["channel_patch_vs_official_patchtst"]["noninferior_at_one_se"]),
            "long_horizon_h336_robustness": bool(h336_robust["all_external_noninferior"]),
            "source_representation_geometry": bool(geometry["representation_geometry_supported"]),
            "source_representation_margin_change": geometry["mean_margin_change"],
            "weather_validation_external_noninferiority": bool(weather["channel_patch_vs_official_patchtst"]["noninferior_at_one_se"]),
            "weather_validation_robustness": bool(weather_robust["all_external_noninferior"]),
            "weather_target_calibrated_channel_patch_replication": bool(weather_calibrated["promotion_eligible"]),
            "weather_target_calibrated_vs_forecast_percent": weather_calibrated["channel_patch_vs_forecast_calibration"]["percent"],
            "weather_target_calibrated_vs_patchtst_percent": weather_calibrated_patch["channel_patch_vs_official_patchtst"]["percent"],
            "weather_target_calibrated_vs_itransformer_percent": weather_calibrated_itransformer["channel_patch_vs_official_itransformer"]["percent"],
            "weather_target_calibrated_itransformer_noninferior": bool(weather_calibrated_itransformer["channel_patch_vs_official_itransformer"]["noninferior_at_one_se"]),
            "weather_target_calibrated_vs_timemixer_percent": weather_calibrated_timemixer["channel_patch_vs_official_timemixer"]["percent"],
            "weather_target_calibrated_timemixer_noninferior": bool(weather_calibrated_timemixer["channel_patch_vs_official_timemixer"]["noninferior_at_one_se"]),
            "weather_target_calibrated_robustness": bool(weather_calibrated_robust["all_external_noninferior"]),
            "weather_target_calibrated_25pct_replication": bool(weather_calibrated_25["promotion_eligible"]),
            "weather_target_calibrated_25pct_vs_forecast_percent": weather_calibrated_25["channel_patch_vs_forecast_calibration"]["percent"],
            "weather_target_calibrated_channelwise_gate_improves": bool(weather_calibrated_gate["screen_passes_expansion_margin"]),
            "pems03_target_excluded_external_replication": bool(
                pems03["target_excluded_from_source_and_initialization"]
                and pems03["external_eligible"]
            ),
            "pems03_vs_forecast_percent": pems03["channel_patch_vs_forecast"]["percent"],
            "pems03_vs_official_patchtst_percent": pems03["channel_patch_vs_official_patchtst"]["percent"],
            "pems03_validation_robustness": bool(pems03_robust["all_external_noninferior"]),
            "pems03_25pct_target_excluded_external_replication": bool(
                pems03_25["target_excluded_from_source_and_initialization"]
                and pems03_25["external_eligible"]
            ),
            "pems03_25pct_vs_forecast_percent": pems03_25["channel_patch_vs_forecast"]["percent"],
            "pems03_25pct_vs_official_patchtst_percent": pems03_25["channel_patch_vs_official_patchtst"]["percent"],
            "pems03_25pct_vs_official_itransformer_percent": pems03_itransformer["channel_patch_vs_official_itransformer"]["percent"],
            "pems03_25pct_official_itransformer_noninferior": bool(pems03_itransformer["channel_patch_vs_official_itransformer"]["noninferior_at_one_se"]),
            "pems03_25pct_vs_official_timemixer_percent": pems03_timemixer["channel_patch_vs_official_timemixer"]["percent"],
            "pems03_25pct_official_timemixer_noninferior": bool(pems03_timemixer["channel_patch_vs_official_timemixer"]["noninferior_at_one_se"]),
            "pems03_25pct_update_matched_vs_timemixer_percent": pems03_timemixer_update_matched["channel_patch_vs_official_timemixer"]["percent"],
            "pems03_25pct_update_matched_timemixer_noninferior": bool(pems03_timemixer_update_matched["channel_patch_vs_official_timemixer"]["noninferior_at_one_se"]),
            "pems03_25pct_final_batch_matched_vs_timemixer_percent": pems03_timemixer_final["channel_patch_vs_official_timemixer"]["percent"],
            "pems03_25pct_final_batch_matched_timemixer_noninferior": bool(pems03_timemixer_final["channel_patch_vs_official_timemixer"]["noninferior_at_one_se"]),
            "pems03_25pct_positionwise_patch_similarity_improves": bool(pems03_positionwise["screen_passes_expansion_margin"]),
            "pems03_25pct_positionwise_vs_trajectory_percent": pems03_positionwise["positionwise_vs_trajectory_percent"],
            "pems03_25pct_shared_projector_improves": bool(pems03_projector["screen_passes_expansion_margin"]),
            "pems03_25pct_shared_projector_vs_identity_percent": pems03_projector["mlp_vs_identity_percent"],
            "pems03_25pct_validation_robustness": bool(pems03_25_robust["all_external_noninferior"]),
        },
        "deployment": {"one_scan_no_relation_loss": one_scan, "latency_ms_median": inference["latency_ms_median"], "external_latency_superiority": False, "pems03_25pct_one_scan_no_relation_loss": bool(pems03_inference["candidate_one_scan_no_relation_loss"]), "pems03_25pct_latency_ms_median": pems03_inference["channel_patch_latency_ms_median"], "pems03_25pct_vs_patchtst_latency_percent": pems03_inference["candidate_vs_patchtst_percent"], "pems03_25pct_cuda_graph_fixed_shape_equivalent": bool(pems03_cuda_graph["cuda_graph_equivalent"]), "pems03_25pct_cuda_graph_latency_ms_median": pems03_cuda_graph["cuda_graph_latency_ms_median"], "pems03_25pct_cuda_graph_vs_patchtst_percent": pems03_cuda_graph["cuda_graph_vs_patchtst_percent"], "pems03_25pct_cuda_graph_faster_batches": pems03_cuda_grid["candidate_faster_batches"], "pems03_25pct_cuda_graph_batch_grid": pems03_cuda_grid["rows"]},
        "reproducibility": {"clean_interpreter_same_machine": bool(clean["audit_valid"] and clean["same_machine"]), "independent_machine": bool(independent["audit_valid"] and independent["independent_machine"] and independent.get("host_evidence_verified", {}).get("different_from_audit_host")), "independent_machine_vs_forecast_percent": independent["channel_patch_vs_forecast"]["percent"], "independent_machine_wins": independent["channel_patch_vs_forecast"]["wins"]},
        "test_extension": {"historical_public_test_access_caveat": bool(test["historical_public_test_access_caveat"]), "traffic_mechanism_replication": traffic_test, "weather_forecast_only_reversal": not test["targets"]["weather"]["channel_patch_vs_forecast"]["noninferior_at_one_se"]},
        "highdim_target_gate": {"improves_electricity_scalar_gate": True, "five_seed_external_noninferiority": bool(channel_gate["channelwise_gate_vs_official_patchtst"]["noninferior_at_one_se"]), "five_seed_external_percent": channel_gate["channelwise_gate_vs_official_patchtst"]["percent"], "final_ten_seed_external_noninferiority": bool(electricity_gate_final["channelwise_gate_vs_official_patchtst"]["noninferior_at_one_se"]), "final_ten_seed_external_percent": electricity_gate_final["channelwise_gate_vs_official_patchtst"]["percent"], "traffic_channel_patch_improves": bool(traffic_channel_patch["screen_passes_expansion_margin"]), "traffic_channel_patch_vs_forecast_percent": traffic_channel_patch["channel_patch_vs_forecast_percent"], "traffic_acf_strength_channel_weighting_improves": bool(traffic_channel_weighting["screen_passes_expansion_margin"]), "traffic_acf_strength_channel_weighting_percent": traffic_channel_weighting["acf_strength_vs_uniform_percent"], "traffic_source_only_channel_patch_improves": bool(traffic_channel_locality["screen_passes_expansion_margin"]), "traffic_source_only_channel_patch_percent": traffic_channel_locality["source_only_vs_source_and_target_percent"], "traffic_last_patch_channel_similarity_improves": bool(traffic_channel_last_patch["screen_passes_expansion_margin"]), "traffic_last_patch_channel_similarity_percent": traffic_channel_last_patch["last_patch_vs_trajectory_percent"], "traffic_grouped_patch_improves": bool(traffic_grouped_patch["screen_passes_expansion_margin"]), "traffic_grouped_patch_vs_forecast_percent": traffic_grouped_patch["grouped_patch_vs_forecast_percent"], "traffic_acf_descriptor_improves": bool(traffic_acf_descriptor["screen_passes_expansion_margin"]), "traffic_acf_descriptor_vs_forecast_percent": traffic_acf_descriptor["descriptor_vs_forecast_percent"], "traffic_acf_descriptor_vs_channel_patch_percent": traffic_acf_descriptor["descriptor_vs_channel_patch_percent"], "traffic_acf_descriptor_target_scale_improves": bool(traffic_acf_descriptor_scale["screen_passes_expansion_margin"]), "traffic_acf_descriptor_target_scale_vs_forecast_percent": traffic_acf_descriptor_scale["target_scale_vs_forecast_percent"], "traffic_acf_descriptor_target_scale_vs_frozen_percent": traffic_acf_descriptor_scale["target_scale_vs_frozen_descriptor_percent"], "pems_external_probe_improves": bool(pems_gate["channelwise_better"]), "traffic_active_null_recovers": bool(traffic_gate["active_better_than_null"])},
        "known_failures": {"electricity_external_patchtst": not electricity["channel_patch_vs_official_patchtst"]["noninferior_at_one_se"], "d32_channel_patch_replication": not d32["channel_patch_vs_forecast"]["noninferior_at_one_se"], "d32_channelwise_gate_external": not electricity_gate_final["channelwise_gate_vs_official_patchtst"]["noninferior_at_one_se"], "traffic_d32_channel_patch": not traffic_channel_patch["screen_passes_expansion_margin"], "traffic_d32_channel_patch_acf_strength_weighting": not traffic_channel_weighting["screen_passes_expansion_margin"], "traffic_d32_channel_patch_source_only": not traffic_channel_locality["screen_passes_expansion_margin"], "traffic_d32_channel_patch_last_patch": not traffic_channel_last_patch["screen_passes_expansion_margin"], "traffic_d32_grouped_patch": not traffic_grouped_patch["screen_passes_expansion_margin"], "traffic_d32_acf_descriptor": not traffic_acf_descriptor["screen_passes_expansion_margin"], "traffic_d32_acf_descriptor_target_scale": not traffic_acf_descriptor_scale["screen_passes_expansion_margin"], "pems03_shared_projector": not pems03_projector["screen_passes_expansion_margin"]},
        "q1_ready": False,
        "q1_blockers": ["historical public-test access caveat", "Weather channel-patch test reversal versus forecast-only", "Weather calibrated Channel-Patch fails TimeMixer noninferiority", "Electricity channelwise gate remains narrowly externally inferior to official PatchTST", "no external latency superiority"],
        "submission_ready": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
