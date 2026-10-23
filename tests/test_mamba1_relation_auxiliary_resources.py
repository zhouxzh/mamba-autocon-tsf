"""Regression checks for the Mamba-1 resource attestation inputs."""

from pathlib import Path

from scripts.attest_mamba1_relation_auxiliary_resources import phase_configs


def test_mamba1_resource_inputs_are_bound_to_complete_phases():
    candidate = Path("artifacts/exploratory/mambatsv3_highdim_relation_auxiliary_only_h96_validation_20260928/candidate")
    baseline = Path("artifacts/exploratory/mambatsv3_mamba1_baseline_highdim_h96_validation_20260929/baseline")
    candidate_snapshot, candidate_configs, candidate_hash = phase_configs(candidate, "candidate")
    baseline_snapshot, baseline_configs, baseline_hash = phase_configs(baseline, "baseline")
    assert candidate_snapshot.is_dir()
    assert baseline_snapshot.is_dir()
    assert candidate_hash == baseline_hash
    assert len(candidate_configs) == 6
    assert len(baseline_configs) == 6


def test_mamba1_resource_readout_preserves_speed_failure_boundary():
    readout = Path("artifacts/exploratory/mambatsv3_relation_auxiliary_mamba1_resource_h96_20260929/readout/readout.md").read_text(encoding="utf-8")
    assert "+79.2%" in readout
    assert "+242.0%" in readout
    assert "test_metrics_produced" not in readout
