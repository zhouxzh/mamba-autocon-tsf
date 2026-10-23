"""Regression checks for the cross-domain resource-manifest binding."""

from pathlib import Path

from scripts.attest_relation_auxiliary_crossdomain_resources import phase_configs


def test_crossdomain_resource_inputs_are_bound_to_complete_frozen_phases():
    candidate = Path("artifacts/exploratory/mambatsv3_relation_auxiliary_crossdomain_h96_validation_20260929/candidate")
    baseline = Path("artifacts/exploratory/mambatsv3_relation_auxiliary_crossdomain_h96_validation_20260929/baselines")
    candidate_snapshot, candidate_configs = phase_configs(candidate, "candidate_h96_validation", 6)
    baseline_snapshot, baseline_configs = phase_configs(baseline, "baselines_h96_validation", 30)
    assert candidate_snapshot == baseline_snapshot
    assert len(candidate_configs) == 6
    assert len(baseline_configs) == 30
