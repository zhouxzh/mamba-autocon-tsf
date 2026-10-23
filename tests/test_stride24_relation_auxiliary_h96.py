"""Contracts for the stride-24 validation-only accuracy/resource screen."""

import json
from pathlib import Path

from scripts.run_training_plan import _validate_plan


def test_stride24_plan_is_a_complete_validation_only_matrix():
    plan = json.loads(Path("docs/research/mambatsv3_stride24_relation_auxiliary_h96_validation_20260929.json").read_text(encoding="utf-8"))
    assert _validate_plan(plan) == []
    assert plan["execution"]["authorized"] is False
    assert plan["global"]["skip_test"] is True
    assert plan["global"]["save_predictions"] is False
    assert plan["registered_job_counts"]["total"] == 30


def test_stride24_readouts_preserve_accuracy_and_speed_boundaries():
    accuracy = Path("artifacts/exploratory/mambatsv3_stride24_relation_auxiliary_h96_validation_20260929/readout/readout.md").read_text(encoding="utf-8")
    resource = Path("artifacts/exploratory/mambatsv3_stride24_relation_auxiliary_h96_resource_20260929/readout/readout.md").read_text(encoding="utf-8")
    assert "30/30 successful manifest rows" in accuracy
    assert "| electricity | 96 | iTransformer | 0.163764 | 0.186443 | -12.165% | 3/3 |" in accuracy
    assert "| traffic | 96 | iTransformer | 0.384616 | 0.402049 | -4.332% | 3/3 |" in accuracy
    assert "faster than PatchTST, Mamba1; slower than DLinear, iTransformer" in resource
    assert "faster than PatchTST, iTransformer; slower than DLinear, Mamba1" in resource
