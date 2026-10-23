import json
from pathlib import Path

from scripts.analyze_metro_pemsbay_mambatsv3_repr_fusion_h12_baseline_confirmation import _baseline_rows
from scripts.run_training_plan import _load_json, _validate_plan


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/research/metro_pemsbay_mambatsv3_repr_fusion_h12_baseline_confirmation_plan_20260927.json"
ARTIFACT_ROOT = ROOT / "artifacts/exploratory/metro_pemsbay_mambatsv3_repr_fusion_h12_baseline_confirmation_20260927"


def test_same_source_baseline_confirmation_is_complete():
    plan = _load_json(PLAN)
    assert _validate_plan(plan) == []
    rows, source = _baseline_rows(ARTIFACT_ROOT, plan)
    assert len(rows) == 36
    assert source
    assert {row["model"] for row in rows} == {"DLinear", "NLinear", "PatchTST", "TimeMixer", "iTransformer", "MambaTSv3"}


def test_baseline_confirmation_has_no_test_metrics():
    assert not list(ARTIFACT_ROOT.rglob("metrics.npy"))
    assert not list(ARTIFACT_ROOT.rglob("stream_metrics.json"))
    validation = json.loads((ARTIFACT_ROOT / "external_context_h12/phase_validation.json").read_text())
    assert validation["valid"] is True
