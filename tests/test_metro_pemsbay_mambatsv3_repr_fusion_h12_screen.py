import json
from pathlib import Path

from scripts.analyze_metro_pemsbay_mambatsv3_repr_fusion_h12_screen import _read
from scripts.run_training_plan import _load_json, _validate_plan


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/research/metro_pemsbay_mambatsv3_autocon_repr_fusion_h12_screen_plan_20260927.json"
ARTIFACT_ROOT = ROOT / "artifacts/exploratory/metro_pemsbay_mambatsv3_autocon_repr_fusion_h12_screen_20260927"


def test_fusion_screen_plan_and_phase_artifacts_are_valid():
    plan = _load_json(PLAN)
    assert _validate_plan(plan) == []
    off, off_hash = _read(ARTIFACT_ROOT, plan, "fusion_off_h12")
    on, on_hash = _read(ARTIFACT_ROOT, plan, "fusion_on_h12")
    assert len(off) == len(on) == 6
    assert off_hash == on_hash
    assert all("autocon_loss_mean" in row for row in on)


def test_fusion_screen_has_no_test_metrics():
    for path in ARTIFACT_ROOT.rglob("phase_validation.json"):
        payload = json.loads(path.read_text())
        assert payload["valid"] is True
    assert not list(ARTIFACT_ROOT.rglob("stream_metrics.json"))
    assert not list(ARTIFACT_ROOT.rglob("metrics.npy"))
