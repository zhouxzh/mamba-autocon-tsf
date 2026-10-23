import json
from pathlib import Path


def test_latent_temporal_mamba_is_registered_with_shared_autocon_path():
    model = Path("models/LatentTemporalMamba.py").read_text(encoding="utf-8")
    runner = Path("run.py").read_text(encoding="utf-8")
    resources = Path("scripts/benchmark_model_resources.py").read_text(encoding="utf-8")
    reader = Path("scripts/analyze_latent_temporal_mamba_highdim_validation.py").read_text(encoding="utf-8")
    audit = Path("scripts/audit_latent_temporal_mamba_highdim_plan.py").read_text(encoding="utf-8")
    smoke = Path("scripts/smoke_latent_temporal_mamba_autocon.py").read_text(encoding="utf-8")
    for token in ("temporal_stride = 4", "variable_compress", "forecast_scale", "repr_projection"):
        assert token in model
    assert "LatentTemporalMambaAutoCon" in runner
    assert "LatentTemporalMambaAutoCon" in resources
    assert "TEST_ARTIFACTS" in reader
    assert "autocon_vs_null_pass" in reader
    assert "speed_gate_failed" in audit
    assert "representation gradient path is zero" in smoke


def test_highdim_validation_plan_keeps_auto_con_control_and_test_boundary():
    plan = json.loads(Path(
        "docs/research/latent_temporal_mamba_highdim_autocon_validation_20261001.json"
    ).read_text(encoding="utf-8"))
    assert plan["execution"]["authorized"] is True
    assert plan["registered_job_counts"]["total"] == 60
    phases = {phase["name"]: phase for phase in plan["phases"]}
    assert phases["latent_temporal_null_h96_validation"]["autocon_lambda"] == 0.0
    assert phases["latent_temporal_active_h96_validation"]["autocon_lambda"] == 0.001
    assert all(phase["skip_test"] is True for phase in phases.values())
    assert all(phase["stream_metrics"] is False for phase in phases.values())


def test_shared_pretraining_backbone_is_channel_agnostic_and_inference_separates_autocon():
    model = Path("models/MambaTSPretrain.py").read_text(encoding="utf-8")
    script = Path("scripts/pretrain_mamba_autocon.py").read_text(encoding="utf-8")
    for token in ("patch_embedding", "self.mamba", "per_channel.mean(dim=1)", "pretrain=True", "reconstruction_head"):
        assert token in model
    for token in ("fit_signed_acf_prior", "ACFRelationLoss", "train_fraction", "--dataset"):
        assert token in script
    assert "relation loss" in model or "reconstruction head" in model
