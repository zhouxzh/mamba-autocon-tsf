from pathlib import Path


def test_nlinear_variable_mamba_is_registered_with_forecast_shared_autocon_projection():
    model = Path("models/NLinearVariableMambaAutoCon.py").read_text(encoding="utf-8")
    runner = Path("run.py").read_text(encoding="utf-8")
    resources = Path("scripts/benchmark_model_resources.py").read_text(encoding="utf-8")
    for token in ("base_head", "temporal_embedding", "torch.bmm(normalized, tokens)", "variable_head"):
        assert token in model
    assert "time_embedding" not in model
    assert "repr_condition" not in model
    assert "NLinearVariableMambaAutoCon" in runner
    assert "NLinearVariableMambaAutoCon" in resources
    assert "set_acf_prior" in model
    assert "_acf_memory" in model
    assert "raw_history = self._restore_history(normalized, restore)" in model
    assert "self.use_acf_memory" in model
    for token in (
        "set_compact_acf_prior",
        "export_acf_prior_state",
        "save_acf_prior",
        "load_acf_prior",
        "acf_full_prior",
    ):
        assert token in model


def test_nlinear_variable_mamba_can_use_train_only_forecast_acf_objective():
    runner = Path("run.py").read_text(encoding="utf-8")
    marker = "'NLinearVariableMambaAutoCon'"
    forecast_marker = "if ts_forecast_acf_consistency:"
    assert marker in runner
    assert runner.index(marker, runner.index(forecast_marker)) > runner.index(forecast_marker)
    assert "relation-capable AutoCon model" in runner
    contract = Path("utils/acf_relation_contract.py").read_text(encoding="utf-8")
    assert '"NLinearVariableMambaAutoCon"' in contract


def test_nlinear_variable_mamba_has_prediction_connected_representation_fusion_contract():
    model = Path("models/NLinearVariableMambaAutoCon.py").read_text(encoding="utf-8")
    runner = Path("run.py").read_text(encoding="utf-8")
    for token in (
        "variable_repr_fusion_enabled",
        "repr_fusion_gate",
        "repr_fusion_scale",
        "forecast_tokens = tokens",
        "residual = self.variable_head(forecast_tokens)",
    ):
        assert token in model
    assert "NLinearVariableMambaAutoCon" in runner
    assert "mamba_variable_repr_fusion" in runner


def test_nlinear_variable_mamba_exposes_low_rank_cross_variable_residual():
    model = Path("models/NLinearVariableMambaAutoCon.py").read_text(encoding="utf-8")
    for token in (
        "variable_mixer_enabled",
        "variable_mixer_input",
        "variable_mixer_output",
        "variable_mixer_scale",
        'torch.einsum(\n                "rc,bcd->brd"',
        'torch.einsum(\n                "cr,brd->bcd"',
    ):
        assert token in model


def test_nlinear_variable_mamba_exposes_optional_bidirectional_variable_scan():
    model = Path("models/NLinearVariableMambaAutoCon.py").read_text(encoding="utf-8")
    for token in ("bidirectional_enabled", "reverse_mamba", "torch.flip(hidden, dims=(1,))"):
        assert token in model


def test_nlinear_variable_mamba_derives_and_restores_train_prior_channel_order():
    model = Path("models/NLinearVariableMambaAutoCon.py").read_text(encoding="utf-8")
    for token in (
        "signed_acf_profile_similarity_order",
        "acf_channel_order_indices",
        "_set_acf_channel_order",
        "_train_prior_channel_order",
        "tokens.index_select(1, inverse)",
    ):
        assert token in model
