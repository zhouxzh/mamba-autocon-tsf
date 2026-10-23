"""Contract tests for optional Mamba imports and normalization validation."""

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from run import mamba_architecture_tag, validate_mamba_acf_arguments
from scripts.run_training_plan import (
    _validate_mamba_options,
    _validate_numeric_plan_config,
)
from scripts.validate_training_phase import _validate_mamba_acf_wnorm_contract


ROOT = Path(__file__).resolve().parents[1]


def _assert_partial_mamba_import(
    available_symbol: str,
    available_attr: str,
    missing_attr: str,
    available_kind: str,
    missing_kind: str,
    missing_label: str,
):
    script = f"""
import sys
import types
import torch.nn as nn

fake = types.ModuleType("mamba_ssm")
class {available_symbol}(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, values):
        return values

fake.{available_symbol} = {available_symbol}
sys.modules["mamba_ssm"] = fake

import models.MambaACF as module

assert getattr(module, "{available_attr}") is getattr(fake, "{available_symbol}")
assert getattr(module, "{missing_attr}") is None
available = module._MambaFFNBlock(8, 4, 2, 2, 4, 0.0, "{available_kind}", 16)
assert isinstance(available.mixer, getattr(fake, "{available_symbol}"))
try:
    module._MambaFFNBlock(8, 4, 2, 2, 4, 0.0, "{missing_kind}", 16)
except ImportError as exc:
    assert "{missing_label}" in str(exc), str(exc)
else:
    raise AssertionError("missing Mamba branch did not fail at construction")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_mamba1_remains_available_when_mamba2_symbol_is_missing():
    _assert_partial_mamba_import(
        "Mamba",
        "_Mamba1",
        "_Mamba2",
        "mamba1",
        "mamba2",
        "Mamba-2",
    )


def test_mamba2_remains_available_when_mamba1_symbol_is_missing():
    _assert_partial_mamba_import(
        "Mamba2",
        "_Mamba2",
        "_Mamba1",
        "mamba2",
        "mamba1",
        "Mamba-1",
    )


def test_run_entry_rejects_decomp_only_for_mamba_acf():
    with pytest.raises(ValueError, match=r"MambaACF.*Decomp"):
        validate_mamba_acf_arguments(
            SimpleNamespace(model="MambaACF", AutoCon_wnorm="Decomp")
        )

    # The normalization mode remains available to the separate Mamba2 model.
    validate_mamba_acf_arguments(
        SimpleNamespace(model="Mamba2", AutoCon_wnorm="Decomp")
    )


def test_run_entry_accepts_explicit_meanstd_parity_mode():
    validate_mamba_acf_arguments(
        SimpleNamespace(model="MambaACF", AutoCon_wnorm="MeanStd")
    )
    validate_mamba_acf_arguments(
        SimpleNamespace(model="Mamba2AutoCon", AutoCon_wnorm="MeanStd")
    )


def test_revin_requires_at_least_two_window_steps_at_runtime_and_plan_boundaries():
    with pytest.raises(ValueError, match=r"AutoCon_wnorm=ReVIN requires seq_len >= 2"):
        validate_mamba_acf_arguments(
            SimpleNamespace(model="MambaACF", seq_len=1, AutoCon_wnorm="ReVIN")
        )
    validate_mamba_acf_arguments(
        SimpleNamespace(model="MambaACF", seq_len=2, AutoCon_wnorm="ReVIN")
    )

    for key in ("autocon_wnorm", "AutoCon_wnorm"):
        errors = _validate_numeric_plan_config(
            "revin", {"seq_len": 1, key: "ReVIN"}
        )
        assert any("AutoCon_wnorm=ReVIN requires seq_len >= 2" in error for error in errors)
    assert _validate_numeric_plan_config(
        "revin", {"seq_len": 2, "autocon_wnorm": "ReVIN"}
    ) == []


def test_acf_repr_fusion_rejects_inactive_or_unsupported_model_bindings():
    with pytest.raises(ValueError, match=r"mamba_acf_repr_fusion.*--AutoCon"):
        validate_mamba_acf_arguments(
            SimpleNamespace(
                model="MambaACF",
                AutoCon=False,
                mamba_acf_repr_fusion=True,
            )
        )
    with pytest.raises(ValueError, match=r"mamba_acf_repr_fusion.*MambaACF AutoCon"):
        validate_mamba_acf_arguments(
            SimpleNamespace(
                model="Mamba2AutoCon",
                AutoCon=True,
                mamba_acf_repr_fusion=True,
            )
        )
    with pytest.raises(ValueError, match=r"mamba_acf_repr_fusion.*MambaACFAutoCon"):
        validate_mamba_acf_arguments(
            SimpleNamespace(
                model="MambaACF",
                AutoCon=True,
                mamba_acf_repr_fusion=True,
            )
        )
    validate_mamba_acf_arguments(
        SimpleNamespace(
            model="MambaACFAutoCon",
            AutoCon=True,
            mamba_acf_relation_loss=True,
            mamba_acf_repr_fusion=True,
        )
    )

    invalid = _validate_mamba_options(
        "acf_repr", {"seq_len": 96, "mamba_acf_repr_fusion": True},
        ["MambaACF"],
    )
    assert any("only MambaACF AutoCon models" in error for error in invalid)
    unknown = _validate_mamba_options(
        "acf_repr", {"seq_len": 96, "mamba_acf_repr_fusion": True},
        ["MambaACFUnexpectedAutoCon"],
    )
    assert any("only MambaACF AutoCon models" in error for error in unknown)


def test_variable_repr_fusion_requires_autocon_variable_mamba_acf():
    with pytest.raises(ValueError, match="mamba_variable_repr_fusion.*AutoCon"):
        validate_mamba_acf_arguments(
            SimpleNamespace(
                model="MambaACF",
                AutoCon=False,
                mamba_variable_mixer=True,
                mamba_variable_repr_fusion=True,
            )
        )
    with pytest.raises(ValueError, match="mamba_variable_repr_fusion.*variable_mixer"):
        validate_mamba_acf_arguments(
            SimpleNamespace(
                model="MambaACFAutoCon",
                AutoCon=True,
                mamba_variable_mixer=False,
                mamba_variable_repr_fusion=True,
                mamba_acf_relation_loss=True,
            )
        )
    with pytest.raises(ValueError, match="supported only by MambaACF"):
        validate_mamba_acf_arguments(
            SimpleNamespace(
                model="Mamba2AutoCon",
                AutoCon=True,
                mamba_variable_mixer=True,
                mamba_variable_repr_fusion=True,
            )
        )


def test_setting_override_is_evaluation_only_and_nonempty():
    validate_mamba_acf_arguments(
        SimpleNamespace(model="Mamba2", AutoCon_wnorm="Decomp", is_training=0,
                         setting_override="historical_setting")
    )
    with pytest.raises(ValueError, match="setting_override.*is_training=0"):
        validate_mamba_acf_arguments(
            SimpleNamespace(model="Mamba2", AutoCon_wnorm="Decomp", is_training=1,
                             setting_override="historical_setting")
        )
    with pytest.raises(ValueError, match="setting_override must be non-empty"):
        validate_mamba_acf_arguments(
            SimpleNamespace(model="Mamba2", AutoCon_wnorm="Decomp", is_training=0,
                             setting_override="   ")
        )


@pytest.mark.parametrize(
    "model,autocon,relation_loss,match",
    [
        ("MambaACF", False, True, "requires --model MambaACFAutoCon"),
        ("Mamba1AutoCon", True, True, "requires --model MambaACFAutoCon"),
        ("MambaACFAutoCon", True, False, "requires --mamba_acf_relation_loss"),
        ("MambaACFAutoCon", False, True, "requires --AutoCon"),
    ],
)
def test_relation_loss_model_binding_is_rejected_before_experiment_construction(
    model, autocon, relation_loss, match
):
    with pytest.raises(ValueError, match=match):
        validate_mamba_acf_arguments(
            SimpleNamespace(
                model=model,
                AutoCon=autocon,
                AutoCon_wnorm="LastVal",
                mamba_acf_relation_loss=relation_loss,
            )
        )
    with pytest.raises(ValueError, match="single directory name"):
        validate_mamba_acf_arguments(
            SimpleNamespace(model="Mamba2", AutoCon_wnorm="Decomp", is_training=0,
                             setting_override="../outside")
        )


def test_mamba_acf_autocon_relation_binding_accepts_the_supported_combination():
    validate_mamba_acf_arguments(
        SimpleNamespace(
            model="MambaACFAutoCon",
            AutoCon=True,
            AutoCon_wnorm="LastVal",
            mamba_acf_relation_loss=True,
        )
    )


def test_run_cli_exposes_eval_acf_diagnostics_switch():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "run.py"), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "--mamba_collect_eval_acf_diagnostics" in completed.stdout


def test_plan_and_phase_validation_reject_the_same_combination():
    errors = _validate_mamba_options(
        "contract",
        {"autocon_wnorm": "Decomp"},
        ["MambaACFAutoCon"],
    )
    assert any("MambaACF" in error and "Decomp" in error for error in errors)
    assert not any(
        "AutoCon_wnorm=Decomp" in error
        for error in _validate_mamba_options(
            "contract", {"autocon_wnorm": "Decomp"}, ["Mamba2"]
        )
    )

    assert _validate_mamba_acf_wnorm_contract(
        {"AutoCon_wnorm": "deCOMP"}, "MambaACFAutoCon"
    ) == ["config_AutoCon_wnorm_decomp_unsupported"]
    assert _validate_mamba_acf_wnorm_contract(
        {"AutoCon_wnorm": "Decomp"}, "Mamba2"
    ) == []


def test_acf_fitter_and_autocon_knobs_have_distinct_architecture_tags():
    base = SimpleNamespace(
        model="MambaACF",
        mamba_acf_max_lag=720,
        mamba_acf_top_k=8,
        mamba_acf_min_lag=2,
        mamba_acf_min_abs=0.1,
        mamba_acf_confidence_z=1.96,
        mamba_acf_min_separation=2,
        mamba_acf_chunk=64,
        AutoCon_wnorm="LastVal",
        AutoCon_acf_source="residual",
        AutoCon_multiscales=[96],
        AutoCon_warmup_epochs=0,
        AutoCon_temperature=0.2,
        AutoCon_relation_temperature=0.25,
        AutoCon_sign_weight=0.25,
        AutoCon_sign_margin=0.1,
    )
    baseline_tag = mamba_architecture_tag(base)
    changes = {
        "mamba_acf_max_lag": 721,
        "mamba_acf_top_k": 7,
        "mamba_acf_min_lag": 3,
        "mamba_acf_min_abs": 0.2,
        "mamba_acf_confidence_z": 2.0,
        "mamba_acf_min_separation": 3,
        "mamba_acf_chunk": 32,
        "AutoCon_wnorm": "Mean",
        "AutoCon_acf_source": "raw",
        "AutoCon_multiscales": [48, 96],
        "AutoCon_warmup_epochs": 1,
        "AutoCon_temperature": 0.3,
        "AutoCon_relation_temperature": 0.3,
        "AutoCon_sign_weight": 0.5,
        "AutoCon_sign_margin": 0.2,
    }
    for name, value in changes.items():
        variant = vars(base).copy()
        variant[name] = value
        assert mamba_architecture_tag(SimpleNamespace(**variant)) != baseline_tag, name
