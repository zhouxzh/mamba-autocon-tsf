"""Regression checks for the AutoCon representation projection contract."""

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from layers.CausalEmbedding import CausalDataEmbedding
from layers.Embed import DataEmbedding
from models.AutoConNet import Model as AutoConNet
import models.Mamba1AutoCon as mamba1_module
import models.Mamba2AutoCon as mamba2_module
from models.Mamba2AutoCon import _MambaForecastModel


def _autoconnet_config(auto_con=True):
    return SimpleNamespace(
        seq_len=8,
        label_len=4,
        pred_len=3,
        enc_in=2,
        c_out=2,
        d_model=4,
        d_ff=6,
        e_layers=1,
        AutoCon=auto_con,
        AutoCon_wnorm="LastVal",
        AutoCon_multiscales=[4],
        embed="timeF",
        freq="h",
        dropout=0.0,
    )


def _inputs(batch=2, channels=2):
    x = torch.randn(batch, 8, channels)
    mark = torch.randn(batch, 8, 4)
    dec = torch.zeros(batch, 7, channels)
    dec_mark = torch.randn(batch, 7, 4)
    return x, mark, dec, dec_mark


class AutoConNetProjectionTests(unittest.TestCase):
    def test_projection_is_consistent_across_representation_interfaces(self):
        torch.manual_seed(101)
        model = AutoConNet(_autoconnet_config()).eval()
        inputs = _inputs()

        _, returned = model(*inputs)
        _, only_repr = model(*inputs, onlyrepr=True)
        embeddings = model.get_embeddings(*inputs)

        self.assertTrue(torch.allclose(returned, only_repr))
        self.assertTrue(torch.allclose(returned, embeddings))

        returned.square().mean().backward()
        self.assertIsNotNone(model.repr_head.weight.grad)
        self.assertTrue(torch.isfinite(model.repr_head.weight.grad).all())

    def test_projection_does_not_change_forecasting_path(self):
        torch.manual_seed(103)
        model = AutoConNet(_autoconnet_config()).eval()
        altered = deepcopy(model).eval()
        with torch.no_grad():
            altered.repr_head.weight.zero_()
            altered.repr_head.bias.fill_(100.0)
        inputs = _inputs()

        prediction, representation = model(*inputs)
        altered_prediction, altered_representation = altered(*inputs)

        self.assertTrue(torch.allclose(prediction, altered_prediction, atol=1e-6, rtol=1e-6))
        self.assertFalse(torch.allclose(representation, altered_representation))


class MambaProjectionContractTests(unittest.TestCase):
    def test_causal_embedding_flag_is_effective_and_default_is_legacy(self):
        class _IdentityMamba(nn.Module):
            def __init__(self, **_kwargs):
                super().__init__()

            def forward(self, values):
                return values

        with patch.object(mamba2_module, "_Mamba2", _IdentityMamba), \
                patch.object(mamba2_module, "_MAMBA_IMPORT_ERROR", None):
            legacy = _MambaForecastModel(_autoconnet_config(auto_con=True))
            causal_config = _autoconnet_config(auto_con=True)
            causal_config.mamba_causal_embedding = True
            causal = _MambaForecastModel(causal_config)

        self.assertIsInstance(legacy.enc_embedding, DataEmbedding)
        self.assertNotIsInstance(legacy.enc_embedding, CausalDataEmbedding)
        self.assertIsInstance(causal.enc_embedding, CausalDataEmbedding)
        self.assertTrue(causal.causal_embedding)

    def test_mamba1_reuses_the_causal_embedding_contract(self):
        class _IdentityMamba(nn.Module):
            def __init__(self, **_kwargs):
                super().__init__()

            def forward(self, values):
                return values

        from models.Mamba1AutoCon import Model as Mamba1Model

        config = _autoconnet_config(auto_con=False)
        config.mamba_causal_embedding = True
        with patch.object(mamba1_module, "_Mamba1", _IdentityMamba), \
                patch.object(mamba1_module, "_MAMBA1_IMPORT_ERROR", None):
            model = Mamba1Model(config)

        self.assertIsInstance(model.enc_embedding, CausalDataEmbedding)
        self.assertTrue(model.causal_embedding)

    def test_shared_mamba_helper_projects_only_in_autocon_mode(self):
        probe = _MambaForecastModel.__new__(_MambaForecastModel)
        nn.Module.__init__(probe)
        probe.AutoCon = True
        probe.repr_head = nn.Linear(4, 4)
        backbone = torch.randn(2, 5, 4, requires_grad=True)

        projected = probe._contrastive_repr(backbone)
        self.assertTrue(torch.allclose(projected, probe.repr_head(backbone)))
        projected.square().mean().backward()
        self.assertIsNotNone(probe.repr_head.weight.grad)
        self.assertTrue(torch.isfinite(probe.repr_head.weight.grad).all())

        plain_probe = _MambaForecastModel.__new__(_MambaForecastModel)
        nn.Module.__init__(plain_probe)
        plain_probe.AutoCon = False
        plain_probe.repr_head = nn.Linear(4, 4)
        plain_backbone = torch.randn(2, 5, 4)
        self.assertIs(plain_probe._contrastive_repr(plain_backbone), plain_backbone)

    def test_explicit_prediction_only_path_skips_autocon_projection(self):
        """The deployed forecast is unchanged when its auxiliary output is disabled."""

        class _IdentityMamba(nn.Module):
            def __init__(self, **_kwargs):
                super().__init__()

            def forward(self, values):
                return values

        torch.manual_seed(107)
        with patch.object(mamba2_module, "_Mamba2", _IdentityMamba), \
                patch.object(mamba2_module, "_MAMBA_IMPORT_ERROR", None):
            model = _MambaForecastModel(_autoconnet_config(auto_con=True)).eval()
            inputs = _inputs()

            prediction, representation = model(*inputs)
            prediction_only, omitted = model(*inputs, return_repr=False)

        self.assertTrue(torch.allclose(prediction, prediction_only, atol=1e-6, rtol=1e-6))
        self.assertIsNotNone(representation)
        self.assertIsNone(omitted)

        # ``onlyrepr`` remains an explicit override for representation callers.
        _, forced = model(*inputs, onlyrepr=True, return_repr=False)
        self.assertTrue(torch.allclose(representation, forced, atol=1e-6, rtol=1e-6))

    def test_mamba2_representation_forecast_fusion_is_zero_gated_and_connected(self):
        """The opt-in residual preserves initialization and receives MSE gradients."""

        class _IdentityMamba(nn.Module):
            def __init__(self, **_kwargs):
                super().__init__()

            def forward(self, values):
                return values

        torch.manual_seed(211)
        base_config = _autoconnet_config(auto_con=True)
        fused_config = _autoconnet_config(auto_con=True)
        fused_config.mamba2_repr_forecast_fusion = True
        with patch.object(mamba2_module, "_Mamba2", _IdentityMamba), \
                patch.object(mamba2_module, "_MAMBA_IMPORT_ERROR", None):
            base = _MambaForecastModel(base_config).eval()
            torch.manual_seed(211)
            fused = _MambaForecastModel(fused_config).eval()
            base_state = base.state_dict()
            fused_state = fused.state_dict()
            shared_keys = sorted(set(base_state).intersection(fused_state))
            self.assertTrue(shared_keys)
            for key in shared_keys:
                torch.testing.assert_close(base_state[key], fused_state[key])
            fused.load_state_dict(base.state_dict(), strict=False)
            inputs = _inputs()
            base_prediction, _ = base(*inputs, return_repr=False)
            fused_prediction, omitted = fused(*inputs, return_repr=False)

            self.assertTrue(fused.mamba2_repr_forecast_fusion_enabled)
            self.assertIsNone(omitted)
            self.assertEqual(float(fused.mamba2_repr_forecast_scale.detach().abs().max()), 0.0)
            torch.testing.assert_close(base_prediction, fused_prediction)

            fused.mamba2_repr_forecast_scale.data.fill_(0.5)
            fused.zero_grad(set_to_none=True)
            prediction, _ = fused(*inputs, return_repr=False)
            prediction.square().mean().backward()

        self.assertIsNotNone(fused.repr_head.weight.grad)
        self.assertGreater(float(fused.repr_head.weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
