import numpy as np
import torch
from types import SimpleNamespace
from torch import nn
from unittest.mock import patch

import models.MambaTS as mamba_ts
import models.MambaTSv3 as mamba_ts_v3
from utils.acf_channel_descriptor import build_acf_channel_descriptor


class _IdentityMamba(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


def _config(**overrides):
    values = dict(
        task_name="long_term_forecast", seq_len=96, pred_len=12, enc_in=5, c_out=5,
        d_model=8, d_ff=16, patch_len=16, patch_stride=8, dropout=0.0, e_layers=1,
        AutoCon=True, AutoCon_wnorm="MeanStd", mamba_d_state=4, mamba_d_conv=2,
        mamba_expand=2, moving_avg=25,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_train_prefix_acf_descriptor_is_finite_and_channel_count_agnostic():
    acf = np.zeros((3, 12), dtype=np.float32)
    acf[0, 2], acf[1, 3], acf[2, 4] = .8, -.7, .6
    descriptor = build_acf_channel_descriptor(acf, [[2, 0], [3, 0], [4, 0]])
    assert descriptor.shape == (3, 6)
    assert np.isfinite(descriptor).all()
    np.testing.assert_allclose(descriptor.mean(axis=0), np.zeros(6), atol=1e-6)


def test_v3_acf_descriptor_is_zero_nested_and_conditions_existing_scan_only():
    torch.manual_seed(29)
    x = torch.randn(2, 96, 5)
    with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
        control = mamba_ts_v3.Model(_config(AutoCon=True, mamba_ts_context_rank=4))
        conditioned = mamba_ts_v3.Model(_config(
            AutoCon=True, mamba_ts_context_rank=4, mamba_ts_acf_channel_descriptor=True,
        ))
        conditioned.load_state_dict(control.state_dict(), strict=False)
        with torch.no_grad():
            conditioned.ts_acf_descriptor_values.copy_(torch.randn(5, 6))
        torch.testing.assert_close(control(x), conditioned(x), rtol=0.0, atol=0.0)
        with torch.no_grad():
            conditioned.ts_acf_channel_descriptor_scale.fill_(.5)
        conditioned(x)[0].square().mean().backward()
    assert conditioned.ts_acf_channel_descriptor_projection[1].weight.grad is not None
    assert float(conditioned.ts_acf_channel_descriptor_projection[1].weight.grad.abs().sum()) > 0.0


def test_v3_shared_postscan_mixer_is_zero_nested_and_channel_count_independent():
    torch.manual_seed(31)
    with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
        control = mamba_ts_v3.Model(_config(AutoCon=True, mamba_ts_context_rank=4))
        mixed = mamba_ts_v3.Model(_config(
            AutoCon=True, mamba_ts_context_rank=4,
            mamba_ts_shared_postscan_mixer=True,
            mamba_ts_shared_postscan_mixer_rank=3,
        ))
        mixed.load_state_dict(control.state_dict(), strict=False)
        x = torch.randn(2, 96, 5)
        torch.testing.assert_close(control(x)[0], mixed(x)[0], rtol=0.0, atol=0.0)
        with torch.no_grad():
            mixed.ts_shared_postscan_mixer.residual_scale.fill_(.5)
        mixed(x)[0].square().mean().backward()
    assert mixed.ts_shared_postscan_mixer.latent_to_affine.weight.grad is not None
    assert float(mixed.ts_shared_postscan_mixer.latent_to_affine.weight.grad.abs().sum()) > 0.0
    with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
        target = mamba_ts_v3.Model(_config(
            AutoCon=True, enc_in=17, c_out=17, mamba_ts_context_rank=4,
            mamba_ts_shared_postscan_mixer=True,
            mamba_ts_shared_postscan_mixer_rank=3,
        ))
    assert target.ts_shared_postscan_mixer.summary_to_latent.weight.shape == mixed.ts_shared_postscan_mixer.summary_to_latent.weight.shape


def test_postscan_mixer_keys_use_transferable_module_prefix():
    model = mamba_ts_v3.Model(_config(
        AutoCon=True, mamba_ts_context_rank=4,
        mamba_ts_shared_postscan_mixer=True,
    ))
    assert any(key.startswith("ts_shared_postscan_mixer.") for key in model.state_dict())
