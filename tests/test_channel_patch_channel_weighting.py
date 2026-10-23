import numpy as np
import pytest
import torch

from layers.losses import ChannelPatchLagContrastiveLoss


def test_channel_patch_strength_weights_change_only_channel_aggregation():
    torch.manual_seed(17)
    acf = np.zeros((2, 8), dtype=np.float32)
    acf[:, 2] = (.8, .6)
    representation = torch.randn(4, 2, 3, 5)
    origins = torch.tensor([0, 2, 4, 6])
    common = dict(acf_values=acf, selected_lags=[2], channel_selected_lags=[[2], [2]])
    uniform = ChannelPatchLagContrastiveLoss(**common, channel_weights=[1., 1.])
    first_channel = ChannelPatchLagContrastiveLoss(**common, channel_weights=[1., 0.])
    uniform_value = uniform(representation, origins)
    weighted_value = first_channel(representation, origins)
    assert torch.isfinite(uniform_value).all()
    assert torch.isfinite(weighted_value).all()
    assert not torch.allclose(uniform_value, weighted_value)


def test_channel_patch_strength_weights_validate_channel_dimension_and_mass():
    acf = np.zeros((2, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="match the ACF channel count"):
        ChannelPatchLagContrastiveLoss(acf, selected_lags=[2], channel_weights=[1.])
    with pytest.raises(ValueError, match="positive mass"):
        ChannelPatchLagContrastiveLoss(acf, selected_lags=[2], channel_weights=[0., 0.])
