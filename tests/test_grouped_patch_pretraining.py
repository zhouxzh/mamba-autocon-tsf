import numpy as np
import pytest
import torch

from layers.losses import ChannelPatchLagContrastiveLoss
from scripts.pretrain_mambatsv3_core_autocon import (
    _acf_grouped_patch_prior,
    _grouped_patch_representation,
)


def test_train_prefix_acf_grouping_is_balanced_and_pooled_states_are_differentiable():
    rng = np.random.default_rng(19)
    train = rng.standard_normal((128, 8), dtype=np.float32)
    peaks = np.tile(np.array([[2, 4]], dtype=np.int64), (8, 1))
    assignment, acf, grouped_peaks = _acf_grouped_patch_prior(train, peaks, 4)
    assert np.bincount(assignment, minlength=4).tolist() == [2, 2, 2, 2]
    assert acf.shape == (4, 128)
    encoded = torch.randn(4, 8, 3, 5, requires_grad=True)
    grouped = _grouped_patch_representation(encoded, assignment, 4)
    assert grouped.shape == (4, 4, 3, 5)
    objective = ChannelPatchLagContrastiveLoss(
        acf, selected_lags=[2], channel_selected_lags=grouped_peaks,
    )
    loss = objective(grouped, torch.tensor([0, 2, 4, 6])).mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert float(encoded.grad.abs().sum()) > 0.0


def test_grouped_patch_pooling_rejects_mismatched_channel_assignment():
    with pytest.raises(ValueError, match="incompatible"):
        _grouped_patch_representation(torch.zeros(2, 3, 2, 4), np.array([0, 1]), 2)
