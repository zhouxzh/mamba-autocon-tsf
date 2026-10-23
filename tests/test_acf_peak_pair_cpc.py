import torch

from layers.losses import ACFPeakPairInfoNCELoss


def test_peak_pair_cpc_masks_unselected_high_acf_lags_and_has_gradients():
    acf = torch.zeros(8)
    acf[2], acf[3], acf[5] = .9, .8, -.7
    objective = ACFPeakPairInfoNCELoss(acf, selected_lags=[2], exclude_abs_acf=.5)
    features = torch.randn(4, 3, 6, requires_grad=True)
    result = objective(features, torch.tensor([0, 2, 3, 5]))
    assert torch.isfinite(result).all()
    assert objective.last_excluded_high_acf_fraction > 0
    result.mean().backward()
    assert float(features.grad.abs().sum()) > 0


def test_peak_pair_cpc_high_acf_mode_increases_positive_coverage():
    acf = torch.zeros(8)
    acf[2], acf[3], acf[5] = .9, .8, -.7
    selected = ACFPeakPairInfoNCELoss(acf, selected_lags=[2], exclude_abs_acf=.5)
    dense = ACFPeakPairInfoNCELoss(acf, selected_lags=[2], exclude_abs_acf=.5, positive_mode="high_acf")
    features = torch.randn(4, 3, 6)
    origins = torch.tensor([0, 2, 3, 5])
    selected(features, origins)
    dense(features, origins)
    assert dense.last_active_anchor_fraction >= selected.last_active_anchor_fraction
