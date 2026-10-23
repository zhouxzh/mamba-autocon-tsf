"""CPU contracts for the low-rank data-dependent channel mixer."""

import pytest
import torch

from layers.LowRankDataDependentChannelMixer import (
    LowRankDataDependentChannelMixer,
    LowRankGlobalChannelStateMixer,
)


@pytest.mark.parametrize(
    "channels,patches,dim",
    [(1, 1, 3), (3, 2, 4), (862, 1, 2)],
)
def test_mixer_preserves_shape_and_has_finite_gradients(channels, patches, dim):
    torch.manual_seed(1200 + channels)
    mixer = LowRankDataDependentChannelMixer(
        channels=channels, dim=dim, rank=8, dropout=0.0
    )
    # Open the residual for this gradient check; production initialization is
    # covered separately by the exact ReZero identity test below.
    with torch.no_grad():
        mixer.residual_scale.fill_(0.5)
    tokens = torch.randn(2, channels, patches, dim, requires_grad=True)

    mixed, residual = mixer(tokens, return_residual=True)
    assert mixed.shape == tokens.shape
    assert residual.shape == tokens.shape
    assert torch.isfinite(mixed).all()
    assert torch.isfinite(residual).all()

    mixed.square().mean().backward()
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()
    for parameter in mixer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_rezero_initialization_is_an_exact_identity():
    torch.manual_seed(1204)
    mixer = LowRankDataDependentChannelMixer(channels=4, dim=5, rank=3)
    tokens = torch.randn(2, 4, 3, 5)

    assert mixer.residual_scale.item() == 0.0
    mixed, residual = mixer(tokens, return_residual=True)
    assert torch.equal(mixed, tokens)
    assert torch.equal(residual, torch.zeros_like(tokens))


def test_nonzero_initial_scale_opens_affine_gradient_path():
    torch.manual_seed(1205)
    mixer = LowRankDataDependentChannelMixer(
        channels=4, dim=5, rank=3, init_scale=0.01
    )
    tokens = torch.randn(2, 4, 1, 5, requires_grad=True)
    assert mixer.residual_scale.item() == pytest.approx(0.01)
    mixer(tokens).square().mean().backward()
    assert mixer.summary_to_latent.weight.grad is not None
    assert mixer.latent_to_affine.weight.grad is not None
    assert mixer.summary_to_latent.weight.grad.abs().sum().item() > 0.0
    assert mixer.latent_to_affine.weight.grad.abs().sum().item() > 0.0


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan")])
def test_initial_scale_is_bounded(value):
    with pytest.raises(ValueError, match="init_scale"):
        LowRankDataDependentChannelMixer(channels=3, dim=4, init_scale=value)


def test_nonzero_residual_uses_other_channels_data():
    """A channel's output must react to a different channel's window."""

    mixer = LowRankDataDependentChannelMixer(channels=3, dim=2, rank=2)
    # Make the dependency deterministic: channel 0's mean drives the latent
    # and that latent produces a shift for channel 1.
    with torch.no_grad():
        mixer.summary_to_latent.weight.zero_()
        mixer.summary_to_latent.bias.zero_()
        mixer.summary_to_latent.weight[0, 0] = 1.0
        mixer.latent_to_affine.weight.zero_()
        mixer.latent_to_affine.bias.zero_()
        mixer.latent_to_affine.weight[mixer.channels + 1, 0] = 1.0
        mixer.residual_scale.fill_(1.0)

    baseline = torch.zeros(1, 3, 1, 2)
    changed = baseline.clone()
    changed[:, 0] = 1.0
    baseline_out = mixer(baseline)
    changed_out = mixer(changed)

    channel_one_delta = changed_out[:, 1] - baseline_out[:, 1]
    assert channel_one_delta.abs().max().detach().item() > 1e-4


def test_rank_is_capped_below_channel_count_without_dense_channel_matrix():
    mixer = LowRankDataDependentChannelMixer(channels=5, dim=4, rank=99)
    assert mixer.rank == 4
    assert mixer.summary_to_latent.out_features == 4
    assert mixer.latent_to_affine.in_features == 4
    # The adapter has no dense channel-to-channel matrix; its only learned
    # channel-sized matrices factor through the bottleneck rank.
    assert all(tuple(parameter.shape) != (5, 5) for parameter in mixer.parameters())

    single = LowRankDataDependentChannelMixer(channels=1, dim=4, rank=99)
    # There is no strictly smaller positive rank for one channel; the module
    # keeps a one-dimensional bottleneck while retaining the same API.
    assert single.rank == 1


def test_invalid_shape_and_rank_fail_loudly():
    with pytest.raises(ValueError, match="rank must be positive"):
        LowRankDataDependentChannelMixer(channels=3, dim=4, rank=0)
    mixer = LowRankDataDependentChannelMixer(channels=3, dim=4, rank=2)
    with pytest.raises(ValueError, match=r"expects \[batch, channels"):
        mixer(torch.randn(2, 3, 4))
    with pytest.raises(ValueError, match="does not match"):
        mixer(torch.randn(2, 3, 1, 5))


def test_global_state_mixer_reuses_summary_and_preserves_hidden_shape():
    torch.manual_seed(1220)
    mixer = LowRankGlobalChannelStateMixer(
        channels=7, dim=16, rank=4, summary_dim=16
    )
    hidden = torch.randn(2, 13, 16)
    summary = torch.randn(2, 16, requires_grad=True)
    output = mixer.forward_from_summary(summary, hidden)
    assert output.shape == hidden.shape
    assert torch.equal(output, hidden)

    with torch.no_grad():
        mixer.residual_scale.fill_(0.5)
    summary_output = mixer.forward_from_summary(summary, hidden)
    assert summary_output.shape == hidden.shape
    assert torch.isfinite(summary_output).all()
    summary_output.square().mean().backward()
    assert summary.grad is not None
    assert torch.isfinite(summary.grad).all()
    for parameter in mixer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_global_state_mixer_has_no_dense_channel_matrix_and_supports_short_state():
    mixer = LowRankGlobalChannelStateMixer(
        channels=862, dim=32, rank=16, summary_dim=32
    )
    assert mixer.rank == 16
    assert mixer.summary_dim == 32
    assert all(
        tuple(parameter.shape) != (862, 862)
        for parameter in mixer.parameters()
    )
    window = torch.randn(1, 96, 862)
    hidden = torch.randn(1, 1, 32)
    with torch.no_grad():
        mixer.residual_scale.fill_(1.0)
    output = mixer(window, hidden)
    assert output.shape == hidden.shape
    assert torch.isfinite(output).all()
