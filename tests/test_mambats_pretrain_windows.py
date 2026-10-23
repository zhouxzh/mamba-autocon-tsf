"""Contracts for relation-preserving source-window selection."""

import importlib.util
from pathlib import Path

import numpy as np


def _windows_class():
    path = Path("scripts/pretrain_mambatsv3_core_autocon.py")
    spec = importlib.util.spec_from_file_location("mambats_pretrain", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Windows


def test_relation_window_selection_keeps_complete_lag_pairs():
    windows = _windows_class()
    starts = np.arange(1000, dtype=np.int64)
    selected = windows._select_starts(starts, max_windows=64, relation_lags=[288, 576])
    members = set(selected.tolist())
    assert len(selected) == 64
    assert any((origin + lag) in members for origin in members for lag in (288, 576))


def test_relation_window_selection_uses_actual_window_origins():
    windows = _windows_class()
    starts = np.arange(100, 1100, dtype=np.int64)
    selected = windows._select_starts(starts, max_windows=32, relation_lags=[144])
    assert selected.min() >= starts.min()
    assert selected.max() <= starts.max()
    members = set(selected.tolist())
    assert any((origin + 144) in members for origin in members)


def test_relation_window_selection_interleaves_small_and_large_lags():
    windows = _windows_class()
    starts = np.arange(1200, dtype=np.int64)
    selected = windows._select_starts(
        starts, max_windows=64, relation_lags=[1, 24, 288], pair_policy="round_robin"
    )
    members = set(selected.tolist())
    # A lag-sorted pass fills all 32 pairs with lag 1.  The balanced selector
    # must retain evidence for slower periodic positives too.
    assert any((origin + 24) in members for origin in members)
    assert any((origin + 288) in members for origin in members)


def test_relation_window_selection_defaults_to_legacy_contract():
    windows = _windows_class()
    starts = np.arange(1200, dtype=np.int64)
    implicit = windows._select_starts(starts, max_windows=64, relation_lags=[1, 24, 288])
    explicit = windows._select_starts(starts, max_windows=64, relation_lags=[1, 24, 288], pair_policy="legacy")
    assert np.array_equal(implicit, explicit)


def test_forecast_contrastive_coverage_pairing_discards_rare_channel_lags():
    path = Path("scripts/pretrain_mambatsv3_core_autocon.py")
    spec = importlib.util.spec_from_file_location("mambats_pretrain_coverage", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    peaks = np.array([
        [288, 576, 2],
        [288, 576, 3],
        [288, 576, 4],
        [288, 5, 6],
    ])
    selected, support = module._coverage_pair_lags(peaks, min_support=0.5)
    assert selected == [288, 576]
    assert support[288] == 1.0
    assert support[576] == 0.75


def test_coverage_restriction_removes_rare_channel_only_positive_lags():
    path = Path("scripts/pretrain_mambatsv3_core_autocon.py")
    spec = importlib.util.spec_from_file_location("mambats_pretrain_restriction", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    peaks = np.asarray([[24, 48, 0], [24, 72, 0], [24, 96, 0], [48, 0, 0]])
    restricted = module._restrict_channel_peaks(peaks, [24, 48])
    assert restricted.tolist() == [[24, 48, 0], [24, 0, 0], [24, 0, 0], [48, 0, 0]]


def test_coverage_restriction_rejects_empty_channelwise_positive_set():
    path = Path("scripts/pretrain_mambatsv3_core_autocon.py")
    spec = importlib.util.spec_from_file_location("mambats_pretrain_empty_restriction", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        module._restrict_channel_peaks(np.asarray([[24], [48]]), [72])
    except ValueError as exc:
        assert "removed every" in str(exc)
    else:
        raise AssertionError("expected an empty coverage restriction to fail")


def test_channel_peak_stability_weights_are_train_only_and_shape_aligned():
    path = Path("scripts/pretrain_mambatsv3_core_autocon.py")
    spec = importlib.util.spec_from_file_location("mambats_pretrain_stability", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = np.stack([np.sin(np.arange(200) * 2 * np.pi / 12)] * 2, axis=1).astype(np.float32)
    peaks = np.array([[12, 0], [12, 0]], dtype=np.int64)
    weights = module._channel_peak_stability_weights(values, peaks)
    assert weights.shape == peaks.shape
    assert np.all(weights[:, 1] == 0)
    assert np.all(np.isfinite(weights))
    assert np.all(weights[:, 0] > 0)


def test_next_patch_pretraining_allows_overlapping_patch_coverage():
    path = Path("scripts/pretrain_mambatsv3_core_autocon.py")
    spec = importlib.util.spec_from_file_location("mambats_pretrain_geometry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._validate_patch_geometry(96, 24, 8, "next_patch")


def test_masked_reconstruction_rejects_overlapping_patch_coverage():
    path = Path("scripts/pretrain_mambatsv3_core_autocon.py")
    spec = importlib.util.spec_from_file_location("mambats_pretrain_mask_geometry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        module._validate_patch_geometry(96, 24, 8, "masked_reconstruction")
    except ValueError as exc:
        assert "non-overlapping" in str(exc)
    else:
        raise AssertionError("expected masked reconstruction overlap to fail")


def test_contrastive_cluster_windows_have_two_pairs_and_local_negatives():
    windows = _windows_class()
    starts = np.arange(10000, dtype=np.int64)
    selected = windows._select_starts(
        starts, max_windows=64, relation_lags=[288, 576],
        pair_policy="contrastive_cluster",
    )
    members = set(selected.tolist())
    clusters = [selected[index:index + 4] for index in range(0, len(selected), 4)]
    assert any(
        len(cluster) == 4 and cluster[-1] - cluster[0] <= 432
        and sum((int(left) + 288) in members for left in cluster) >= 2
        for cluster in clusters
    )
