"""Deterministic channel-order utilities derived from a train-only ACF table."""

from __future__ import annotations

import numpy as np


def signed_acf_profile_similarity_order(values):
    """Return a greedy order for signed ACF profiles.

    ``values`` is a ``[channels, selected_lags]`` table.  Profiles are
    L2-normalized and compared with signed cosine similarity.  The path starts
    at the strongest profile (stable lowest-index tie break), then appends the
    remaining channel most similar to the current endpoint (again using
    ``argmax``'s stable lowest-index tie break).  The implementation performs
    vectorized ``O(C^2 K)`` dot products but stores only ``O(CK)`` normalized
    profiles; no dense ``C x C`` matrix is materialized.

    The function is intentionally stateless: callers must provide a table
    fitted on the training split and persist the resulting order with the
    experiment provenance.  It never inspects runtime windows or targets.
    """

    profiles = np.asarray(values, dtype=np.float32)
    if profiles.ndim != 2 or profiles.shape[0] <= 0 or profiles.shape[1] <= 0:
        raise ValueError("ACF profiles must have shape [channels, selected_lags]")
    if not np.isfinite(profiles).all():
        raise ValueError("ACF profiles must contain only finite values")

    norms = np.linalg.norm(profiles, axis=1)
    safe_norms = np.maximum(norms, np.finfo(np.float32).eps)
    normalized = profiles / safe_norms[:, None]
    remaining = np.ones(profiles.shape[0], dtype=bool)
    start = int(np.argmax(norms))
    order = [start]
    remaining[start] = False
    current = start
    for _ in range(profiles.shape[0] - 1):
        scores = normalized @ normalized[current]
        scores[~remaining] = -np.inf
        next_channel = int(np.argmax(scores))
        order.append(next_channel)
        remaining[next_channel] = False
        current = next_channel
    return np.asarray(order, dtype=np.int64)
