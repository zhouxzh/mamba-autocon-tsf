"""Batch samplers for relation-aware time-series objectives.

The ordinary shuffled DataLoader is a good default for forecasting, but it can
make an all-pair temporal relation loss almost empty when its ACF table is
compact.  ``ACFLagPairBatchSampler`` keeps the same sample budget while
placing observed windows separated by selected training lags in the same
mini-batch.  It is intentionally opt-in and is not used by historical
AutoCon experiments.
"""

from __future__ import annotations

import math
import random
from typing import Iterable, List, Sequence

import numpy as np
from torch.utils.data import Sampler


def _strict_positive_int(value, label):
    """Parse sampler geometry without silently truncating decimal values."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError("{} must be a positive integer".format(label))
    if isinstance(value, (float, np.floating)) and (
        not math.isfinite(float(value)) or not float(value).is_integer()
    ):
        raise ValueError("{} must be a positive integer".format(label))
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must be a positive integer".format(label)) from exc
    if not isinstance(value, (str, bytes)) and parsed != value:
        raise ValueError("{} must be a positive integer".format(label))
    if parsed <= 0:
        raise ValueError("{} must be a positive integer".format(label))
    return parsed


class ACFLagPairBatchSampler(Sampler[List[int]]):
    """Yield batches containing disjoint pairs at selected temporal lags.

    Every index is yielded at most once per sampler iteration (unless the
    caller chooses to use a non-standard ``__len__``/dataset).  Pair matching
    is performed on the dataset-local window origin, so no target values are
    accessed.  The random order changes on every iteration, while the seed
    makes the sequence reproducible for a fixed epoch counter.

    Parameters
    ----------
    data_source:
        Dataset or an object implementing ``__len__``.
    batch_size:
        Number of window origins per batch.
    lags:
        Positive temporal distances that should be paired when possible.
    drop_last:
        Whether to discard a final incomplete batch.
    seed:
        Base seed for deterministic matching.
    lag_balance:
        ``'random'`` preserves the historical random selected-lag order.
        ``'balanced'`` is an opt-in least-exposed-lag policy: when several
        unused partners are possible for an anchor, it prefers the lag with
        the fewest pairs accepted so far.  It improves multi-lag exposure but
        cannot guarantee exact equality because pairs are disjoint and edge
        availability differs by lag.
    """

    def __init__(
        self,
        data_source,
        batch_size: int,
        lags: Sequence[int] | Iterable[int],
        drop_last: bool = False,
        seed: int = 0,
        lag_balance: str = "random",
        origins: Sequence[int] | None = None,
        batch_packing: str = "random",
    ) -> None:
        self.num_samples = int(len(data_source))
        self.batch_size = _strict_positive_int(batch_size, "batch_size")
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        if self.num_samples < 0:
            raise ValueError("dataset length must be non-negative")
        self.origins = None
        self._origin_index = None
        if origins is not None:
            parsed_origins = np.asarray(list(origins), dtype=np.int64)
            if parsed_origins.ndim != 1 or len(parsed_origins) != self.num_samples:
                raise ValueError("origins must have one value for each dataset item")
            if len(parsed_origins) and (np.diff(parsed_origins) <= 0).any():
                raise ValueError("origins must be strictly increasing and unique")
            self.origins = tuple(int(value) for value in parsed_origins)
            self._origin_index = {origin: index for index, origin in enumerate(self.origins)}
        if self.batch_size < 2:
            raise ValueError("ACF lag-pair batching requires batch_size >= 2")
        # A matched pair is indivisible.  With an odd capacity, a greedy
        # pairing pass can produce ``batch_size - 1`` origins; under
        # ``drop_last=True`` that silently discards an otherwise valid pair
        # batch and makes ``__len__`` disagree with iteration.  Require the
        # explicit even contract until a sampler with a different packing
        # policy is introduced.
        if self.batch_size % 2:
            raise ValueError(
                "ACF lag-pair batching requires an even batch_size so matched "
                "pairs remain intact"
            )
        parsed = []
        for lag in lags:
            parsed.append(_strict_positive_int(lag, "lag"))
        parsed = sorted(set(parsed))
        # A one-step fallback keeps the sampler useful when an ACF peak list
        # is empty or all requested periods exceed a short training split.
        self.lags = tuple(parsed or [1])
        self.lag_balance = str(lag_balance).lower()
        if self.lag_balance not in {"random", "balanced"}:
            raise ValueError("lag_balance must be 'random' or 'balanced'")
        self.batch_packing = str(batch_packing).lower()
        if self.batch_packing not in {"random", "local"}:
            raise ValueError("batch_packing must be 'random' or 'local'")
        # Telemetry for the latest iteration.  This is intentionally not used
        # as training state and does not affect the default sampler path.
        self.last_lag_pair_counts = {lag: 0 for lag in self.lags}
        self._iteration = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the deterministic epoch counter used by the next iteration."""

        self._iteration = int(epoch)

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size
        return math.ceil(self.num_samples / self.batch_size)

    def _candidate_partners(self, index: int, lag: int):
        """Return both directions for a lag without leaving the dataset."""

        if self.origins is not None:
            origin = self.origins[index]
            return [
                partner for partner in (
                    self._origin_index.get(origin + lag),
                    self._origin_index.get(origin - lag),
                ) if partner is not None
            ]
        candidates = []
        forward = index + lag
        backward = index - lag
        if forward < self.num_samples:
            candidates.append(forward)
        if backward >= 0:
            candidates.append(backward)
        return candidates

    def _make_pairs(self, rng: random.Random):
        order = list(range(self.num_samples))
        rng.shuffle(order)
        available = bytearray(b"\x01") * self.num_samples
        pairs = []
        lag_pair_counts = {lag: 0 for lag in self.lags}

        for anchor in order:
            if not available[anchor]:
                continue
            lags = list(self.lags)
            rng.shuffle(lags)
            if self.lag_balance == "balanced":
                # Keep the shuffled order as a deterministic tie-breaker.
                # Python's sort is stable, so equally exposed lags retain it.
                lags.sort(key=lambda lag: lag_pair_counts[lag])
            partner = None
            selected_lag = None
            for lag in lags:
                candidates = self._candidate_partners(anchor, lag)
                rng.shuffle(candidates)
                for candidate in candidates:
                    if available[candidate]:
                        partner = candidate
                        selected_lag = lag
                        break
                if partner is not None:
                    break
            if partner is None:
                # Leave an unmatched origin for the residual stream below;
                # another origin may still use it as a partner.
                continue
            available[anchor] = 0
            available[partner] = 0
            pairs.append((anchor, partner))
            lag_pair_counts[selected_lag] += 1

        leftovers = [index for index, flag in enumerate(available) if flag]
        rng.shuffle(leftovers)
        rng.shuffle(pairs)
        self.last_lag_pair_counts = lag_pair_counts
        return pairs, leftovers

    def __iter__(self):
        rng = random.Random(self.seed + 1000003 * self._iteration)
        self._iteration += 1
        pairs, leftovers = self._make_pairs(rng)
        if self.batch_packing == "local" and self.origins is not None:
            # Forecast InfoNCE needs a positive and a denominator negative
            # inside one batch. Keep temporally nearby matched pairs together
            # so an ACF lookup remains in range for cross-pair negatives.
            pairs.sort(
                key=lambda pair: (
                    self.origins[pair[0]] + self.origins[pair[1]],
                    pair[0], pair[1],
                )
            )
        pair_capacity = max(1, self.batch_size // 2)
        batches = []

        # Keep each matched pair together.  The final pair batch can be filled
        # with unmatched origins without introducing duplicates.
        pair_cursor = 0
        leftover_cursor = 0
        while pair_cursor < len(pairs):
            batch = []
            stop = min(len(pairs), pair_cursor + pair_capacity)
            for anchor, partner in pairs[pair_cursor:stop]:
                batch.extend((anchor, partner))
            pair_cursor = stop
            while len(batch) < self.batch_size and leftover_cursor < len(leftovers):
                batch.append(leftovers[leftover_cursor])
                leftover_cursor += 1
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)

        while leftover_cursor < len(leftovers):
            batch = leftovers[leftover_cursor:leftover_cursor + self.batch_size]
            leftover_cursor += len(batch)
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)

        if self.batch_packing == "local":
            rng.shuffle(batches)

        # A zero-length dataset naturally yields no batches.  For normal
        # datasets, assert the sampler's central invariant during development;
        # this catches accidental duplicate or dropped origins early.
        flattened = [index for batch in batches for index in batch]
        if not self.drop_last and len(flattened) != len(set(flattened)):
            raise RuntimeError("ACF lag sampler produced duplicate origins")
        for batch in batches:
            yield batch


__all__ = ["ACFLagPairBatchSampler"]
