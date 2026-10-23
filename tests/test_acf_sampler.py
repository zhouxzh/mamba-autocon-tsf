"""Contract tests for relation-aware temporal mini-batching."""

import unittest

import numpy as np

from data_provider.samplers import ACFLagPairBatchSampler


class _SizedDataset:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


class ACFLagPairBatchSamplerTests(unittest.TestCase):
    def test_every_origin_is_seen_once_without_drop_last(self):
        sampler = ACFLagPairBatchSampler(_SizedDataset(37), 8, [3, 7], seed=17)
        batches = list(sampler)
        flat = [index for batch in batches for index in batch]
        self.assertEqual(sorted(flat), list(range(37)))
        self.assertEqual(len(flat), len(set(flat)))
        self.assertTrue(all(1 <= len(batch) <= 8 for batch in batches))
        self.assertEqual(len(sampler), len(batches))

    def test_full_pair_batches_contain_selected_lag_relation(self):
        lags = {4, 9}
        sampler = ACFLagPairBatchSampler(_SizedDataset(64), 8, lags, seed=23)
        batches = list(sampler)
        for batch in batches[:-1]:
            distances = {
                abs(left - right)
                for position, left in enumerate(batch)
                for right in batch[position + 1:]
            }
            self.assertTrue(distances & lags)

    def test_epoch_seed_is_reproducible_and_can_change_order(self):
        first = ACFLagPairBatchSampler(_SizedDataset(41), 6, [5], seed=101)
        second = ACFLagPairBatchSampler(_SizedDataset(41), 6, [5], seed=101)
        self.assertEqual(list(first), list(second))
        first.set_epoch(7)
        second.set_epoch(7)
        self.assertEqual(list(first), list(second))

    def test_balanced_lag_policy_nearly_equalizes_feasible_exposure(self):
        sampler = ACFLagPairBatchSampler(
            _SizedDataset(512), 8, [5, 17], seed=211, lag_balance="balanced"
        )
        list(sampler)
        counts = sampler.last_lag_pair_counts
        self.assertEqual(set(counts), {5, 17})
        self.assertGreater(min(counts.values()), 0)
        # On this symmetric, long dataset both lags are equally feasible.  A
        # least-exposed policy should therefore differ by at most one pair.
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_default_random_policy_is_behaviorally_compatible(self):
        default = ACFLagPairBatchSampler(_SizedDataset(73), 8, [4, 9], seed=19)
        explicit = ACFLagPairBatchSampler(
            _SizedDataset(73), 8, [4, 9], seed=19, lag_balance="random"
        )
        self.assertEqual(list(default), list(explicit))

    def test_invalid_lag_balance_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "lag_balance"):
            ACFLagPairBatchSampler(_SizedDataset(8), 2, [2], lag_balance="all")

    def test_pair_batch_requires_two_examples(self):
        with self.assertRaises(ValueError):
            ACFLagPairBatchSampler(_SizedDataset(8), 1, [2])

    def test_pair_batch_requires_even_batch_size(self):
        with self.assertRaisesRegex(ValueError, "even batch_size"):
            ACFLagPairBatchSampler(_SizedDataset(8), 3, [2])

    def test_pair_batch_rejects_truncated_lag_values(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            ACFLagPairBatchSampler(_SizedDataset(8), 4, [1.5])
        with self.assertRaisesRegex(ValueError, "positive integer"):
            ACFLagPairBatchSampler(_SizedDataset(8), 4, [np.bool_(True)])

    def test_sparse_origins_pair_in_physical_time_units(self):
        origins = [0, 11, 24, 35, 48]
        sampler = ACFLagPairBatchSampler(
            _SizedDataset(len(origins)), 4, [11, 13], origins=origins, seed=71,
        )
        batches = list(sampler)
        pair_distances = {
            abs(origins[left] - origins[right])
            for batch in batches for left, right in zip(batch[::2], batch[1::2])
        }
        self.assertTrue(pair_distances & {11, 13})

    def test_local_packing_keeps_nearby_pairs_in_one_batch(self):
        origins = [0, 144, 288, 432, 2000, 2144, 2288, 2432]
        sampler = ACFLagPairBatchSampler(
            _SizedDataset(len(origins)), 4, [288], origins=origins,
            batch_packing="local", seed=71,
        )
        batches = list(sampler)
        for batch in batches:
            values = [origins[index] for index in batch]
            self.assertLessEqual(max(values) - min(values), 432)


if __name__ == "__main__":
    unittest.main()
