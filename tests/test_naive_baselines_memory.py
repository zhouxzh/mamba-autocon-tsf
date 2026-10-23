"""Memory-safety contracts for high-dimensional naive baselines."""

import unittest

from scripts.compute_naive_baselines import (
    DEFAULT_HIGH_DIM_MAX_CHUNK_BYTES,
    DEFAULT_LOW_DIM_CHUNK_SIZE,
    estimated_peak_bytes_per_window,
    resolve_chunk_size,
)


class NaiveBaselineChunkSizingTests(unittest.TestCase):
    def test_traffic_h720_uses_a_small_memory_safe_chunk(self):
        bytes_per_window = estimated_peak_bytes_per_window(
            seq_len=96, horizon=720, channels=862,
        )
        chunk = resolve_chunk_size(
            "traffic", horizon=720, seq_len=96, channels=862,
        )

        self.assertEqual(chunk, 4)
        self.assertLess(chunk, 32)
        self.assertLessEqual(chunk * bytes_per_window, DEFAULT_HIGH_DIM_MAX_CHUNK_BYTES)
        self.assertGreater((chunk + 1) * bytes_per_window, DEFAULT_HIGH_DIM_MAX_CHUNK_BYTES)

    def test_explicit_chunk_size_has_priority_over_memory_budget(self):
        chunk = resolve_chunk_size(
            "traffic", horizon=720, seq_len=96, channels=862,
            chunk_size=32, max_chunk_bytes=1,
        )

        self.assertEqual(chunk, 32)

    def test_low_dimensional_default_is_preserved(self):
        chunk = resolve_chunk_size(
            "ETTh1", horizon=720, seq_len=96, channels=7,
        )

        self.assertEqual(chunk, DEFAULT_LOW_DIM_CHUNK_SIZE)


if __name__ == "__main__":
    unittest.main()
