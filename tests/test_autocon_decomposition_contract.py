"""Length-preserving moving-average contracts for AutoCon forecasting heads."""

import unittest

import torch

from models.AutoConCI import moving_avg as CIMovingAvg
from models.AutoConNet import moving_avg as NetMovingAvg
from models.Mamba2AutoCon import moving_avg as MambaMovingAvg


class AutoConDecompositionContractTests(unittest.TestCase):
    def test_even_kernels_preserve_sequence_length(self):
        values = torch.arange(2 * 7 * 3, dtype=torch.float32).reshape(2, 7, 3)
        for moving_avg in (NetMovingAvg, CIMovingAvg, MambaMovingAvg):
            with self.subTest(moving_avg=moving_avg.__module__):
                output = moving_avg(kernel_size=4, stride=1)(values)
                self.assertEqual(tuple(output.shape), tuple(values.shape))

    def test_odd_kernel_values_keep_the_existing_symmetric_padding(self):
        values = torch.randn(2, 7, 3)
        for moving_avg in (NetMovingAvg, CIMovingAvg, MambaMovingAvg):
            with self.subTest(moving_avg=moving_avg.__module__):
                module = moving_avg(kernel_size=3, stride=1)
                expected = module.avg(
                    torch.cat(
                        [values[:, :1], values, values[:, -1:]], dim=1
                    ).permute(0, 2, 1)
                ).permute(0, 2, 1)
                self.assertTrue(torch.allclose(module(values), expected))


if __name__ == "__main__":
    unittest.main()
