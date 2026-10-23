"""CPU contract tests for the strictly left-padded forecasting embedding."""

import unittest

import torch

from layers.CausalEmbedding import CausalDataEmbedding, CausalTokenEmbedding
from layers.Embed import TimeFeatureEmbedding


class CausalTokenEmbeddingTests(unittest.TestCase):
    def test_shape_and_left_padding(self):
        torch.manual_seed(7)
        module = CausalTokenEmbedding(c_in=3, d_model=5)
        module.eval()
        x = torch.randn(2, 11, 3)
        output = module(x)
        self.assertEqual(tuple(output.shape), (2, 11, 5))

        # The first output token can depend on x[:, 0], but not any later
        # value, including the final value of the window.
        changed = x.clone()
        changed[:, 1:, :] += torch.randn_like(changed[:, 1:, :]) * 100.0
        first_original = module(x)[:, :1]
        first_changed = module(changed)[:, :1]
        self.assertTrue(torch.equal(first_original, first_changed))

    def test_every_output_has_only_a_past_receptive_field(self):
        torch.manual_seed(8)
        module = CausalTokenEmbedding(c_in=2, d_model=4, kernel_size=3)
        module.eval()
        x = torch.randn(1, 9, 2)
        output = module(x)

        for time_index in range(x.shape[1]):
            perturbed = x.clone()
            if time_index + 1 < x.shape[1]:
                perturbed[:, time_index + 1 :, :] += 17.0
            self.assertTrue(
                torch.equal(output[:, time_index], module(perturbed)[:, time_index]),
                msg="output at time {} reads a future value".format(time_index),
            )


class CausalDataEmbeddingTests(unittest.TestCase):
    def test_supports_none_marks_and_dropout(self):
        module = CausalDataEmbedding(
            c_in=3,
            d_model=8,
            embed_type="fixed",
            freq="h",
            dropout=0.25,
        )
        module.eval()
        x = torch.randn(2, 12, 3)
        output = module(x, None)
        self.assertEqual(tuple(output.shape), (2, 12, 8))

    def test_timef_shape_and_marks_are_not_modified(self):
        torch.manual_seed(9)
        module = CausalDataEmbedding(
            c_in=2,
            d_model=6,
            embed_type="timeF",
            freq="h",
            wo_freq="h",
            dropout=0.0,
        )
        module.eval()
        x = torch.randn(2, 10, 2)
        x_mark = torch.randn(2, 10, 4)
        original_mark = x_mark.clone()
        output = module(x, x_mark)
        self.assertEqual(tuple(output.shape), (2, 10, 6))
        self.assertTrue(torch.equal(x_mark, original_mark))


class TimeFeatureEmbeddingTests(unittest.TestCase):
    def test_wo_freq_does_not_modify_input_marks(self):
        module = TimeFeatureEmbedding(
            d_model=6,
            embed_type="timeF",
            freq="h",
            wo_freq="h",
        )
        marks = torch.randn(2, 8, 4)
        original_marks = marks.clone()

        output = module(marks)

        self.assertEqual(tuple(output.shape), (2, 8, 6))
        self.assertTrue(torch.equal(marks, original_marks))

    def test_wo_freq_does_not_pollute_reused_mark_view(self):
        """A pooled branch must not zero marks consumed by the full branch."""

        torch.manual_seed(13)
        module = TimeFeatureEmbedding(
            d_model=5,
            embed_type="timeF",
            freq="h",
            wo_freq="h",
        )
        marks = torch.randn(1, 10, 4)
        original_marks = marks.clone()

        # ``[:, 1::2, :]`` mirrors the strided view used by a pooled branch.
        module(marks[:, 1::2, :])
        actual_full = module(marks)
        expected_full = module(original_marks)

        torch.testing.assert_close(actual_full, expected_full)
        self.assertTrue(torch.equal(marks, original_marks))


if __name__ == "__main__":
    unittest.main()
