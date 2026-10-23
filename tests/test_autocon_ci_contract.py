"""Regression tests for AutoConCI channel-independent stream alignment."""

from types import SimpleNamespace
import unittest

import torch

from models.AutoConCI import Model


def _args(auto_con=False):
    return SimpleNamespace(
        seq_len=8,
        label_len=4,
        pred_len=4,
        enc_in=3,
        c_out=3,
        d_model=4,
        d_ff=4,
        e_layers=1,
        AutoCon=auto_con,
        AutoCon_wnorm="LastVal",
        AutoCon_multiscales=[2],
        embed="timeF",
        freq="h",
        dropout=0.0,
    )


class AutoConCIContractTests(unittest.TestCase):
    def test_meanstd_uses_patchtst_population_statistics_and_roundtrips(self):
        args = _args()
        args.AutoCon_wnorm = "MeanStd"
        model = Model(args).eval()
        x_enc = torch.randn(2, args.seq_len, args.enc_in)
        normalized, _, restore = model._normalize_window(x_enc)
        mean = x_enc.mean(dim=1, keepdim=True)
        centered = x_enc - mean
        std = torch.sqrt(centered.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        self.assertTrue(torch.allclose(normalized, centered / std))
        restored = normalized * restore[2] + restore[1]
        self.assertTrue(torch.allclose(restored, x_enc, atol=1e-6, rtol=1e-6))

    def test_temporal_marks_follow_batch_major_channel_flattening(self):
        """Each flattened (batch, channel) stream gets its own batch marks."""

        args = _args()
        model = Model(args).eval()
        batch, channels, steps, mark_dims = 2, args.enc_in, args.seq_len, 4
        x_enc = torch.randn(batch, steps, channels)
        # Every batch/time entry is unique, making a cross-batch permutation
        # observable even when the model itself is randomly initialized.
        x_mark_enc = torch.arange(
            batch * steps * mark_dims, dtype=torch.float32
        ).reshape(batch, steps, mark_dims)
        x_dec = torch.zeros(batch, args.label_len + args.pred_len, channels)
        x_mark_dec = torch.zeros(
            batch, args.label_len + args.pred_len, mark_dims
        )

        observed = []

        def capture_marks(_module, inputs):
            observed.append(inputs[1].detach().clone())

        handle = model.enc_embedding.register_forward_pre_hook(capture_marks)
        try:
            with torch.no_grad():
                prediction = model(
                    x_enc, x_mark_enc, x_dec, x_mark_dec
                )
        finally:
            handle.remove()

        self.assertEqual(len(observed), 1)
        expected = x_mark_enc.repeat_interleave(channels, dim=0)
        self.assertTrue(torch.equal(observed[0], expected))
        self.assertEqual(
            tuple(prediction.shape), (batch, args.pred_len, channels)
        )

    def test_projection_is_used_only_for_autocon_representation(self):
        args = _args(auto_con=True)
        model = Model(args).eval()
        x_enc = torch.randn(2, args.seq_len, args.enc_in)
        x_mark = torch.randn(2, args.seq_len, 4)
        x_dec = torch.zeros(2, args.label_len + args.pred_len, args.enc_in)
        x_dec_mark = torch.randn(2, args.label_len + args.pred_len, 4)

        prediction, returned = model(x_enc, x_mark, x_dec, x_dec_mark)
        _, only_repr = model(x_enc, x_mark, x_dec, x_dec_mark, onlyrepr=True)
        embeddings = model.get_embeddings(x_enc, x_mark, x_dec, x_dec_mark)
        self.assertTrue(torch.allclose(returned, only_repr))
        self.assertTrue(torch.allclose(returned, embeddings))

        returned.square().mean().backward()
        self.assertIsNotNone(model.repr_head.weight.grad)
        self.assertTrue(torch.isfinite(model.repr_head.weight.grad).all())
        self.assertEqual(tuple(prediction.shape), (2, args.pred_len, args.enc_in))


if __name__ == "__main__":
    unittest.main()
