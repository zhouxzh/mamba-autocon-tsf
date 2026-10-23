"""Contract tests for the dependency-free inverted Transformer baseline."""

from types import SimpleNamespace
import unittest

import torch

from models.iTransformer import Model
from scripts.run_training_plan import MODEL_SPECS, _command


def _configs(*, enc_in=4, c_out=None, seq_len=12, pred_len=5, d_model=16):
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=seq_len,
        label_len=max(1, seq_len // 2),
        pred_len=pred_len,
        enc_in=enc_in,
        dec_in=enc_in,
        c_out=enc_in if c_out is None else c_out,
        d_model=d_model,
        n_heads=4,
        e_layers=2,
        d_ff=32,
        dropout=0.0,
    )


class ITransformerContractTests(unittest.TestCase):
    def test_model_is_registered_as_plain_plan_baseline(self):
        self.assertEqual(
            MODEL_SPECS["iTransformer"],
            {"model": "iTransformer", "autocon": False},
        )

    def test_plan_command_routes_to_generic_plain_model(self):
        from pathlib import Path

        job = {
            "phase": "itransformer_smoke",
            "dataset": "ETTh1",
            "model": "iTransformer",
            "pred_len": 96,
            "seed": 2021,
            "autocon_lambda": None,
            "config": {
                "features": "M",
                "enc_in": 7,
                "dec_in": 7,
                "c_out": 7,
                "seq_len": 96,
                "label_len": 48,
                "epochs": 1,
            },
        }
        roots = {
            "checkpoints": Path("/tmp/itr-checkpoints"),
            "results": Path("/tmp/itr-results"),
            "test": Path("/tmp/itr-test"),
            "result_log": Path("/tmp/itr-results.txt"),
        }
        _, command = _command(job, roots)
        self.assertIn("--model", command)
        self.assertEqual(command[command.index("--model") + 1], "iTransformer")
        self.assertNotIn("--AutoCon", command)

    def test_multivariate_forecast_shape_and_gradients(self):
        model = Model(_configs(enc_in=4, c_out=4))
        x = torch.randn(3, 12, 4)
        output = model(x, None, None, None)
        self.assertEqual(tuple(output.shape), (3, 5, 4))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all(parameter.grad is not None for parameter in trainable))

    def test_ms_forecast_projects_to_target_channel(self):
        model = Model(_configs(enc_in=6, c_out=1, d_model=12))
        x = torch.randn(2, 12, 6)
        output = model(x, None, None, None)
        self.assertEqual(tuple(output.shape), (2, 5, 1))
        self.assertTrue(torch.isfinite(output).all())

    def test_single_channel_forecast_shape(self):
        model = Model(_configs(enc_in=1, c_out=1, d_model=8))
        output = model(torch.randn(2, 12, 1), None, None, None)
        self.assertEqual(tuple(output.shape), (2, 5, 1))

    def test_input_contract_errors_are_explicit(self):
        model = Model(_configs(enc_in=3, c_out=3))
        with self.assertRaisesRegex(ValueError, "input length"):
            model(torch.randn(2, 11, 3), None, None, None)
        with self.assertRaisesRegex(ValueError, "input channels"):
            model(torch.randn(2, 12, 2), None, None, None)

    def test_head_count_must_divide_model_width(self):
        config = _configs(d_model=10)
        config.n_heads = 4
        with self.assertRaisesRegex(ValueError, "divisible"):
            Model(config)


if __name__ == "__main__":
    unittest.main()
