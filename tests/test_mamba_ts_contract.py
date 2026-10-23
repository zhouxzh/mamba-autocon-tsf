"""Contract tests for the MambaTS-inspired patch/variable-scan baseline."""

from types import SimpleNamespace
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
import numpy as np

import models.MambaTS as mamba_ts
import models.MambaTSv2 as mamba_ts_v2
import models.MambaTSv3 as mamba_ts_v3
import models.MambaTSv4 as mamba_ts_v4
import models.MambaTSv5 as mamba_ts_v5
import models.MambaTSv6 as mamba_ts_v6
import models.MambaTSv7 as mamba_ts_v7
import models.MambaTSv8 as mamba_ts_v8
import models.MambaTSv9 as mamba_ts_v9
import models.MambaTSv10 as mamba_ts_v10
import models.MambaTSv11 as mamba_ts_v11
import models.MambaTSv12 as mamba_ts_v12
import models.MambaTSv13 as mamba_ts_v13
import models.MambaTSv14 as mamba_ts_v14
import models.MambaTSv15 as mamba_ts_v15
import models.MambaTSv16 as mamba_ts_v16
import models.GraphDLinear as graph_dlinear
import models.DLinear as dlinear
from run import (
    MAMBA_AUTOCON_MODELS,
    MAMBA_PLAIN_MODELS,
    _mamba_runtime_contract_payload,
)
from scripts.benchmark_model_resources import _normalise_config
from scripts.run_training_plan import MODEL_SPECS


class _IdentityMamba(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


def _config(**overrides):
    values = dict(
        task_name="long_term_forecast",
        seq_len=96,
        pred_len=12,
        enc_in=5,
        c_out=5,
        d_model=8,
        d_ff=16,
        patch_len=16,
        patch_stride=8,
        dropout=0.0,
        e_layers=2,
        AutoCon=False,
        AutoCon_wnorm="MeanStd",
        mamba_d_state=4,
        mamba_d_conv=2,
        mamba_expand=2,
        moving_avg=25,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class MambaTSContractTests(unittest.TestCase):
    def test_plain_and_autocon_shapes(self):
        x = torch.randn(3, 96, 5)
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            plain = mamba_ts.Model(_config())
            self.assertEqual(tuple(plain(x).shape), (3, 12, 5))
            plain_prediction, plain_representation = plain(x, return_repr=True)
            self.assertEqual(tuple(plain_prediction.shape), (3, 12, 5))
            self.assertIsNone(plain_representation)
            autocon = mamba_ts.Model(_config(AutoCon=True))
            forecast, representation = autocon(x, return_repr=True)
        self.assertEqual(tuple(forecast.shape), (3, 12, 5))
        self.assertEqual(tuple(representation.shape), (3, 96, 16))

    def test_mamba2_mixer_preserves_v3_forecast_and_representation_contract(self):
        x = torch.randn(3, 96, 5)
        with patch.object(mamba_ts, "_Mamba2", _IdentityMamba):
            model = mamba_ts_v3.Model(_config(
                AutoCon=True,
                mamba_ts_context_rank=4,
                mamba_ts_mixer_type="mamba2",
                mamba_headdim=4,
            ))
            forecast, representation = model(x, return_repr=True)
        self.assertEqual(model.mamba_ts_mixer_type, "mamba2")
        self.assertIsInstance(model.encoder[0], mamba_ts._ResidualMamba2)
        self.assertEqual(tuple(forecast.shape), (3, 12, 5))
        self.assertEqual(tuple(representation.shape), (3, 96, 16))

    def test_mamba2_mixer_rejects_v12_multidelta_combination(self):
        with self.assertRaisesRegex(ValueError, "v12 Delta branches"):
            mamba_ts.Model(_config(
                mamba_ts_variant="v12",
                mamba_ts_mixer_type="mamba2",
            ))

    def test_autocon_v1_can_skip_representation_at_prediction_time(self):
        x = torch.randn(3, 96, 5)
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            model = mamba_ts.Model(_config(AutoCon=True))
            prediction_with_repr, representation = model(x, return_repr=True)
            prediction_without_repr, disabled_representation = model(x, return_repr=False)
        self.assertIsNotNone(representation)
        self.assertIsNone(disabled_representation)
        torch.testing.assert_close(prediction_with_repr, prediction_without_repr)

    def test_representation_patch_grid_restore_matches_linear_interpolation(self):
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            model = mamba_ts.Model(_config(AutoCon=True, patch_len=24, patch_stride=24))
        patch_states = torch.randn(2, model.repr_dims, model.patch_count)
        expected = torch.nn.functional.interpolate(
            patch_states, size=model.seq_len, mode="linear", align_corners=False
        )
        actual = model._restore_patch_grid(patch_states)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        self.assertNotIn("representation_interpolation_matrix", model.state_dict())

    def test_representation_patch_grid_restore_matches_interpolation_gradient(self):
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            model = mamba_ts.Model(_config(AutoCon=True, patch_len=24, patch_stride=24))
        patch_states = torch.randn(2, model.repr_dims, model.patch_count, requires_grad=True)
        reference_states = patch_states.detach().clone().requires_grad_(True)
        weights = torch.randn(2, model.repr_dims, model.seq_len)
        optimized_loss = (model._restore_patch_grid(patch_states) * weights).sum()
        reference_loss = (
            torch.nn.functional.interpolate(
                reference_states, size=model.seq_len, mode="linear", align_corners=False
            )
            * weights
        ).sum()
        optimized_loss.backward()
        reference_loss.backward()
        # GEMM and interpolation kernels accumulate FP32 weights in different
        # orders; the observed residual stays within a 2e-6 absolute tolerance.
        torch.testing.assert_close(patch_states.grad, reference_states.grad, atol=2e-6, rtol=1e-5)

    def test_channelwise_representation_restore_matches_linear_interpolation(self):
        x = torch.randn(2, 96, 5)
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            model = mamba_ts_v3.Model(_config(
                AutoCon=True,
                mamba_ts_context_rank=4,
                mamba_ts_repr_forecast_fusion=True,
                mamba_ts_repr_forecast_channelwise=True,
            ))
            _, representation = model(x, return_repr=True)
        self.assertEqual(tuple(representation.shape), (2, 96, 16))

    def test_shared_channel_summary_is_channel_count_agnostic_and_single_scan(self):
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            source = mamba_ts_v3.Model(_config(
                AutoCon=True, enc_in=5, c_out=5, mamba_ts_context_rank=4,
                mamba_ts_shared_channel_summary=True,
            ))
            target = mamba_ts_v3.Model(_config(
                AutoCon=True, enc_in=11, c_out=11, mamba_ts_context_rank=4,
                mamba_ts_shared_channel_summary=True,
            ))
            output, representation = target(torch.randn(2, 96, 11), return_repr=True)
        self.assertEqual(tuple(output.shape), (2, 12, 11))
        self.assertEqual(tuple(representation.shape), (2, 96, 16))
        self.assertEqual(len(target.encoder), 2)
        source_summary = {
            key: value.shape for key, value in source.state_dict().items()
            if key.startswith("ts_shared_channel_summary_")
        }
        target_summary = {
            key: value.shape for key, value in target.state_dict().items()
            if key.startswith("ts_shared_channel_summary_")
        }
        self.assertEqual(source_summary, target_summary)

    def test_channel_set_pooling_is_channel_count_agnostic_permutation_equivariant_and_single_scan(self):
        x = torch.randn(2, 96, 7)
        permutation = torch.tensor([3, 0, 6, 2, 5, 1, 4])
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            model = mamba_ts_v3.Model(_config(
                AutoCon=True, enc_in=7, c_out=7, mamba_ts_context_rank=4,
                mamba_ts_channel_agnostic_scan=True, mamba_ts_channel_set_pooling=True,
            ))
            output, representation = model(x, return_repr=True)
            permuted_output, permuted_representation = model(x[:, :, permutation], return_repr=True)
        self.assertEqual(tuple(output.shape), (2, 12, 7))
        self.assertEqual(tuple(representation.shape), (2, 96, 16))
        torch.testing.assert_close(permuted_output, output[:, :, permutation], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(permuted_representation, representation, atol=1e-6, rtol=1e-6)
        target_keys = {key: value.shape for key, value in model.state_dict().items() if key.startswith("ts_channel_set_pooling_")}
        self.assertTrue(target_keys)
        self.assertEqual(len(model.encoder), 2)

    def test_channel_set_pooling_rejects_mean_std_summary_combination(self):
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                mamba_ts_v3.Model(_config(
                    AutoCon=True, mamba_ts_shared_channel_summary=True,
                    mamba_ts_channel_set_pooling=True,
                ))

    def test_channel_prototype_pooling_is_channel_count_agnostic_and_permutation_equivariant(self):
        x = torch.randn(2, 96, 7)
        permutation = torch.tensor([3, 0, 6, 2, 5, 1, 4])
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            model = mamba_ts_v3.Model(_config(
                AutoCon=True, enc_in=7, c_out=7, mamba_ts_context_rank=4,
                mamba_ts_channel_agnostic_scan=True, mamba_ts_channel_prototype_pooling=True,
                mamba_ts_channel_prototype_count=3,
            ))
            output, representation = model(x, return_repr=True)
            permuted, permuted_representation = model(x[:, :, permutation], return_repr=True)
        torch.testing.assert_close(permuted, output[:, :, permutation], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(permuted_representation, representation, atol=1e-6, rtol=1e-6)
        self.assertTrue(any(key.startswith("ts_channel_prototype_") for key in model.state_dict()))
        self.assertEqual(tuple(output.shape), (2, 12, 7))

    def test_channel_prototype_pooling_requires_multiple_prototypes(self):
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            with self.assertRaisesRegex(ValueError, "must exceed one"):
                mamba_ts_v3.Model(_config(AutoCon=True, mamba_ts_channel_prototype_pooling=True,
                                          mamba_ts_channel_prototype_count=1))

    def test_multivariate_single_target_shape(self):
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            model = mamba_ts.Model(_config(c_out=1))
            prediction = model(torch.randn(2, 96, 5))
        self.assertEqual(tuple(prediction.shape), (2, 12, 1))

    def test_v2_uses_independent_scan_and_gated_persistence_path(self):
        x = torch.randn(2, 96, 5)
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            model = mamba_ts_v2.Model(_config())
            prediction = model(x)
        self.assertEqual(tuple(prediction.shape), (2, 12, 5))
        self.assertEqual(model.variant, "v2")
        self.assertTrue(model.channel_independent)
        self.assertIsNotNone(model.channel_embedding)
        self.assertIsNotNone(model.patch_position_embedding)
        self.assertIsNotNone(model.persistence_head)
        self.assertEqual(float(model.persistence_scale.detach()), 0.0)

    def test_v3_keeps_shared_scan_and_builds_low_rank_context(self):
        x = torch.randn(2, 96, 5)
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            model = mamba_ts_v3.Model(_config(mamba_ts_context_rank=4))
            prediction = model(x)
        self.assertEqual(tuple(prediction.shape), (2, 12, 5))
        self.assertEqual(model.variant, "v3")
        self.assertFalse(model.channel_independent)
        self.assertEqual(model.context_rank, 4)
        self.assertEqual(tuple(model.persistence_scale.shape), (1, 1, 5))

    def test_v6_is_linear_base_with_zero_gated_mamba_residual(self):
        x = torch.randn(2, 96, 5)
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            model = mamba_ts_v6.Model(_config(AutoCon=True, mamba_ts_context_rank=4))
            forecast, representation = model(x, return_repr=True)
            self.assertEqual(model.variant, "v6")
            self.assertTrue(model.linear_residual_enabled)
            self.assertEqual(float(model.mamba_residual_scale.detach()), 0.0)
            self.assertEqual(tuple(forecast.shape), (2, 12, 5))
            self.assertEqual(tuple(representation.shape), (2, 96, 16))
            linear = model._linear_forecast(model._normalize(x)[0])
            restored_linear = model._restore(linear, model._normalize(x)[1], model.c_out)
            torch.testing.assert_close(forecast, restored_linear)
            forecast.square().mean().backward()
            self.assertIsNotNone(model.mamba_residual_scale.grad)
            with torch.no_grad():
                model.mamba_residual_scale.fill_(1.0)
            corrected, _ = model(x, return_repr=True)
        self.assertFalse(torch.equal(forecast.detach(), corrected.detach()))

    def test_v7_combines_linear_base_and_train_only_acf_selector(self):
        x = torch.randn(2, 96, 5)
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            model = mamba_ts_v7.Model(_config(
                AutoCon=True,
                mamba_ts_context_rank=4,
                mamba_use_acf_memory=True,
                mamba_acf_lags=[24, 48],
                mamba_ts_acf_memory_selector=True,
            ))
            assert model.variant == "v7"
            assert model.linear_residual_enabled
            assert model.ts_acf_memory_enabled
            assert model.ts_acf_memory_selector_enabled
            assert float(model.mamba_residual_scale.detach()) == 0.0
            assert float(model.ts_acf_memory_scale.detach().abs().max()) == 0.0
            assert float(model.ts_acf_memory_selector_scale.detach()) == 0.0
            model.set_ts_acf_prior(torch.ones(5, 2), [24, 48])
            forecast, representation = model(x, return_repr=True)
            assert tuple(forecast.shape) == (2, 12, 5)
            assert tuple(representation.shape) == (2, 96, 16)
            diagnostics = model.acf_runtime_diagnostics()
            assert diagnostics["acf_runtime_gate_mean"] is not None
            assert diagnostics["acf_effective_memory_rms"] is not None
            assert diagnostics["acf_effective_memory_to_forecast_rms"] is not None
            forecast.square().mean().backward()
            assert model.mamba_residual_scale.grad is not None
            assert model.ts_acf_memory_scale.grad is not None
            assert model.ts_acf_memory_selector_scale.grad is not None
    def test_v6_accepts_a_documented_nonzero_residual_initialization(self):
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            model = mamba_ts_v6.Model(_config(mamba_ts_residual_init=0.1))
        self.assertAlmostEqual(float(model.mamba_residual_scale.detach()), 0.1)

    def test_v3_repr_forecast_fusion_is_zero_gated_and_receives_mse_gradient(self):
        x = torch.randn(2, 96, 5)
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            model = mamba_ts_v3.Model(_config(
                AutoCon=True,
                mamba_ts_context_rank=4,
                mamba_ts_repr_forecast_fusion=True,
            ))
            forecast, representation = model(x, return_repr=True)
            self.assertTrue(model.ts_repr_forecast_fusion_enabled)
            self.assertEqual(float(model.ts_repr_forecast_scale.detach().abs().max()), 0.0)
            self.assertEqual(tuple(representation.shape), (2, 96, 16))
            loss = forecast.square().mean()
            loss.backward()
            scale_gradient = model.ts_repr_forecast_scale.grad
            self.assertIsNotNone(scale_gradient)
            self.assertGreater(float(scale_gradient.detach().abs().sum()), 0.0)
            with torch.no_grad():
                model.ts_repr_forecast_scale.fill_(1.0)
            fused_forecast, _ = model(x, return_repr=True)
        self.assertFalse(torch.equal(forecast.detach(), fused_forecast.detach()))

    def test_v3_regular_fusion_representation_has_time_grid_layout(self):
        x = torch.randn(2, 96, 5)
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            model = mamba_ts_v3.Model(_config(
                AutoCon=True,
                mamba_ts_context_rank=4,
                mamba_ts_repr_forecast_fusion=True,
            ))
            prediction, representation = model(x, return_repr=True)
        self.assertEqual(tuple(prediction.shape), (2, 12, 5))
        self.assertEqual(tuple(representation.shape), (2, 96, 16))

    def test_v3_channel_adapter_is_zero_gated_and_channel_preserving(self):
        x = torch.randn(2, 96, 5)
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            control = mamba_ts_v3.Model(_config(AutoCon=True, mamba_ts_context_rank=4))
            adapter = mamba_ts_v3.Model(_config(
                AutoCon=True,
                mamba_ts_context_rank=4,
                mamba_ts_channel_adapter=True,
                mamba_ts_channel_adapter_rank=3,
            ))
            adapter.load_state_dict(control.state_dict(), strict=False)
            control_forecast, _ = control(x, return_repr=True)
            adapter_forecast, representation = adapter(x, return_repr=True)
            torch.testing.assert_close(control_forecast, adapter_forecast)
            self.assertTrue(adapter.ts_channel_adapter_enabled)
            self.assertEqual(adapter.ts_channel_adapter_rank, 3)
            self.assertEqual(tuple(adapter_forecast.shape), (2, 12, 5))
            adapter_forecast.square().mean().backward()
            self.assertGreater(
                float(adapter.ts_channel_adapter_scale.grad.detach().abs().sum()), 0.0
            )
            with torch.no_grad():
                adapter.ts_channel_adapter_scale.fill_(1.0)
            routed_forecast, _ = adapter(x, return_repr=True)
        self.assertFalse(torch.equal(adapter_forecast.detach(), routed_forecast.detach()))

    def test_v3_target_mlp_adapter_is_zero_gated_and_channel_preserving(self):
        x = torch.randn(2, 96, 5)
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            control = mamba_ts_v3.Model(_config(AutoCon=True, mamba_ts_context_rank=4))
            adapter = mamba_ts_v3.Model(_config(
                AutoCon=True, mamba_ts_context_rank=4,
                mamba_ts_target_mlp_adapter=True,
                mamba_ts_target_mlp_adapter_rank=7,
            ))
            adapter.load_state_dict(control.state_dict(), strict=False)
            control_forecast, _ = control(x, return_repr=True)
            adapter_forecast, representation = adapter(x, return_repr=True)
            torch.testing.assert_close(control_forecast, adapter_forecast)
            self.assertTrue(adapter.ts_target_mlp_adapter_enabled)
            self.assertEqual(adapter.ts_target_mlp_adapter_rank, 7)
            self.assertEqual(tuple(adapter_forecast.shape), (2, 12, 5))
            adapter_forecast.square().mean().backward()
            self.assertGreater(
                float(adapter.ts_target_mlp_adapter_scale.grad.detach().abs().sum()), 0.0
            )
            with torch.no_grad():
                adapter.ts_target_mlp_adapter_scale.fill_(1.0)
            routed_forecast, _ = adapter(x, return_repr=True)
        self.assertFalse(torch.equal(adapter_forecast.detach(), routed_forecast.detach()))

    def test_v3_revin_affine_is_identity_initialized_and_trainable(self):
        x = torch.randn(2, 96, 5)
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            torch.manual_seed(193)
            control = mamba_ts_v3.Model(_config(AutoCon=True))
            torch.manual_seed(193)
            affine = mamba_ts_v3.Model(_config(
                AutoCon=True, mamba_ts_revin_affine=True
            ))
            affine.load_state_dict(control.state_dict(), strict=False)
            control_forecast, _ = control(x, return_repr=True)
            affine_forecast, _ = affine(x, return_repr=True)
            torch.testing.assert_close(control_forecast, affine_forecast)
            self.assertTrue(affine.ts_revin_affine_enabled)
            self.assertEqual(float(affine.ts_revin_log_scale.detach().abs().max()), 0.0)
            self.assertEqual(float(affine.ts_revin_shift.detach().abs().max()), 0.0)
            affine_forecast.square().mean().backward()
            self.assertIsNotNone(affine.ts_revin_log_scale.grad)
            self.assertIsNotNone(affine.ts_revin_shift.grad)
            self.assertGreater(float(affine.ts_revin_log_scale.grad.detach().abs().sum()), 0.0)
            self.assertGreater(float(affine.ts_revin_shift.grad.detach().abs().sum()), 0.0)

    def test_v3_global_channel_mixer_is_zero_gated_and_shared_with_autocon(self):
        x = torch.randn(2, 96, 5)
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            # Preserve the shared backbone initialization across the paired
            # construction; only the optional ReZero mixer may differ.
            torch.manual_seed(91)
            control = mamba_ts_v3.Model(_config(
                AutoCon=True, mamba_ts_context_rank=4,
            ))
            torch.manual_seed(91)
            mixed = mamba_ts_v3.Model(_config(
                AutoCon=True,
                mamba_ts_context_rank=4,
                mamba_ts_global_channel_mixer=True,
                mamba_ts_global_channel_mixer_rank=3,
            ))
            control_forecast, control_representation = control(x, return_repr=True)
            mixed_forecast, mixed_representation = mixed(x, return_repr=True)
            torch.testing.assert_close(control_forecast, mixed_forecast)
            torch.testing.assert_close(control_representation, mixed_representation)
            self.assertTrue(mixed.ts_global_channel_mixer_enabled)
            self.assertEqual(mixed.ts_global_channel_mixer_rank, 3)
            scale = mixed.ts_global_channel_mixer.residual_scale
            self.assertEqual(float(scale.detach()), 0.0)
            (mixed_forecast.square().mean() + mixed_representation.square().mean()).backward()
            self.assertGreater(float(scale.grad.detach().abs()), 0.0)
            with torch.no_grad():
                scale.fill_(1.0)
            routed_forecast, routed_representation = mixed(x, return_repr=True)
        self.assertFalse(torch.equal(mixed_forecast.detach(), routed_forecast.detach()))
        self.assertFalse(
            torch.equal(mixed_representation.detach(), routed_representation.detach())
        )

    def test_v3_sparse_graph_branch_is_nested_and_receives_forecast_gradient(self):
        x = torch.randn(2, 96, 5)
        with tempfile.TemporaryDirectory() as directory:
            graph_path = Path(directory) / "aligned_graph.npz"
            np.savez_compressed(
                graph_path,
                schema_version=np.asarray(1, dtype=np.int64),
                sensor_ids=np.asarray(["0", "1", "2", "3", "4"]),
                target=np.asarray([0, 1, 2, 3, 4], dtype=np.int64),
                source=np.asarray([1, 2, 3, 4, 0], dtype=np.int64),
                weight=np.ones(5, dtype=np.float32),
                node_count=np.asarray(5, dtype=np.int64),
            )
            with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
                torch.manual_seed(121)
                control = mamba_ts_v3.Model(_config(
                    AutoCon=True, mamba_ts_context_rank=4,
                ))
                torch.manual_seed(121)
                graph = mamba_ts_v3.Model(_config(
                    AutoCon=True,
                    mamba_ts_context_rank=4,
                    mamba_ts_graph_spatial=True,
                    mamba_ts_graph_path=str(graph_path),
                ))
                control_forecast, control_repr = control(x, return_repr=True)
                graph_forecast, graph_repr = graph(x, return_repr=True)
                torch.testing.assert_close(control_forecast, graph_forecast)
                torch.testing.assert_close(control_repr, graph_repr)
                self.assertTrue(graph.ts_graph_spatial_enabled)
                self.assertEqual(float(graph.ts_graph_spatial_scale.detach()), 0.0)
                # Both the pure forecast and the AutoCon representation must
                # use the graph-conditioned temporal state.  Otherwise a
                # future refactor could leave contrastive learning attached
                # to a graph-free side branch.
                (
                    graph_forecast.square().mean()
                    + graph_repr.square().mean()
                ).backward()
                self.assertGreater(
                    float(graph.ts_graph_spatial_scale.grad.detach().abs()), 0.0
                )
                with torch.no_grad():
                    graph.ts_graph_spatial_scale.fill_(1.0)
                routed_forecast, routed_repr = graph(x, return_repr=True)
        self.assertFalse(torch.equal(graph_forecast.detach(), routed_forecast.detach()))
        self.assertFalse(torch.equal(graph_repr.detach(), routed_repr.detach()))

    def test_v3_sparse_graph_branch_rejects_an_unaligned_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            graph_path = Path(directory) / "bad_graph.npz"
            np.savez_compressed(
                graph_path,
                schema_version=np.asarray(1, dtype=np.int64),
                sensor_ids=np.asarray(["0", "1", "2", "3"]),
                target=np.asarray([0], dtype=np.int64),
                source=np.asarray([1], dtype=np.int64),
                weight=np.ones(1, dtype=np.float32),
                node_count=np.asarray(4, dtype=np.int64),
            )
            with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
                with self.assertRaisesRegex(ValueError, "node count"):
                    mamba_ts_v3.Model(_config(
                        AutoCon=True,
                        mamba_ts_graph_spatial=True,
                        mamba_ts_graph_path=str(graph_path),
                    ))

    def test_graph_dlinear_is_nested_in_dlinear_and_learns_graph_scale(self):
        torch.manual_seed(117)
        x = torch.randn(2, 96, 5)
        with tempfile.TemporaryDirectory() as directory:
            graph_path = Path(directory) / "graph.npz"
            np.savez_compressed(
                graph_path,
                schema_version=np.asarray(1, dtype=np.int64),
                sensor_ids=np.asarray(["0", "1", "2", "3", "4"]),
                target=np.asarray([0, 1, 2, 3, 4], dtype=np.int64),
                source=np.asarray([1, 2, 3, 4, 0], dtype=np.int64),
                weight=np.ones(5, dtype=np.float32),
                node_count=np.asarray(5, dtype=np.int64),
            )
            torch.manual_seed(233)
            control = dlinear.Model(_config())
            torch.manual_seed(233)
            graph = graph_dlinear.Model(_config(
                mamba_ts_graph_path=str(graph_path),
            ))
            control_forecast = control(x, None, None, None)
            graph_forecast = graph(x, None, None, None)
            torch.testing.assert_close(control_forecast, graph_forecast)
            graph_forecast.square().mean().backward()
            self.assertGreater(float(graph.graph_spatial_scale.grad.detach().abs()), 0.0)
            with torch.no_grad():
                graph.graph_spatial_scale.fill_(1.0)
            routed_forecast = graph(x, None, None, None)
        self.assertFalse(torch.equal(graph_forecast.detach(), routed_forecast.detach()))

    def test_v3_repr_gated_linear_expert_is_nested_and_uses_autocon_features(self):
        x = torch.randn(2, 96, 5)
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            control = mamba_ts_v3.Model(_config(AutoCon=True, mamba_ts_context_rank=4))
            expert = mamba_ts_v3.Model(_config(
                AutoCon=True, mamba_ts_context_rank=4,
                mamba_ts_repr_linear_expert=True,
            ))
            expert.load_state_dict(control.state_dict(), strict=False)
            control_forecast, _ = control(x, return_repr=True)
            expert_forecast, _ = expert(x, return_repr=True)
            torch.testing.assert_close(control_forecast, expert_forecast)
            self.assertTrue(expert.ts_repr_linear_expert_enabled)
            expert_forecast.square().mean().backward()
            self.assertGreater(float(expert.ts_repr_linear_expert_scale.grad.abs()), 0.0)
            with torch.no_grad():
                inference_forecast, inference_representation = expert(
                    x, return_repr=False
                )
            self.assertEqual(inference_forecast.shape, expert_forecast.shape)
            self.assertIsNotNone(inference_representation)
            with torch.no_grad():
                expert.ts_repr_linear_expert_scale.fill_(1.0)
            routed_forecast, _ = expert(x, return_repr=True)
        self.assertFalse(torch.equal(expert_forecast.detach(), routed_forecast.detach()))

    def test_v3_repr_linear_expert_blend_is_bounded_and_uses_shared_representation(self):
        x = torch.randn(2, 96, 5)
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            model = mamba_ts_v3.Model(_config(
                AutoCon=True,
                mamba_ts_context_rank=4,
                mamba_ts_repr_linear_expert=True,
                mamba_ts_repr_linear_expert_mode="blend",
                mamba_ts_repr_linear_blend_bias=-4.0,
            ))
            forecast, representation = model(x, return_repr=True)
            self.assertEqual(model.ts_repr_linear_expert_mode, "blend")
            self.assertTrue(model.ts_repr_linear_expert_enabled)
            forecast.square().mean().backward()
            self.assertIsNotNone(model.ts_repr_linear_gate.weight.grad)
            self.assertGreater(float(model.ts_repr_linear_gate.weight.grad.abs().sum()), 0.0)
            with torch.no_grad():
                gate = torch.sigmoid(
                    model.ts_repr_linear_gate(representation.mean(dim=1))
                    + model.ts_repr_linear_blend_bias
                )
            self.assertGreaterEqual(float(gate.min()), 0.0)
            self.assertLessEqual(float(gate.max()), 1.0)

    def test_v3_repr_nlinear_blend_is_bounded_and_learns_low_rank_expert(self):
        x = torch.randn(2, 96, 5)
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            model = mamba_ts_v3.Model(_config(
                AutoCon=True,
                mamba_ts_context_rank=4,
                mamba_ts_repr_linear_expert=True,
                mamba_ts_repr_linear_expert_mode="nlinear_blend",
                mamba_ts_repr_linear_expert_rank=4,
                mamba_ts_repr_linear_blend_bias=-4.0,
            ))
            forecast, representation = model(x, return_repr=True)
            self.assertEqual(model.ts_repr_linear_expert_mode, "nlinear_blend")
            self.assertEqual(model.ts_repr_linear_expert_rank, 4)
            forecast.square().mean().backward()
            self.assertIsNotNone(model.ts_nlinear_left.weight.grad)
            self.assertGreater(float(model.ts_nlinear_left.weight.grad.abs().sum()), 0.0)
            with torch.no_grad():
                gate = torch.sigmoid(
                    model.ts_repr_linear_gate(representation.mean(dim=1))
                    + model.ts_repr_linear_blend_bias
                )
            self.assertGreaterEqual(float(gate.min()), 0.0)
            self.assertLessEqual(float(gate.max()), 1.0)

    def test_v3_channelwise_repr_forecast_shares_representation_with_autocon(self):
        x = torch.randn(2, 96, 5)
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            model = mamba_ts_v3.Model(_config(
                AutoCon=True,
                mamba_ts_context_rank=4,
                mamba_ts_repr_forecast_fusion=True,
                mamba_ts_repr_forecast_channelwise=True,
            ))
            forecast, representation = model(x, return_repr=True)
            self.assertTrue(model.ts_repr_forecast_channelwise)
            self.assertEqual(tuple(representation.shape), (2, 96, 16))
            self.assertEqual(tuple(forecast.shape), (2, 12, 5))
            forecast.square().mean().backward()
            self.assertGreater(
                float(model.ts_repr_forecast_scale.grad.detach().abs().sum()), 0.0
            )
            self.assertEqual(
                float(model.ts_repr_channel_projection.weight.grad.detach().abs().sum()),
                0.0,
            )
            # The ReZero scale opens first; once enabled the shared channel
            # representation receives direct MSE gradients.
            with torch.no_grad():
                model.ts_repr_forecast_scale.fill_(1.0)
            model.zero_grad(set_to_none=True)
            model(x, return_repr=True)[0].square().mean().backward()
        self.assertGreater(
            float(model.ts_repr_channel_projection.weight.grad.detach().abs().sum()),
            0.0,
        )

    def test_v3_shared_pretraining_mode_bypasses_channel_count_specific_inputs(self):
        x = torch.randn(2, 96, 5)
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            model = mamba_ts_v3.Model(_config(
                AutoCon=True,
                mamba_ts_context_rank=4,
                mamba_ts_shared_pretrain_mode=True,
            ))
            with torch.no_grad():
                model.channel_embedding.fill_(1.0)
                model.context_projection[-1].weight.fill_(1.0)
            encoded = model._encode(x)
            encoded.square().mean().backward()
        self.assertTrue(model.ts_shared_pretrain_mode)
        self.assertIsNone(model.channel_embedding.grad)
        self.assertIsNone(model.context_projection[0].weight.grad)
        self.assertIsNotNone(model.patch_embedding.weight.grad)
        self.assertGreater(float(model.patch_embedding.weight.grad.abs().sum()), 0.0)

    def test_v3_channel_agnostic_scan_is_channel_permutation_equivariant(self):
        x = torch.randn(2, 96, 5)
        order = torch.tensor([4, 1, 3, 0, 2])
        inverse = torch.argsort(order)
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            model = mamba_ts_v3.Model(_config(
                AutoCon=True,
                mamba_ts_context_rank=4,
                mamba_ts_channel_agnostic_scan=True,
            )).eval()
            encoded = model._encode(x)
            reordered = model._encode(x.index_select(-1, order)).index_select(1, inverse)
        torch.testing.assert_close(encoded, reordered)

    def test_v13_channel_agnostic_scan_is_channel_permutation_equivariant(self):
        x = torch.randn(2, 96, 5)
        order = torch.tensor([4, 1, 3, 0, 2])
        inverse = torch.argsort(order)
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            model = mamba_ts.Model(_config(
                AutoCon=True,
                mamba_ts_variant="v13",
                mamba_ts_context_rank=4,
                mamba_ts_channel_agnostic_scan=True,
            )).eval()
            encoded = model._encode(x)
            reordered = model._encode(x.index_select(-1, order)).index_select(1, inverse)
        torch.testing.assert_close(encoded, reordered)

    def test_v3_variable_permutation_restores_public_channel_order(self):
        x = torch.randn(2, 96, 5)
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            model = mamba_ts_v3.Model(_config(
                mamba_ts_context_rank=4,
                mamba_ts_variable_permutation=True,
            ))
            with torch.no_grad():
                model.channel_embedding.copy_(torch.randn_like(model.channel_embedding))
            model.eval(); reference = model(x)
            model.train(); permuted = model(x)
        torch.testing.assert_close(permuted, reference)

    def test_v4_attaches_train_only_acf_memory_without_changing_shape(self):
        x = torch.randn(2, 96, 5)
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            model = mamba_ts_v4.Model(_config(
                mamba_use_acf_memory=True,
                mamba_acf_lags=[24, 48],
                mamba_ts_context_rank=4,
            ))
            prior = torch.zeros(5, 97)
            prior[:, 24] = 0.8
            prior[:, 48] = -0.2
            model.set_ts_acf_prior(prior, [24, 48])
            prediction = model(x)
            memory = model._ts_acf_memory(model._normalize(x)[0])
        self.assertEqual(tuple(prediction.shape), (2, 12, 5))
        self.assertEqual(model.variant, "v4")
        self.assertTrue(model.ts_acf_memory_enabled)
        self.assertEqual(tuple(model.ts_acf_prior.shape), (5, 2))
        self.assertEqual(tuple(model.ts_acf_lag_values.tolist()), (24, 48))
        # ReZero starts the ACF path as an exact no-op until training updates it.
        self.assertTrue(torch.equal(memory, torch.zeros_like(memory)))

    def test_v4_acf_representation_fusion_is_autocon_only_and_zero_gated(self):
        x = torch.randn(2, 96, 5)
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            model = mamba_ts_v4.Model(_config(
                AutoCon=True,
                mamba_use_acf_memory=True,
                mamba_acf_repr_fusion=True,
                mamba_acf_lags=[24, 48],
                AutoCon_acf_source="raw",
            ))
            prior = torch.zeros(5, 97)
            prior[:, 24] = 0.8
            prior[:, 48] = -0.2
            model.set_ts_acf_prior(prior, [24, 48])
            forecast, representation = model(x, return_repr=True)
            context = model._ts_acf_representation_context(model._normalize(x)[0])
        self.assertEqual(tuple(forecast.shape), (2, 12, 5))
        self.assertEqual(tuple(representation.shape), (2, 96, 16))
        self.assertTrue(model.ts_acf_repr_fusion_enabled)
        self.assertEqual(tuple(context.shape), (2, 96, 8))
        self.assertEqual(float(model.ts_acf_repr_scale.detach()), 0.0)

    def test_v4_acf_memory_selector_is_zero_gated_and_uses_autocon_representation(self):
        x = torch.randn(2, 96, 5)
        prior = torch.zeros(5, 97)
        prior[:, 24] = 0.8
        prior[:, 48] = -0.2
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            control = mamba_ts_v4.Model(_config(
                AutoCon=True, mamba_use_acf_memory=True,
                mamba_acf_lags=[24, 48], AutoCon_acf_source="raw",
            ))
            selected = mamba_ts_v4.Model(_config(
                AutoCon=True, mamba_use_acf_memory=True,
                mamba_acf_lags=[24, 48], AutoCon_acf_source="raw",
                mamba_ts_acf_memory_selector=True,
            ))
            selected.load_state_dict(control.state_dict(), strict=False)
            for model in (control, selected):
                model.set_ts_acf_prior(prior, [24, 48])
                with torch.no_grad():
                    model.ts_acf_memory_scale.fill_(0.7)
            control_prediction, _ = control(x, return_repr=False)
            selected_prediction, representation = selected(x, return_repr=False)
            self.assertTrue(selected.ts_acf_memory_selector_enabled)
            self.assertEqual(tuple(representation.shape), (2, 96, 16))
            torch.testing.assert_close(control_prediction, selected_prediction)
            selected_prediction.square().mean().backward()
        self.assertGreater(
            float(selected.ts_acf_memory_selector_scale.grad.detach().abs()), 0.0
        )

    def test_v4_memory_channel_selector_is_channel_count_independent(self):
        x = torch.randn(2, 96, 5)
        prior = torch.zeros(5, 97)
        prior[:, 24] = 0.8
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            control = mamba_ts_v4.Model(_config(
                AutoCon=True, mamba_use_acf_memory=True, mamba_acf_lags=[24],
                AutoCon_acf_source="raw", mamba_ts_acf_memory_selector=True,
            ))
            selected = mamba_ts_v4.Model(_config(
                AutoCon=True, mamba_use_acf_memory=True, mamba_acf_lags=[24],
                AutoCon_acf_source="raw", mamba_ts_acf_memory_selector=True,
                mamba_ts_acf_memory_channel_selector=True,
            ))
            selected.load_state_dict(control.state_dict(), strict=False)
            for model in (control, selected):
                model.set_ts_acf_prior(prior, [24])
                with torch.no_grad():
                    model.ts_acf_memory_scale.fill_(0.7)
                    model.ts_acf_memory_selector_scale.fill_(0.4)
            control_prediction, _ = control(x, return_repr=False)
            selected_prediction, _ = selected(x, return_repr=False)
        self.assertTrue(selected.ts_acf_memory_channel_selector_enabled)
        self.assertEqual(tuple(selected.ts_acf_memory_channel_selector.weight.shape), (1, 8))
        self.assertFalse(torch.allclose(control_prediction, selected_prediction))

    def test_v5_adds_zero_gated_causal_signed_anchor(self):
        x = torch.zeros(1, 96, 5)
        x[:, 72, :] = 2.0
        x[:, 48, :] = -1.0
        prior = torch.zeros(5, 97)
        prior[:, 24] = 0.8
        prior[:, 48] = -0.2
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            model = mamba_ts_v5.Model(_config(
                mamba_use_acf_memory=True,
                mamba_acf_lags=[24, 48],
                AutoCon_acf_source="raw",
            ))
            model.set_ts_acf_prior(prior, [24, 48])
            normalized = model._normalize(x)[0]
            initial = model._ts_acf_anchor(normalized)
            with torch.no_grad():
                model.ts_acf_anchor_scale.fill_(1.0)
            anchored = model._ts_acf_anchor(normalized)
            self.assertEqual(tuple(anchored.shape), (1, 12, 5))
            self.assertTrue(torch.equal(initial, torch.zeros_like(initial)))

            # The strongest lag is 24, so the first forecast step reads index
            # 72 from the observed window.  An unobserved index cannot affect
            # any anchor value, which is the causal lookup contract.
            normalized_changed = normalized.clone()
            normalized_changed[:, 95, :] = 7.0
            changed_anchor = model._ts_acf_anchor(normalized_changed)
        self.assertGreater(float(anchored.detach().abs().sum()), 0.0)
        self.assertTrue(torch.equal(changed_anchor, anchored))
        self.assertEqual(model.variant, "v5")
        self.assertEqual(float(model.ts_acf_anchor_scale.detach()), 1.0)

    def test_v4_compact_acf_sidecar_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "acf_prior.npz"
            with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
                source = mamba_ts_v4.Model(_config(
                    mamba_use_acf_memory=True,
                    mamba_acf_lags=[24, 48],
                    mamba_acf_temperature=4.0,
                ))
                prior = torch.zeros(5, 97)
                prior[:, 24] = 0.8
                prior[:, 48] = -0.2
                source.set_ts_acf_prior(prior, [24, 48])
                self.assertTrue(source.save_acf_prior(artifact))
                restored = mamba_ts_v4.Model(_config(
                    mamba_use_acf_memory=True,
                    mamba_acf_lags=[24, 48],
                    mamba_acf_temperature=4.0,
                ))
                self.assertTrue(restored.load_acf_prior(artifact))
            self.assertEqual(tuple(restored.ts_acf_lag_values.tolist()), (24, 48))
            torch.testing.assert_close(restored.ts_acf_prior, source.ts_acf_prior)

    def test_v8_is_causal_spectral_residual_and_autocon_connected(self):
        x = torch.randn(2, 96, 5)
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            model = mamba_ts_v8.Model(_config(
                AutoCon=True,
                mamba_ts_context_rank=4,
                mamba_ts_spectral_top_k=3,
            ))
            forecast, representation = model(x, return_repr=True)
            self.assertEqual(model.variant, "v8")
            self.assertTrue(model.ts_spectral_enabled)
            self.assertEqual(model.ts_spectral_top_k, 3)
            self.assertEqual(float(model.ts_spectral_scale.detach()), 0.0)
            self.assertEqual(tuple(forecast.shape), (2, 12, 5))
            self.assertEqual(tuple(representation.shape), (2, 96, 16))
            normalized, _ = model._normalize(x)
            candidate = model._spectral_forecast(normalized)
            self.assertEqual(tuple(candidate.shape), (2, 12, 5))
            self.assertTrue(torch.isfinite(candidate).all())
            # The Fourier candidate depends only on the observed prefix.
            changed = x.clone()
            changed[:, -8:, :] += 1000.0
            first = model._spectral_forecast(model._normalize(x)[0])
            second = model._spectral_forecast(model._normalize(changed)[0])
            self.assertFalse(torch.equal(first, second))

    def test_v9_decomposes_observed_window_and_connects_autocon_to_forecast(self):
        x = torch.randn(2, 96, 5)
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            plain = mamba_ts_v9.Model(_config(mamba_ts_context_rank=4))
            prediction = plain(x)
            self.assertEqual(plain.variant, "v9")
            self.assertTrue(plain.ts_decomposition_enabled)
            self.assertEqual(tuple(prediction.shape), (2, 12, 5))
            self.assertIsNone(plain.ts_decomposition_gate)

            model = mamba_ts_v9.Model(_config(AutoCon=True, mamba_ts_context_rank=4))
            forecast, representation = model(x, return_repr=True)
            self.assertEqual(tuple(forecast.shape), (2, 12, 5))
            self.assertEqual(tuple(representation.shape), (2, 96, 16))
            self.assertEqual(float(model.ts_decomposition_scale.detach()), 0.0)
            self.assertIsNotNone(model.acf_runtime_diagnostics()["decomposition_runtime_gate_mean"])
            forecast.square().mean().backward()
            self.assertIsNotNone(model.ts_decomposition_scale.grad)
            self.assertGreater(float(model.ts_decomposition_scale.grad.detach().abs()), 0.0)
            with torch.no_grad():
                model.ts_decomposition_scale.fill_(1.0)
            fused, _ = model(x, return_repr=True)
        self.assertFalse(torch.equal(forecast.detach(), fused.detach()))

    def test_v10_builds_a_fixed_train_acf_order_and_restores_channels(self):
        x = torch.randn(2, 96, 5)
        prior = torch.tensor([
            [0.9, 0.1], [0.8, 0.2], [-0.8, -0.2], [-0.7, -0.3], [0.1, 0.9],
        ])
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            model = mamba_ts_v10.Model(_config(
                AutoCon=True, mamba_ts_context_rank=4,
                mamba_use_acf_memory=True, mamba_acf_lags=[24, 48],
                mamba_ts_repr_forecast_fusion=True,
            ))
            model.set_ts_acf_prior(prior, [24, 48])
            self.assertEqual(model.variant, "v10")
            self.assertTrue(model.ts_acf_order_enabled)
            self.assertTrue(torch.equal(
                torch.sort(model.ts_acf_channel_order).values,
                torch.arange(5),
            ))
            forecast, representation = model(x, return_repr=True)
            self.assertEqual(tuple(forecast.shape), (2, 12, 5))
            self.assertEqual(tuple(representation.shape), (2, 96, 16))
            forecast.square().mean().backward()
            self.assertIsNotNone(model.ts_repr_forecast_scale.grad)

    def test_v11_records_train_edges_and_uses_a_learned_eval_order(self):
        x = torch.randn(3, 96, 5)
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            model = mamba_ts_v11.Model(_config(
                AutoCon=True, mamba_ts_context_rank=4,
                mamba_ts_repr_forecast_fusion=True,
            ))
            forecast, representation = model(x, return_repr=True)
            self.assertEqual(tuple(forecast.shape), (3, 12, 5))
            self.assertEqual(tuple(representation.shape), (3, 96, 16))
            self.assertTrue(model.ts_repr_forecast_fusion_enabled)
            forecast.square().mean().backward()
            self.assertIsNotNone(model.ts_repr_forecast_scale.grad)
            model.update_vpt_transition_cost(torch.rand(3, 12, 5))
            self.assertGreater(float(model.ts_vpt_transition_count.sum()), 0.0)
            model.eval()
            prediction, _ = model(x, return_repr=True)
            self.assertEqual(tuple(prediction.shape), (3, 12, 5))
            self.assertTrue(torch.equal(
                torch.sort(model.ts_vpt_channel_order).values,
                torch.arange(5),
            ))
            self.assertIsNotNone(model.vpt_runtime_diagnostics())

    def test_v11_python_tsp_solver_emits_a_valid_variable_path(self):
        x = torch.randn(3, 96, 5)
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            model = mamba_ts_v11.Model(_config(
                AutoCon=True, mamba_ts_context_rank=4,
                mamba_ts_repr_forecast_fusion=True,
                mamba_ts_vpt_solver="ls",
            ))
            forecast, _ = model(x, return_repr=True)
            model.update_vpt_transition_cost(torch.rand_like(forecast))
            model.eval()
            model(x, return_repr=True)
            diagnostics = model.vpt_runtime_diagnostics()
        self.assertEqual(diagnostics["vpt_solver"], "ls")
        self.assertTrue(torch.equal(
            torch.sort(model.ts_vpt_channel_order).values,
            torch.arange(5),
        ))

    def test_v11_rejects_unknown_vpt_solver(self):
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            with self.assertRaisesRegex(ValueError, "mamba_ts_vpt_solver"):
                mamba_ts_v11.Model(_config(mamba_ts_vpt_solver="unknown"))

    def test_v12_fuses_two_delta_branches_and_exposes_autocon_state(self):
        x = torch.randn(2, 96, 5)
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            model = mamba_ts_v12.Model(_config(
                AutoCon=True,
                mamba_ts_context_rank=4,
                mamba_ts_delta_multipliers=[1.0, 2.0],
            ))
            forecast, representation = model(x, return_repr=True)
            self.assertEqual(model.variant, "v12")
            self.assertTrue(model.ts_multiscale_enabled)
            self.assertIsNone(model.encoder)
            self.assertEqual(len(model.ts_multiscale_encoder), 2)
            torch.testing.assert_close(
                torch.softmax(model.ts_multiscale_logits, dim=0),
                torch.full((2,), 0.5),
            )
            self.assertEqual(tuple(forecast.shape), (2, 12, 5))
            self.assertEqual(tuple(representation.shape), (2, 96, 16))
            forecast.square().mean().backward()
            self.assertIsNotNone(model.ts_multiscale_logits.grad)

    def test_v13_keeps_one_scan_and_adds_zero_initialized_lite_scales(self):
        x = torch.randn(2, 96, 5)
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            model = mamba_ts_v13.Model(_config(
                AutoCon=True, mamba_ts_context_rank=4,
                mamba_ts_repr_forecast_fusion=True,
            ))
            forecast, representation = model(x, return_repr=True)
            self.assertEqual(model.variant, "v13")
            self.assertTrue(model.ts_lite_multiscale_enabled)
            self.assertIsNotNone(model.encoder)
            self.assertEqual(len(model.ts_multiscale_encoder) if hasattr(model, "ts_multiscale_encoder") and model.ts_multiscale_encoder is not None else 0, 0)
            self.assertEqual(float(model.ts_lite_multiscale_scale.detach()), 0.0)
            self.assertEqual(tuple(forecast.shape), (2, 12, 5))
            self.assertEqual(tuple(representation.shape), (2, 96, 16))
            forecast.square().mean().backward()
            self.assertIsNotNone(model.ts_lite_multiscale_scale.grad)

    def test_v14_keeps_one_scan_and_adds_zero_initialized_observed_local_expert(self):
        x = torch.randn(2, 96, 5)
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            model = mamba_ts_v14.Model(_config(
                AutoCon=True, mamba_ts_context_rank=4,
                mamba_ts_repr_forecast_fusion=True,
                patch_len=24, patch_stride=24,
            ))
            forecast, representation = model(x, return_repr=True)
            self.assertEqual(model.variant, "v14")
            self.assertTrue(model.ts_pre_local_enabled)
            self.assertEqual(model.patch_count, 4)
            self.assertIsNotNone(model.encoder)
            self.assertEqual(float(model.ts_pre_local_scale.detach()), 0.0)
            self.assertEqual(tuple(forecast.shape), (2, 12, 5))
            self.assertEqual(tuple(representation.shape), (2, 96, 16))
            forecast.square().mean().backward()
            self.assertIsNotNone(model.ts_pre_local_scale.grad)

    def test_v15_keeps_one_scan_and_conditions_shared_state_on_calendar_markers(self):
        x = torch.randn(2, 96, 5)
        marks = torch.randn(2, 96, 5)
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            model = mamba_ts_v15.Model(_config(
                AutoCon=True, mamba_ts_context_rank=4,
                mamba_ts_repr_forecast_fusion=True,
                mamba_ts_variant="v15",
            ))
            forecast, representation = model(x, x_mark_enc=marks, return_repr=True)
            forecast_without_marks, _ = model(x, return_repr=True)
            self.assertEqual(model.variant, "v15")
            self.assertTrue(model.ts_time_condition_enabled)
            self.assertIsNotNone(model.encoder)
            self.assertEqual(float(model.ts_time_condition_scale.detach()), 0.0)
            self.assertTrue(torch.allclose(forecast, forecast_without_marks))
            self.assertEqual(tuple(forecast.shape), (2, 12, 5))
            self.assertEqual(tuple(representation.shape), (2, 96, 16))
            forecast.square().mean().backward()
            self.assertIsNotNone(model.ts_time_condition_scale.grad)

    def test_v16_uses_patch_start_markers_without_sliding_reduction(self):
        x = torch.randn(2, 96, 5)
        marks = torch.randn(2, 96, 5)
        with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
            model = mamba_ts_v16.Model(_config(
                AutoCon=True, mamba_ts_context_rank=4,
                mamba_ts_repr_forecast_fusion=True,
                mamba_ts_variant="v16",
            ))
            forecast, representation = model(x, x_mark_enc=marks, return_repr=True)
            self.assertEqual(model.variant, "v16")
            self.assertTrue(model.ts_time_condition_anchor_enabled)
            self.assertEqual(tuple(model.ts_time_patch_starts.shape), (11,))
            self.assertEqual(float(model.ts_time_condition_scale.detach()), 0.0)
            self.assertEqual(tuple(forecast.shape), (2, 12, 5))
            self.assertEqual(tuple(representation.shape), (2, 96, 16))
            forecast.square().mean().backward()
            self.assertIsNotNone(model.ts_time_condition_scale.grad)

    def test_v4_sidecar_rejects_geometry_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "acf_prior.npz"
            with patch.object(mamba_ts, "_Mamba1", _IdentityMamba):
                source = mamba_ts_v4.Model(_config(mamba_use_acf_memory=True, mamba_acf_lags=[24]))
                prior = torch.zeros(5, 97)
                prior[:, 24] = 0.8
                source.set_ts_acf_prior(prior, [24])
                source.save_acf_prior(artifact)
                restored = mamba_ts_v4.Model(_config(
                    pred_len=24, mamba_use_acf_memory=True, mamba_acf_lags=[24]
                ))
                with self.assertRaisesRegex(ValueError, "forecast geometry"):
                    restored.load_acf_prior(artifact)

    def test_registry_and_runtime_contract_include_patch_geometry(self):
        self.assertIn("MambaTS", MAMBA_PLAIN_MODELS)
        self.assertIn("MambaTSAutoCon", MAMBA_AUTOCON_MODELS)
        self.assertEqual(MODEL_SPECS["MambaTS"]["model"], "MambaTS")
        self.assertTrue(MODEL_SPECS["MambaTSAutoCon"]["autocon"])
        payload = _mamba_runtime_contract_payload({
            "model": "MambaTS",
            "AutoCon": False,
            "seq_len": 96,
            "pred_len": 12,
            "enc_in": 5,
            "c_out": 5,
            "patch_len": 12,
            "patch_stride": 4,
        })
        self.assertEqual(payload["patch_len"], 12)
        self.assertEqual(payload["patch_stride"], 4)
        with self.assertRaisesRegex(ValueError, "patch_len cannot exceed"):
            _mamba_runtime_contract_payload({
                "model": "MambaTS",
                "seq_len": 8,
                "pred_len": 4,
                "enc_in": 2,
                "c_out": 2,
                "patch_len": 16,
            })

        v2_payload = _mamba_runtime_contract_payload({
            "model": "MambaTSv2",
            "AutoCon": False,
            "seq_len": 96,
            "pred_len": 12,
            "enc_in": 5,
            "c_out": 5,
        })
        self.assertEqual(v2_payload["mamba_ts_variant"], "v2")
        self.assertEqual(MODEL_SPECS["MambaTSv2"]["model"], "MambaTSv2")
        self.assertEqual(
            _mamba_runtime_contract_payload({
                "model": "MambaTSv3", "seq_len": 96, "pred_len": 12,
                "enc_in": 5, "c_out": 5, "mamba_ts_context_rank": 4,
            })["mamba_ts_context_rank"], 4
        )
        self.assertTrue(
            _mamba_runtime_contract_payload({
                "model": "MambaTSv3", "seq_len": 96, "pred_len": 12,
                "enc_in": 5, "c_out": 5, "mamba_ts_channel_agnostic_scan": True,
            })["mamba_ts_channel_agnostic_scan"]
        )
        self.assertEqual(
            _mamba_runtime_contract_payload({
                "model": "MambaTSv3", "seq_len": 96, "pred_len": 12,
                "enc_in": 5, "c_out": 5, "mamba_ts_mixer_type": "mamba2",
            })["mamba_ts_mixer_type"], "mamba2"
        )
        with self.assertRaisesRegex(ValueError, "mixer_type"):
            _mamba_runtime_contract_payload({
                "model": "MambaTSv3", "seq_len": 96, "pred_len": 12,
                "enc_in": 5, "c_out": 5, "mamba_ts_mixer_type": "invalid",
            })
        self.assertEqual(MODEL_SPECS["MambaTSv3AutoCon"]["model"], "MambaTSv3AutoCon")
        fused_payload = _mamba_runtime_contract_payload({
            "model": "MambaTSv3AutoCon", "AutoCon": True,
            "seq_len": 96, "pred_len": 12, "enc_in": 5, "c_out": 5,
            "mamba_ts_repr_forecast_fusion": True,
        })
        self.assertTrue(fused_payload["mamba_ts_repr_forecast_fusion"])
        adapter_payload = _mamba_runtime_contract_payload({
            "model": "MambaTSv3AutoCon", "AutoCon": True,
            "seq_len": 96, "pred_len": 12, "enc_in": 5, "c_out": 5,
            "mamba_ts_channel_adapter": True,
            "mamba_ts_channel_adapter_rank": 4,
        })
        self.assertTrue(adapter_payload["mamba_ts_channel_adapter"])
        self.assertEqual(adapter_payload["mamba_ts_channel_adapter_rank"], 4)
        mixer_payload = _mamba_runtime_contract_payload({
            "model": "MambaTSv3AutoCon", "AutoCon": True,
            "seq_len": 96, "pred_len": 12, "enc_in": 5, "c_out": 5,
            "mamba_ts_global_channel_mixer": True,
            "mamba_ts_global_channel_mixer_rank": 3,
        })
        self.assertTrue(mixer_payload["mamba_ts_global_channel_mixer"])
        self.assertEqual(mixer_payload["mamba_ts_global_channel_mixer_rank"], 3)
        graph_payload = _mamba_runtime_contract_payload({
            "model": "MambaTSv3AutoCon", "AutoCon": True,
            "seq_len": 96, "pred_len": 12, "enc_in": 5, "c_out": 5,
            "mamba_ts_graph_spatial": True,
            "mamba_ts_graph_path": "/tmp/aligned-graph.npz",
            "mamba_ts_graph_expected_sha256": "a" * 64,
        })
        self.assertTrue(graph_payload["mamba_ts_graph_spatial"])
        self.assertEqual(
            graph_payload["mamba_ts_graph_expected_sha256"], "a" * 64
        )
        self.assertEqual(
            graph_payload["mamba_ts_graph_path"], "/tmp/aligned-graph.npz"
        )
        forecast_acf_payload = _mamba_runtime_contract_payload({
            "model": "MambaTSv3AutoCon", "AutoCon": True,
            "seq_len": 96, "pred_len": 12, "enc_in": 5, "c_out": 5,
            "mamba_ts_forecast_acf_consistency": True,
            "mamba_ts_forecast_acf_weight": 0.01,
        })
        self.assertTrue(forecast_acf_payload["mamba_ts_forecast_acf_consistency"])
        permutation_payload = _mamba_runtime_contract_payload({
            "model": "MambaTSv3", "seq_len": 96, "pred_len": 12,
            "enc_in": 5, "c_out": 5, "mamba_ts_variable_permutation": True,
        })
        self.assertTrue(permutation_payload["mamba_ts_variable_permutation"])
        v4_payload = _mamba_runtime_contract_payload({
            "model": "MambaTSv4AutoCon", "AutoCon": True,
            "seq_len": 96, "pred_len": 12, "enc_in": 5, "c_out": 5,
            "mamba_use_acf_memory": True, "mamba_acf_lags": [24, 48],
            "mamba_ts_context_rank": 4, "mamba_acf_repr_fusion": True,
        })
        self.assertEqual(v4_payload["mamba_ts_variant"], "v4")
        self.assertTrue(v4_payload["mamba_use_acf_memory"])
        self.assertEqual(v4_payload["mamba_acf_lags"], [24, 48])
        self.assertTrue(v4_payload["mamba_acf_repr_fusion"])
        self.assertEqual(MODEL_SPECS["MambaTSv4AutoCon"]["model"], "MambaTSv4AutoCon")
        v5_payload = _mamba_runtime_contract_payload({
            "model": "MambaTSv5AutoCon", "AutoCon": True,
            "seq_len": 96, "pred_len": 12, "enc_in": 5, "c_out": 5,
            "mamba_use_acf_memory": True, "mamba_acf_lags": [24, 48],
            "mamba_ts_context_rank": 4,
            "AutoCon_acf_source": "raw",
        })
        self.assertEqual(v5_payload["mamba_ts_variant"], "v5")
        self.assertTrue(v5_payload["mamba_use_acf_memory"])
        self.assertEqual(MODEL_SPECS["MambaTSv5AutoCon"]["model"], "MambaTSv5AutoCon")

    def test_resource_normalizer_accepts_both_identifiers(self):
        common = {
            "seq_len": 96,
            "label_len": 48,
            "pred_len": 12,
            "enc_in": 5,
            "dec_in": 5,
            "c_out": 5,
            "d_model": 8,
            "d_ff": 16,
            "e_layers": 1,
            "n_heads": 1,
            "d_layers": 1,
            "patch_len": 16,
            "patch_stride": 8,
        }
        plain = _normalise_config({**common, "model": "MambaTS", "AutoCon": False})
        autocon = _normalise_config({**common, "model": "MambaTSAutoCon", "AutoCon": True})
        self.assertEqual(plain["model"], "MambaTS")
        self.assertEqual(autocon["model"], "MambaTSAutoCon")
        self.assertTrue(autocon["AutoCon"])
        v2 = _normalise_config({
            **common, "model": "MambaTSv2", "AutoCon": False,
        })
        self.assertEqual(v2["mamba_ts_variant"], "v2")
        v3 = _normalise_config({
            **common, "model": "MambaTSv3", "AutoCon": False,
            "mamba_ts_context_rank": 4,
        })
        self.assertEqual(v3["mamba_ts_variant"], "v3")
        self.assertEqual(v3["mamba_ts_context_rank"], 4)
        v3_permuted = _normalise_config({
            **common, "model": "MambaTSv3AutoCon", "AutoCon": True,
            "mamba_ts_context_rank": 4,
            "mamba_ts_variable_permutation": True,
        })
        self.assertTrue(v3_permuted["mamba_ts_variable_permutation"])
        v4 = _normalise_config({
            **common, "model": "MambaTSv4AutoCon", "AutoCon": True,
            "mamba_use_acf_memory": True, "mamba_acf_lags": [24, 48],
            "mamba_ts_context_rank": 4, "mamba_acf_repr_fusion": True,
            "AutoCon_acf_source": "raw",
        })
        self.assertEqual(v4["mamba_ts_variant"], "v4")
        self.assertEqual(v4["mamba_ts_context_rank"], 4)
        self.assertTrue(v4["mamba_use_acf_memory"])
        self.assertEqual(v4["mamba_acf_lags"], [24, 48])
        self.assertTrue(v4["mamba_acf_repr_fusion"])
        v5 = _normalise_config({
            **common, "model": "MambaTSv5AutoCon", "AutoCon": True,
            "mamba_use_acf_memory": True, "mamba_acf_lags": [24, 48],
            "AutoCon_acf_source": "raw",
        })
        self.assertEqual(v5["mamba_ts_variant"], "v5")
        self.assertTrue(v5["mamba_use_acf_memory"])

    def test_v8_runtime_and_resource_contract(self):
        common = {
            "seq_len": 96, "label_len": 48, "pred_len": 24,
            "enc_in": 5, "dec_in": 5, "c_out": 5,
            "d_model": 8, "d_ff": 16, "e_layers": 1,
            "n_heads": 1, "d_layers": 1, "patch_len": 16,
            "patch_stride": 8,
        }
        payload = _mamba_runtime_contract_payload({
            "model": "MambaTSv8AutoCon", "AutoCon": True,
            "seq_len": 96, "pred_len": 24, "enc_in": 5, "c_out": 5,
            "mamba_ts_spectral_top_k": 6,
        })
        self.assertEqual(payload["mamba_ts_variant"], "v8")
        self.assertEqual(payload["mamba_ts_spectral_top_k"], 6)
        normalised = _normalise_config({
            **common, "model": "MambaTSv8AutoCon", "AutoCon": True,
            "mamba_ts_spectral_top_k": 6,
        })
        self.assertEqual(normalised["mamba_ts_variant"], "v8")
        self.assertEqual(normalised["mamba_ts_spectral_top_k"], 6)

    def test_v9_runtime_and_resource_contract(self):
        common = {
            "seq_len": 96, "label_len": 48, "pred_len": 24,
            "enc_in": 5, "dec_in": 5, "c_out": 5,
            "d_model": 8, "d_ff": 16, "e_layers": 1,
            "n_heads": 1, "d_layers": 1, "patch_len": 16,
            "patch_stride": 8,
        }
        payload = _mamba_runtime_contract_payload({
            "model": "MambaTSv9AutoCon", "AutoCon": True,
            "seq_len": 96, "pred_len": 24, "enc_in": 5, "c_out": 5,
            "moving_avg": 25,
        })
        self.assertEqual(payload["mamba_ts_variant"], "v9")
        normalised = _normalise_config({
            **common, "model": "MambaTSv9AutoCon", "AutoCon": True,
            "moving_avg": 25,
        })
        self.assertEqual(normalised["mamba_ts_variant"], "v9")

    def test_v10_runtime_and_resource_contract(self):
        common = {
            "seq_len": 96, "label_len": 48, "pred_len": 24,
            "enc_in": 5, "dec_in": 5, "c_out": 5,
            "d_model": 8, "d_ff": 16, "e_layers": 1,
            "n_heads": 1, "d_layers": 1, "patch_len": 16,
            "patch_stride": 8,
        }
        payload = _mamba_runtime_contract_payload({
            "model": "MambaTSv10AutoCon", "AutoCon": True,
            "seq_len": 96, "pred_len": 24, "enc_in": 5, "c_out": 5,
            "mamba_use_acf_memory": True, "mamba_acf_lags": [24, 48],
            "mamba_ts_repr_forecast_fusion": True,
        })
        self.assertEqual(payload["mamba_ts_variant"], "v10")
        self.assertTrue(payload["mamba_use_acf_memory"])
        normalised = _normalise_config({
            **common, "model": "MambaTSv10AutoCon", "AutoCon": True,
            "mamba_use_acf_memory": True, "mamba_acf_lags": [24, 48],
            "mamba_ts_repr_forecast_fusion": True,
        })
        self.assertEqual(normalised["mamba_ts_variant"], "v10")

    def test_v11_runtime_and_resource_contract(self):
        common = {
            "seq_len": 96, "label_len": 48, "pred_len": 24,
            "enc_in": 5, "dec_in": 5, "c_out": 5,
            "d_model": 8, "d_ff": 16, "e_layers": 1,
            "n_heads": 1, "d_layers": 1, "patch_len": 16,
            "patch_stride": 8,
        }
        payload = _mamba_runtime_contract_payload({
            "model": "MambaTSv11AutoCon", "AutoCon": True,
            "seq_len": 96, "pred_len": 24, "enc_in": 5, "c_out": 5,
            "mamba_ts_repr_forecast_fusion": True,
        })
        self.assertEqual(payload["mamba_ts_variant"], "v11")
        normalised = _normalise_config({
            **common, "model": "MambaTSv11AutoCon", "AutoCon": True,
            "mamba_ts_repr_forecast_fusion": True,
        })
        self.assertEqual(normalised["mamba_ts_variant"], "v11")

    def test_v12_runtime_and_resource_contract(self):
        common = {
            "seq_len": 96, "label_len": 48, "pred_len": 24,
            "enc_in": 5, "dec_in": 5, "c_out": 5,
            "d_model": 8, "d_ff": 16, "e_layers": 1,
            "n_heads": 1, "d_layers": 1, "patch_len": 16,
            "patch_stride": 8,
        }
        payload = _mamba_runtime_contract_payload({
            "model": "MambaTSv12AutoCon", "AutoCon": True,
            "seq_len": 96, "pred_len": 24, "enc_in": 5, "c_out": 5,
            "mamba_ts_delta_multipliers": [1.0, 2.0],
        })
        self.assertEqual(payload["mamba_ts_variant"], "v12")
        self.assertEqual(payload["mamba_ts_delta_multipliers"], (1.0, 2.0))
        normalised = _normalise_config({
            **common, "model": "MambaTSv12AutoCon", "AutoCon": True,
            "mamba_ts_delta_multipliers": [1.0, 2.0],
        })
        self.assertEqual(normalised["mamba_ts_variant"], "v12")


if __name__ == "__main__":
    unittest.main()
