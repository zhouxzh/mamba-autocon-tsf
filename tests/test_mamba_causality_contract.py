import unittest

from utils.mamba_causality_contract import strict_representation_causality_errors


class MambaCausalityContractTests(unittest.TestCase):
    def _strict_config(self):
        return {
            "model": "MambaACFAutoCon",
            "mamba_causal_embedding": True,
            "mamba_bidirectional": False,
            "mamba_token_stride": 1,
            "mamba_scales": [1],
            "mamba_channel_gate": False,
            "mamba_data_dependent_channel_mixer": False,
            "mamba_acf_scale_conditioning": False,
            "mamba_local_mixer": False,
            "mamba_acf_repr_fusion": False,
            "mamba_variable_repr_fusion": False,
            "autocon_acf_source": "raw",
        }

    def test_strict_grid_rejects_current_window_normalization(self):
        self.assertEqual(
            strict_representation_causality_errors(self._strict_config()),
            ["window_full_normalization_forbidden"],
        )

    def test_all_supported_window_normalizations_are_blocked(self):
        for mode in ("MeanStd", "LastVal", "Mean", "ReVIN"):
            errors = strict_representation_causality_errors(
                {**self._strict_config(), "AutoCon_wnorm": mode}
            )
            self.assertIn("window_full_normalization_forbidden", errors)

    def test_strict_grid_reports_each_noncausal_representation_path(self):
        config = {
            **self._strict_config(),
            "mamba_causal_embedding": False,
            "mamba_bidirectional": True,
            "mamba_token_stride": 4,
            "mamba_scales": [1, 2],
            "mamba_channel_gate": True,
            "mamba_data_dependent_channel_mixer": True,
            "mamba_acf_scale_conditioning": True,
            "mamba_local_mixer": True,
            "mamba_acf_repr_fusion": True,
            "autocon_acf_source": "residual",
        }
        self.assertEqual(
            strict_representation_causality_errors(config),
            [
                "autocon_acf_source_raw_required",
                "mamba_acf_scale_conditioning_forbidden",
                "mamba_bidirectional_forbidden",
                "mamba_causal_embedding_required",
                "mamba_channel_gate_forbidden",
                "mamba_data_dependent_channel_mixer_forbidden",
                "mamba_local_mixer_forbidden",
                "mamba_scales_must_be_single_scale_one",
                "mamba_token_stride_must_be_one",
                "window_full_normalization_forbidden",
            ],
        )

    def test_lowercase_source_alias_controls_repr_fusion_contract(self):
        config = {
            **self._strict_config(),
            "mamba_acf_repr_fusion": True,
            "autocon_acf_source": "raw",
            "AutoCon_acf_source": "residual",
        }
        self.assertEqual(
            strict_representation_causality_errors(config),
            ["window_full_normalization_forbidden"],
        )

    def test_variable_repr_fusion_is_marked_noncausal(self):
        config = {
            **self._strict_config(),
            "mamba_variable_repr_fusion": True,
            "mamba_variable_mixer": True,
        }
        self.assertEqual(
            strict_representation_causality_errors(config),
            [
                "mamba_variable_repr_fusion_forbidden",
                "window_full_normalization_forbidden",
            ],
        )

    def test_non_acf_models_have_no_mamba_acf_contract_errors(self):
        config = {**self._strict_config(), "model": "Mamba2"}
        self.assertEqual(strict_representation_causality_errors(config), [])


if __name__ == "__main__":
    unittest.main()
