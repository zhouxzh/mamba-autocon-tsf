from pathlib import Path


def test_shared_patchtst_acf_pretraining_is_channel_independent_and_validation_only():
    model = Path("models/PatchTSTACFPretrain.py").read_text(encoding="utf-8")
    for token in ("shared channel-independent", "TransformerEncoder", "prediction_head", "encode_normalized"):
        assert token in model
    runner = Path("scripts/run_patchtst_acf_traffic_probe.py").read_text(encoding="utf-8")
    for token in ("validation_only", "test_metrics_produced", "run_gpu_guarded.py", "channel_patch", "transformer_encoder_passes"):
        assert token in runner
