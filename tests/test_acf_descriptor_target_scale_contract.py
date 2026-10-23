from pathlib import Path


def test_target_descriptor_scale_is_explicit_and_requires_descriptor_path():
    text = Path("scripts/finetune_mambatsv3_dlinear_hybrid.py").read_text(encoding="utf-8")
    for token in (
        "--mamba-ts-acf-channel-descriptor-tune-scale",
        "requires descriptor conditioning", "ts_acf_channel_descriptor_scale",
        "acf_descriptor_scale_trainable", "acf_descriptor_scale",
    ):
        assert token in text
