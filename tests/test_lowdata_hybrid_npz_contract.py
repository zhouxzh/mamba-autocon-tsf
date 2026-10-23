from pathlib import Path


def test_hybrid_training_and_test_reader_support_numeric_array_targets():
    for path in (
        Path("scripts/finetune_mambatsv3_dlinear_hybrid.py"),
        Path("scripts/evaluate_lowdata_checkpoint_test.py"),
    ):
        text = path.read_text(encoding="utf-8")
        assert "load_numeric_array_file" in text
        assert "data_format=\"auto\"" in text
