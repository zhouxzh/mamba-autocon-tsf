from pathlib import Path


def test_official_patchtst_benchmark_is_prediction_only_and_source_hash_bound():
    text = Path("scripts/benchmark_official_patchtst_checkpoint_inference.py").read_text(encoding="utf-8")
    for token in (
        "official PatchTST source hash", "prediction_only_official_patchtst",
        "relation_loss_included", "latency_ms_median", "torch.cuda.Event",
    ):
        assert token in text
    assert "optimizer" not in text
