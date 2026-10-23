from pathlib import Path


def test_transfer_compatibility_analysis_is_train_selection_only_and_exploratory():
    text = Path("scripts/analyze_acf_transfer_compatibility.py").read_text(encoding="utf-8")
    for token in (
        "selected_sources", "mean_selected_source_similarity",
        "active_vs_null_percent", "exploratory", "test_metrics_produced",
        "target train-prefix source selection",
    ):
        assert token in text
