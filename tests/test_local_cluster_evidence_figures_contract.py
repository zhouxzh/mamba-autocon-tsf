from pathlib import Path


def test_evidence_figure_binds_audits_and_keeps_failed_gates_visible():
    text = Path("scripts/build_local_cluster_evidence_figures.py").read_text(encoding="utf-8")
    for token in ("failed_gates", "robustness_extensions", "inference_boundary", "Negative is better", "test_metrics_produced"):
        assert token in text
