from pathlib import Path


def test_local_cluster_results_draft_has_evidence_bindings_and_boundaries():
    text = Path("docs/research/mambatsv3_acf_infonce_local_cluster_results_draft_20261003.md").read_text(encoding="utf-8")
    for token in ("Not for submission", "[claim:C101]", "[evidence:E101]", "Boundaries", "Electricity", "Traffic", "independent", "machine replication"):
        assert token in text
