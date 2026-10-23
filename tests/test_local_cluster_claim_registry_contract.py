from pathlib import Path


def test_local_cluster_draft_uses_registered_claim_and_evidence_ids():
    draft = Path("docs/research/mambatsv3_acf_infonce_local_cluster_results_draft_20261003.md").read_text(encoding="utf-8")
    claims = Path("docs/research/mambatsv3_acf_infonce_local_cluster_claim_registry_20261003.md").read_text(encoding="utf-8")
    evidence = Path("docs/research/mambatsv3_acf_infonce_local_cluster_evidence_registry_20261003.md").read_text(encoding="utf-8")
    for token in ("[claim:C101]", "[claim:C109]", "[evidence:E101]", "[evidence:E103]"):
        assert token in draft
    for token in ("C101", "C109", "unverified"):
        assert token in claims
    for token in ("E101", "E103", "unverified"):
        assert token in evidence
