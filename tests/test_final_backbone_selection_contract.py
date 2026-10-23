from pathlib import Path


def test_final_backbone_selection_keeps_release_and_search_boundaries():
    text = Path("scripts/audit_final_backbone_selection.py").read_text(encoding="utf-8")
    for token in (
        "MambaTSv3 local-cluster, Mamba-1 mixer",
        "mamba1",
        "mamba2_remains_validation_only",
        "v13_did_not_clear_promotion_rule",
        "mamba3_resource_gate_rejected",
        "pooled_effect_reported",
        "submission_ready",
    ):
        assert token in text
