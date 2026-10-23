from pathlib import Path

from scripts.validate_paper_release_bundle import BUNDLES, render_markdown


def test_release_bundle_specs_cover_four_frozen_releases():
    assert len(BUNDLES) == 4
    assert {spec["summary_rows"] for spec in BUNDLES} == {12, 14, 28}
    assert sum(spec["expected_noninferior_cells"] for spec in BUNDLES) == 65
    assert all(spec["test_rows"] > spec["summary_rows"] for spec in BUNDLES)


def test_render_markdown_exposes_scope_and_status():
    text = render_markdown(
        {
            "passed": True,
            "bundles": [
                {
                    "name": "Example",
                    "summary_cells": 1,
                    "test_rows": 5,
                    "noninferior_cells": 1,
                    "candidate_wins_total": 1,
                    "passed": True,
                }
            ],
        }
    )
    assert "Overall status: **PASS**" in text
    assert "not a universal claim" in text
