"""Plan-layout tests for the high-dimensional held-out release reader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_mambatsv3_relation_auxiliary_highdim_test_release import _release_layout


ROOT = Path(__file__).resolve().parents[1]
STRIDE8 = ROOT / "docs/research/mambatsv3_relation_auxiliary_highdim_test_release_draft_20260928.json"
STRIDE24 = ROOT / "docs/research/mambatsv3_stride24_relation_auxiliary_highdim_test_release_draft_20261001.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "path,expected_patch,expected_confirmation_seeds",
    [
        (STRIDE8, (16, 8), (3311, 3312, 3313, 3314, 3315)),
        (STRIDE24, (24, 24), (3371, 3372, 3373, 3374, 3375)),
    ],
)
def test_release_layout_derives_registered_highdim_contract(
    path: Path, expected_patch: tuple[int, int], expected_confirmation_seeds: tuple[int, ...]
):
    layout = _release_layout(_load(path))
    assert layout["candidate_contract"]["patch_len"] == expected_patch[0]
    assert layout["candidate_contract"]["patch_stride"] == expected_patch[1]
    assert layout["confirmation_seeds"] == expected_confirmation_seeds
    expected_selection_seeds = (3291, 3292, 3293) if expected_patch == (16, 8) else (3361, 3362, 3363)
    assert layout["selection_seeds"] == expected_selection_seeds
    assert layout["baseline_models"] == (
        "DLinear", "NLinear", "RLinear", "PatchTST", "TimesNet", "iTransformer", "Mamba2",
    )


def test_release_layout_rejects_mismatched_confirmation_seeds():
    plan = _load(STRIDE24)
    plan["phases"][2]["seeds"] = [1, 2, 3, 4, 5]
    with pytest.raises(ValueError, match="baseline confirmation seeds"):
        _release_layout(plan)
