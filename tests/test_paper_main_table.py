from scripts.build_paper_main_table import render_markdown


def test_render_main_table_preserves_scope_and_seed_wins():
    text = render_markdown(
        [
            {
                "release": "Example",
                "dataset": "x",
                "horizon": 12,
                "baseline": "PatchTST",
                "candidate_mse": 0.1,
                "baseline_mse": 0.2,
                "mean_mse_change_pct": -50.0,
                "candidate_wins": 5,
                "n": 5,
                "upper_ratio_95_one_sided": 0.9,
                "noninferior_ratio_1p02": True,
                "candidate_elapsed_mean_sec": 1.0,
                "baseline_elapsed_mean_sec": 2.0,
                "candidate_elapsed_faster": True,
            }
        ]
    )
    assert "5/5" in text
    assert "not support universal accuracy or speed claims" in text
