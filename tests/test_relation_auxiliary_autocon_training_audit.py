"""Regression coverage for the read-only AutoCon training audit."""

from pathlib import Path

from scripts.audit_relation_auxiliary_autocon_training import read_phase


def test_relation_auxiliary_candidates_have_active_train_only_autocon_objective():
    highdim_root = Path("artifacts/exploratory/mambatsv3_highdim_relation_auxiliary_only_h96_validation_20260928/candidate")
    crossdomain_root = Path("artifacts/exploratory/mambatsv3_relation_auxiliary_crossdomain_h96_validation_20260929/candidate")
    solar_root = Path("artifacts/exploratory/mambatsv3_relation_auxiliary_solar_h96_validation_20260929/candidate")
    highdim_source, highdim = read_phase(
        highdim_root,
        "relation_auxiliary_only_candidate_h96_validation",
        6,
        {"electricity", "traffic"},
        {3291, 3292, 3293},
    )
    crossdomain_source, crossdomain = read_phase(
        crossdomain_root,
        "candidate_h96_validation",
        6,
        {"ETTh1", "weather"},
        {3331, 3332, 3333},
    )
    solar_source, solar = read_phase(
        solar_root,
        "candidate_solar_h96_validation",
        3,
        {"solar"},
        {3341, 3342, 3343},
    )
    assert highdim_source == crossdomain_source == solar_source
    assert len(highdim) == len(crossdomain) == 6
    assert len(solar) == 3
    for row in highdim + crossdomain + solar:
        assert row["autocon_lambda"] == 0.001
        assert row["autocon_loss_first"] > 0.0
        assert row["autocon_loss_last"] > 0.0
        assert row["mean_weighted_autocon_fraction_pct"] > 0.0
        assert row["max_loss_identity_residual"] < 1e-6
