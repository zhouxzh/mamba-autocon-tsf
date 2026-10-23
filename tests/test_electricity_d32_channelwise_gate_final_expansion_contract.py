from pathlib import Path


def test_electricity_final_expansion_has_fixed_five_seed_cap_and_stop_rule():
    text = Path("scripts/run_electricity_d32_channelwise_gate_final_expansion.py").read_text(encoding="utf-8")
    for token in ("SEEDS = (6094, 6095, 6096, 6097, 6098)", "prior_seeds", "stop_rule", "same_seed_pairing", "--evaluate-test"):
        assert token in text


def test_electricity_final_audit_combines_only_registered_ten_seed_pairs():
    text = Path("scripts/audit_electricity_d32_channelwise_gate_final_expansion.py").read_text(encoding="utf-8")
    for token in ("PRIOR_SEEDS", "FRESH_SEEDS", "_official_patchtst_patch_event.json", "final expansion registration failure", "final_precision_expansion", "noninferior_at_one_se"):
        assert token in text
