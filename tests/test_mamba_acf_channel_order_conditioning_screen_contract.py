import json
import tempfile
from pathlib import Path

import numpy as np

from scripts import analyze_mamba_acf_channel_order_conditioning_screen as analyzer
from scripts.run_training_plan import _job_specs, _load_json, _validate_plan


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "docs/research/mamba_acf_channel_order_conditioning_screen_plan.json"


def test_registered_plan_is_three_arm_and_exactly_18_jobs():
    plan = _load_json(PLAN_PATH)
    assert _validate_plan(plan) == []
    assert [phase["name"] for phase in plan["phases"]] == [
        "order_input_control_h96",
        "order_strength_h96",
        "conditioning_input_h96",
    ]
    assert sum(len(_job_specs(plan, phase, {}, allow_unresolved=True)) for phase in plan["phases"]) == 18


def test_order_replay_matches_model_sort_rule_and_input_identity():
    values = np.zeros((4, 5), dtype=np.float32)
    values[:, 1] = [0.2, 0.9, 0.6, 0.6]
    values[:, 3] = [0.4, 0.1, 0.2, -0.1]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "acf_prior.npz"
        np.savez(path, acf_values=values, lags=np.array([24, 48, 72, 96, 120]),
                 acf_full_values=np.zeros((4, 121)), acf_source=np.array("raw"))
        strength = analyzer._order_activation(path, {"mamba_acf_channel_order": "strength"}, "x")
        identity = analyzer._order_activation(path, {"mamba_acf_channel_order": "input"}, "x")
    assert strength["order_indices"] == [1, 2, 3, 0]
    assert strength["order_changed"] is True
    assert strength["target_scan_position"] == 2
    assert identity["order_indices"] == [0, 1, 2, 3]
    assert identity["order_is_identity"] is True


def test_command_contract_binds_order_and_conditioning_flag():
    base = {
        "command": [
            "python", "run.py", "--model", "MambaACFAutoCon",
            "--mamba_variable_mixer", "--mamba_use_acf_memory", "true",
            "--mamba_acf_relation_min_lag", "96", "--AutoCon", "--AutoCon_lambda", "0.03",
            "--mamba_acf_relation_loss", "--AutoCon_acf_source", "raw",
            "--mamba_acf_channel_order", "input",
        ]
    }
    base["command_sha256"] = analyzer.hashlib.sha256(
        json.dumps(base["command"], separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    analyzer._validate_command(base, {"executable": "MambaACFAutoCon", "memory": True,
                                      "relation": True, "lambda": 0.03,
                                      "calibrated": False, "order": "input", "conditioning": False}, "x")
    conditioned = dict(base)
    conditioned["command"] = list(base["command"]) + ["--mamba_acf_channel_conditioning"]
    conditioned["command_sha256"] = analyzer.hashlib.sha256(
        json.dumps(conditioned["command"], separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    analyzer._validate_command(conditioned, {"executable": "MambaACFAutoCon", "memory": True,
                                             "relation": True, "lambda": 0.03,
                                             "calibrated": False, "order": "input", "conditioning": True}, "x")


def test_contrast_contains_exact_per_dataset_pairs():
    rows = []
    for arm, offset in (("input_control", 0.0), ("strength", 0.1), ("conditioning", -0.2)):
        for dataset in analyzer.EXPECTED_DATASETS:
            for seed in analyzer.EXPECTED_SEEDS:
                rows.append({"arm": arm, "dataset": dataset, "seed": seed,
                             "mse": 1.0 + offset, "mae": 0.5 + offset})
    contrast = analyzer._make_contrast(rows, "input_control", "strength")
    assert len(contrast["pairs"]) == 6
    assert set(contrast["by_dataset"]) == {"ETTh1", "traffic"}
    assert all(item["n"] == 3 for item in contrast["by_dataset"].values())
    assert np.isclose(contrast["by_dataset"]["ETTh1"]["metrics"]["mse"]["mean"], 0.1)
