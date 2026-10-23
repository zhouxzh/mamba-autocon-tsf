import json
import tempfile
import unittest
from pathlib import Path

from scripts import summarize_multidataset_baseline_screen as analyzer


def _row(dataset, horizon, model, seed, mse, elapsed=1.0):
    return {
        "dataset": dataset,
        "horizon": horizon,
        "model": model,
        "seed": seed,
        "best_val_mse": mse,
        "best_epoch": 1,
        "epochs_ran": 1,
        "elapsed_sec": elapsed,
        "job_id": f"{dataset}_{horizon}_{model}_{seed}",
        "config_path": "config.json",
        "checkpoint_path": "loss_logs.json",
    }


class MultidatasetBaselineScreenContractTests(unittest.TestCase):
    def test_group_summary_keeps_dataset_horizon_and_model(self):
        rows = [
            _row("ETTh1", 96, "DLinear", 2021, 1.0),
            _row("ETTh1", 96, "DLinear", 2022, 3.0),
            _row("ETTh1", 96, "Mamba1", 2021, 2.0),
            _row("ETTh1", 96, "Mamba1", 2022, 4.0),
        ]
        summary = analyzer.group_summary(rows)
        index = {(item["dataset"], item["horizon"], item["model"]): item for item in summary}
        self.assertEqual(index[("ETTh1", 96, "DLinear")]["n_seeds"], 2)
        self.assertAlmostEqual(index[("ETTh1", 96, "DLinear")]["best_val_mse_mean"], 2.0)
        self.assertAlmostEqual(index[("ETTh1", 96, "Mamba1")]["best_val_mse_sd"], 2 ** 0.5)

    def test_paired_rows_uses_exact_seed_coordinates(self):
        rows = [
            _row("ETTh1", 96, "DLinear", 2021, 2.0),
            _row("ETTh1", 96, "DLinear", 2022, 2.5),
            _row("ETTh1", 96, "Mamba1", 2021, 1.0),
            _row("ETTh1", 96, "Mamba1", 2022, 3.0),
        ]
        paired = analyzer.paired_rows(rows, references=("DLinear",))
        self.assertEqual(len(paired), 1)
        self.assertEqual(paired[0]["n_pairs"], 2)
        self.assertEqual(paired[0]["candidate_wins"], 1)
        self.assertAlmostEqual(paired[0]["mean_delta_mse"], -0.25)

    def test_manifest_rejects_duplicate_job_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            record = {"job_id": "same", "status": "ok"}
            path.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate manifest job"):
                analyzer.read_manifest(path)


if __name__ == "__main__":
    unittest.main()
