"""Smoke and contract tests for optional Solar/PEMS numeric-array inputs."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from data_provider.data_factory import data_provider
from data_provider.data_loader import Dataset_Array, load_numeric_array_file
from scripts.run_training_plan import (
    DATASETS,
    _validate_numeric_array_contract,
)


def _args(**overrides):
    values = {
        "data": "solar",
        "embed": "timeF",
        "task_name": "long_term_forecast",
        "model": "Mamba2",
        "root_path": ".",
        "data_path": "solar.txt",
        "data_format": "txt",
        "data_key": "data",
        "data_feature_index": 0,
        "target_index": -1,
        "seq_len": 8,
        "label_len": 4,
        "pred_len": 4,
        "features": "M",
        "target": "OT",
        "freq": "h",
        "seasonal_patterns": "Monthly",
        "train_ratio": 0.6,
        "batch_size": 2,
        "num_workers": 0,
        "drop_last_train": False,
        "mamba_acf_relation_loss": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class ArrayDataLoaderTests(unittest.TestCase):
    def test_npz_three_dimensional_feature_selection_and_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = np.arange(60, dtype=np.float32).reshape(10, 3, 2)
            np.savez(root / "pems.npz", data=raw)
            dataset = Dataset_Array(
                root_path=str(root), data_path="pems.npz", data_format="npz",
                data_feature_index=1, flag="train", size=[2, 1, 1],
                features="M", timeenc=1, freq="h", train_ratio=0.6,
            )
            self.assertEqual(dataset.data_x.shape, (6, 3))
            self.assertEqual(dataset.data_stamp.shape, (6, 4))
            self.assertEqual(len(dataset), 4)
            self.assertTrue(np.all(np.isfinite(dataset.data_x)))

    def test_txt_is_header_free_numeric_and_supports_target_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "solar.txt"
            path.write_text("1 10\n2 20\n3 30\n4 40\n5 50\n6 60\n7 70\n8 80\n9 90\n10 100\n", encoding="utf-8")
            values = load_numeric_array_file(path, data_format="txt")
            self.assertEqual(values.shape, (10, 2))
            dataset = Dataset_Array(
                root_path=str(root), data_path="solar.txt", data_format="txt",
                target_index=0, flag="test", size=[2, 1, 1],
                features="S", freq="h", train_ratio=0.6,
            )
            self.assertEqual(dataset.data_x.shape[-1], 1)
            self.assertEqual(dataset.selected_target_index, 0)

    def test_csv_numeric_discards_only_the_timestamp_column(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "traffic.csv"
            path.write_text(
                ",sensor_a,sensor_b\n"
                "2012-03-01 00:00:00,1.0,10.0\n"
                "2012-03-01 00:05:00,2.0,20.0\n"
                "2012-03-01 00:10:00,3.0,30.0\n"
                "2012-03-01 00:15:00,4.0,40.0\n"
                "2012-03-01 00:20:00,5.0,50.0\n"
                "2012-03-01 00:25:00,6.0,60.0\n"
                "2012-03-01 00:30:00,7.0,70.0\n"
                "2012-03-01 00:35:00,8.0,80.0\n"
                "2012-03-01 00:40:00,9.0,90.0\n"
                "2012-03-01 00:45:00,10.0,100.0\n",
                encoding="utf-8",
            )
            values = load_numeric_array_file(path, data_format="csv_numeric")
            self.assertEqual(values.shape, (10, 2))
            self.assertEqual(float(values[-1, -1]), 100.0)
            dataset = Dataset_Array(
                root_path=str(root), data_path="traffic.csv", data_format="csv_numeric",
                flag="train", size=[2, 1, 1], features="M", timeenc=1,
                freq="t", data_interval_minutes=5, train_ratio=0.6,
                data_start_timestamp="2012-03-01 00:00:00",
            )
            self.assertEqual(dataset.channel_count, 2)
            self.assertEqual(str(dataset.data_start_timestamp), "2012-03-01 00:00:00")

    def test_minute_array_requires_declared_sampling_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savetxt(root / "pems.txt", np.arange(40, dtype=np.float32).reshape(20, 2))
            with self.assertRaisesRegex(ValueError, "requires a positive data_interval_minutes"):
                Dataset_Array(
                    root_path=str(root), data_path="pems.txt", data_format="txt",
                    flag="train", size=[2, 1, 1], features="M", freq="t",
                    train_ratio=0.6,
                )

    def test_missing_file_and_wrong_npz_key_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "numeric array dataset file is missing"):
                Dataset_Array(
                    root_path=str(root), data_path="missing.npz", data_format="npz",
                    flag="train", size=[2, 1, 1], features="M", freq="h",
                )
            np.savez(root / "pems.npz", values=np.ones((10, 2), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "does not contain data_key 'data'"):
                Dataset_Array(
                    root_path=str(root), data_path="pems.npz", data_format="npz",
                    flag="train", size=[2, 1, 1], features="M", freq="h",
                )

    def test_factory_passes_array_options_only_to_array_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savetxt(root / "solar.txt", np.arange(40, dtype=np.float32).reshape(20, 2))
            args = _args(root_path=str(root), data_path="solar.txt")
            dataset, loader = data_provider(args, "val")
            self.assertIsInstance(dataset, Dataset_Array)
            self.assertEqual(loader.batch_size, 2)

            # Existing CSV classes retain their historical constructor call;
            # array-only kwargs must not leak into a patched legacy class.
            class LegacyDataset:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs
                    self.size = 3

                def __len__(self):
                    return self.size

                def __getitem__(self, index):
                    return index

            csv_args = _args(data="legacy", data_format="npz")
            with patch.dict(
                __import__("data_provider.data_factory", fromlist=["data_dict"]).data_dict,
                {"legacy": LegacyDataset}, clear=False,
            ):
                legacy, _ = data_provider(csv_args, "val")
            self.assertNotIn("data_format", legacy.kwargs)
            self.assertNotIn("target_index", legacy.kwargs)

    def test_registry_reports_missing_optional_dataset_without_downloading(self):
        contract = {
            "relative_path": "dataset/PEMS03/PEMS03.npz",
            "format": "npz",
            "data_key": "data",
            "frequency": "t",
            "time_column": None,
            "target_index": DATASETS["pems03"]["target_index"],
            "data_interval_minutes": DATASETS["pems03"]["data_interval_minutes"],
            "channels_including_target": DATASETS["pems03"]["channels"],
            "sha256": "0" * 64,
        }
        # The real PEMS files are present in this checkout.  Patch the
        # registry resolver to keep this test focused on the missing-file
        # contract without deleting a user-provided benchmark download.
        with patch(
            "scripts.run_training_plan._dataset_path",
            return_value=Path(tempfile.gettempdir()) / "missing-PEMS03.npz",
        ):
            errors = _validate_numeric_array_contract("pems03", contract)
        self.assertTrue(any("missing dataset file" in error for error in errors))
        self.assertTrue(any("PEMS03.npz" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
