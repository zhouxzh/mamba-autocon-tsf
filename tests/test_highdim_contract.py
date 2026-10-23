import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.validate_highdim_contract import _canonical_report_hash, validate_plan


class HighdimContractTests(unittest.TestCase):
    def _fixture(self, quality_pass=True, expected_report_hash=None,
                 expected_content_hash=None, bad_row=False):
        root = Path(tempfile.mkdtemp())
        dataset = root / "dataset" / "electricity"
        report_dir = root / "artifacts" / "quality"
        dataset.mkdir(parents=True)
        report_dir.mkdir(parents=True)
        csv_path = dataset / "electricity.csv"
        rows = ["date,0,OT", "2024-01-01 00:00:00,1,2", "2024-01-01 01:00:00,3,4"]
        if bad_row:
            rows[-1] = "2024-01-01 01:00:00,3"
        csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        data_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        report = {
            "schema_version": 1,
            "generated_at_utc": "2026-08-28T00:00:00+00:00",
            "dataset": "electricity",
            "path": str(csv_path),
            "bytes": csv_path.stat().st_size,
            "sha256": data_hash,
            "schema": {
                "time_column": "date",
                "target": "OT",
                "columns": ["date", "0", "OT"],
                "value_columns": ["0", "OT"],
                "rows": 2,
                "channels": 2,
                "expected_channels": 2,
            },
            "time": {"expected_frequency": "h"},
            "quality_pass": quality_pass,
            "quality_errors": [] if quality_pass else ["fixture_error"],
        }
        report_path = report_dir / "electricity.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        report_hash = hashlib.sha256(report_path.read_bytes()).hexdigest()
        contract = {
            "relative_path": "dataset/electricity/electricity.csv",
            "channels_including_target": 2,
            "frequency": "h",
            "time_column": "date",
            "target": "OT",
            "sha256": data_hash,
            "quality_report": "artifacts/quality/electricity.json",
        }
        if expected_report_hash is not None:
            contract["quality_report_sha256"] = expected_report_hash
        if expected_content_hash is not None:
            contract["quality_report_content_sha256"] = expected_content_hash
        plan = root / "plan.json"
        plan.write_text(json.dumps({"dataset_contract": {"electricity": contract}}), encoding="utf-8")
        return root, plan, report_hash

    def test_valid_contract_checks_report_binding_and_emits_report_hash(self):
        root, plan, report_hash = self._fixture()
        result = validate_plan(plan, root)
        item = result["datasets"]["electricity"]
        self.assertTrue(result["valid"])
        self.assertTrue(item["valid"])
        self.assertEqual(item["quality_report"]["sha256"], report_hash)
        self.assertFalse(item["quality_report"]["hash"]["registered"])

    def test_bad_row_or_quality_report_fails_strictly(self):
        root, plan, _ = self._fixture(quality_pass=False, bad_row=True)
        result = validate_plan(plan, root)
        self.assertFalse(result["valid"])
        self.assertIn("electricity:csv_row_width_mismatch", result["errors"])
        self.assertIn("electricity:quality_report_quality_pass_false", result["errors"])

    def test_registered_report_hash_mismatch_is_an_error(self):
        root, plan, report_hash = self._fixture(expected_report_hash="0" * 64)
        result = validate_plan(plan, root)
        self.assertFalse(result["valid"])
        self.assertIn("electricity:quality_report_sha256_mismatch", result["errors"])
        self.assertNotEqual(report_hash, "0" * 64)

    def test_content_hash_ignores_generated_time_and_path(self):
        root, plan, _ = self._fixture()
        report_path = root / "artifacts" / "quality" / "electricity.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        content_hash = _canonical_report_hash(report)
        contract = json.loads(plan.read_text(encoding="utf-8"))
        contract["dataset_contract"]["electricity"]["quality_report_content_sha256"] = content_hash
        plan.write_text(json.dumps(contract), encoding="utf-8")
        report["generated_at_utc"] = "2026-08-29T12:34:56+00:00"
        report["path"] = "/another/machine/electricity.csv"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        result = validate_plan(plan, root)
        self.assertTrue(result["valid"])
        info = result["datasets"]["electricity"]["quality_report"]
        self.assertTrue(info["content_hash"]["registered"])
        self.assertTrue(info["content_hash"]["match"])

    def test_content_hash_mismatch_is_an_error(self):
        root, plan, _ = self._fixture(expected_content_hash="0" * 64)
        result = validate_plan(plan, root)
        self.assertFalse(result["valid"])
        self.assertIn("electricity:quality_report_content_sha256_mismatch", result["errors"])

    def test_cli_strict_returns_two_and_writes_json(self):
        root, plan, _ = self._fixture(quality_pass=False)
        output = root / "contract.json"
        completed = subprocess.run(
            [sys.executable, "scripts/validate_highdim_contract.py", "--plan", str(plan),
             "--repo-root", str(root), "--strict", "--output", str(output)],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertTrue(output.is_file())
        self.assertFalse(json.loads(output.read_text())["valid"])

    def test_non_object_contract_entry_is_reported(self):
        root = Path(tempfile.mkdtemp())
        plan = root / "plan.json"
        plan.write_text(json.dumps({"dataset_contract": {"electricity": []}}), encoding="utf-8")
        result = validate_plan(plan, root)
        self.assertFalse(result["valid"])
        self.assertIn("electricity:contract_entry_not_object", result["errors"])


if __name__ == "__main__":
    unittest.main()
