import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_graph_spatial_mambatsv3_screen import _summary, _validate_phase_validation


def _row(phase, dataset, seed, model, value):
    return {
        "phase": phase,
        "dataset": dataset,
        "seed": seed,
        "model": model,
        "best_val_mse": value,
    }


class GraphSpatialScreenReaderTests(unittest.TestCase):
    def _rows(self, treatment=0.98, graph_linear=0.99, external=0.97):
        rows = []
        for dataset in ("metr_la", "pems_bay"):
            for seed in (2511, 2512, 2513):
                rows.extend([
                    _row("graph_off", dataset, seed, "MambaTSv3AutoCon", 1.0),
                    _row("graph_on", dataset, seed, "MambaTSv3AutoCon", treatment),
                    _row("graph_dlinear", dataset, seed, "GraphDLinear", graph_linear),
                ])
                for model in ("DLinear", "PatchTST", "TimeMixer", "iTransformer", "MambaTSv3"):
                    rows.append(_row("external", dataset, seed, model, external))
        return rows

    def test_summary_applies_paired_graph_and_external_gates(self):
        summary = _summary(self._rows())
        self.assertEqual(len(summary), 2)
        self.assertTrue(all(item["passes_dataset_gate"] for item in summary))
        self.assertEqual({item["graph_on_wins_vs_off"] for item in summary}, {3})

    def test_summary_rejects_large_paired_degradation(self):
        rows = self._rows(treatment=1.03)
        summary = _summary(rows)
        self.assertTrue(all(not item["passes_graph_gate"] for item in summary))
        self.assertTrue(all(not item["passes_dataset_gate"] for item in summary))

    def test_reader_accepts_strict_validator_phase_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            phase = "graph_phase"
            manifest = root / phase / "logs" / "job_manifest.jsonl"
            manifest.parent.mkdir(parents=True)
            source_hash = "a" * 64
            record = {"source_tree_sha256_before": source_hash}
            manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
            manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
            provenance = root / phase / "provenance.json"
            provenance.write_text(json.dumps({"source_tree_sha256": source_hash}), encoding="utf-8")
            validation = {
                "valid": True, "phase": phase, "provenance": str(provenance),
                "legacy_unknown_provenance": False, "expected_jobs": 1,
                "manifest_unique_jobs": 1, "malformed_manifest_lines": 0,
                "status_counts": {"ok": 1}, "manifest_sha256": manifest_hash,
            }
            (root / phase / "phase_validation.json").write_text(
                json.dumps(validation), encoding="utf-8"
            )
            self.assertEqual(_validate_phase_validation(root, phase, 1, [record]), source_hash)


if __name__ == "__main__":
    unittest.main()
