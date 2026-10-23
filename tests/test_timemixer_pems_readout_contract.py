import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_timemixer_pems_baseline_extension.py"
TIMEMIXER_ROOT = ROOT / "artifacts" / "exploratory" / "timemixer_pems_baseline_extension_20260923"
REFERENCE_ROOT = ROOT / "artifacts" / "exploratory" / "mamba_ts_v3_channel_adapter_confirmation_20260923"


def test_timemixer_readout_preserves_validation_only_reference_boundary(tmp_path):
    output = tmp_path / "readout"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--timemixer-root", str(TIMEMIXER_ROOT),
            "--reference-rows", str(REFERENCE_ROOT / "readout" / "validation_rows.csv"),
            "--reference-provenance", str(REFERENCE_ROOT / "dlinear" / "provenance.json"),
            "--output-dir", str(output),
        ],
        check=True,
        text=True,
        capture_output=True,
        cwd=ROOT,
    )
    assert "'timemixer_rows': 40" in result.stdout
    assert "'reference_rows': 200" in result.stdout
    assert "'test_metrics_produced': False" in result.stdout
    with (output / "pairwise_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 40
    assert {row["reference_model"] for row in rows} == {
        "DLinear", "PatchTST", "TimesNet", "Mamba2", "MambaTSv3AutoCon-null",
    }
    assert "different source snapshots" in (output / "readout.md").read_text(encoding="utf-8")
