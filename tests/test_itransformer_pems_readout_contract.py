import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_itransformer_pems_baseline_extension.py"
ITRANSFORMER_ROOT = ROOT / "artifacts" / "exploratory" / "itransformer_pems_baseline_extension_20260924"
MAMBA_ROOT = ROOT / "artifacts" / "exploratory" / "mamba_ts_v3_channel_adapter_confirmation_20260923"
TIMEMIXER_ROOT = ROOT / "artifacts" / "exploratory" / "timemixer_pems_baseline_extension_20260923"


def test_itransformer_readout_preserves_validation_only_reference_boundary(tmp_path):
    output = tmp_path / "readout"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--itransformer-root", str(ITRANSFORMER_ROOT),
            "--mamba-reference-rows", str(MAMBA_ROOT / "readout" / "validation_rows.csv"),
            "--mamba-reference-provenance", str(MAMBA_ROOT / "dlinear" / "provenance.json"),
            "--timemixer-rows", str(TIMEMIXER_ROOT / "readout" / "validation_rows.csv"),
            "--timemixer-provenance", str(TIMEMIXER_ROOT / "timemixer" / "provenance.json"),
            "--output-dir", str(output),
        ],
        check=True,
        text=True,
        capture_output=True,
        cwd=ROOT,
    )
    assert "'itransformer_rows': 40" in result.stdout
    assert "'reference_rows': 240" in result.stdout
    assert "'test_metrics_produced': False" in result.stdout
    with (output / "pairwise_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 48
    assert {row["reference_model"] for row in rows} == {
        "DLinear", "PatchTST", "TimesNet", "Mamba2", "MambaTSv3AutoCon-null", "TimeMixer",
    }
    assert "different source snapshots" in (output / "readout.md").read_text(encoding="utf-8")
