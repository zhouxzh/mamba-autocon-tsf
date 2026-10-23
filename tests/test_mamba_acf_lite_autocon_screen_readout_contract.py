import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_mamba_acf_lite_autocon_screen.py"
ARTIFACT_ROOT = ROOT / "artifacts" / "exploratory" / "mamba_acf_lite_autocon_screen_20260924"


def test_lite_autocon_readout_requires_complete_validation_only_pairs(tmp_path):
    output = tmp_path / "readout"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(ARTIFACT_ROOT), "--output-dir", str(output)],
        check=True,
        text=True,
        capture_output=True,
        cwd=ROOT,
    )
    assert "'rows': 12" in result.stdout
    assert "'test_metrics_produced': False" in result.stdout
    with (output / "paired_summary.csv").open(newline="", encoding="utf-8") as handle:
        pairs = list(csv.DictReader(handle))
    assert len(pairs) == 6
    assert {(row["dataset"], row["seed"]) for row in pairs} == {
        (dataset, str(seed)) for dataset in ("ETTh1", "traffic") for seed in (2021, 2022, 2023)
    }
    text = (output / "readout.md").read_text(encoding="utf-8")
    assert "Validation-only paired readout" in text
    assert "No test metric was read or produced" in text
