from pathlib import Path

from scripts.audit_dcrnn_evidence_consistency import close


def test_close_respects_reported_precision():
    assert close(0.27325131538548814, 0.273251)
    assert close(-5.2873570530275416, -5.287, 3)
    assert not close(0.27325131538548814, 0.273252)


def test_dcrnn_evidence_sources_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "artifacts/exploratory/metro_pemsbay_dcrnn_mambatsv3_h12_readout_20260930/paired_summary.csv").is_file()
    assert (root / "artifacts/exploratory/metro_pemsbay_dcrnn_mambatsv3_h12_resource_attestation_20260930/resource_rows.csv").is_file()
