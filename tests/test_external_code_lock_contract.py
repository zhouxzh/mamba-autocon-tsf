import json
from pathlib import Path


def test_external_code_lock_manifest_has_all_adapter_sources():
    path = Path("docs/research/mambatsv3_acf_infonce_external_code_locks_20261003.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload["sources"]) == {"patchtst", "itransformer", "timesnet", "timemixer"}
    for record in payload["sources"].values():
        assert len(record["commit"]) == 40
        assert len(record["model_sha256"]) == 64
        assert len(record["requirements_sha256"]) == 64


def test_external_code_lock_verifier_is_documented():
    verifier = Path("scripts/verify_external_code_locks.py").read_text(encoding="utf-8")
    guide = Path("docs/research/mambatsv3_acf_infonce_reproducibility_20261003.md").read_text(encoding="utf-8")
    assert "--checkout" in verifier
    assert "verify_external_code_locks.py" in guide
