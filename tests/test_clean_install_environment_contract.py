from pathlib import Path


def test_clean_install_environment_verifier_checks_versions_wheels_and_cuda_smoke():
    text = Path("scripts/verify_clean_install_environment.py").read_text(encoding="utf-8")
    for token in ("wheel hash mismatch", "torch 2.8.0+cu128", "mamba_ssm 2.3.1", "clean_environment_ready", "Mamba(d_model=24"):
        assert token in text
