from pathlib import Path


def test_independent_machine_launcher_checks_inputs_and_preserves_validation_first_gate():
    text = Path("reproducibility/run_independent_machine_replication_20261003.sh").read_text(encoding="utf-8")
    for token in ("sha256sum -c", "verify_clean_install_environment.py", "Validation gate failed", "evaluate_lowdata_checkpoint_test.py", "temperature-limit-c 78", 'ACTIVE_VAL=$("$PYTHON_BIN"'):
        assert token in text
