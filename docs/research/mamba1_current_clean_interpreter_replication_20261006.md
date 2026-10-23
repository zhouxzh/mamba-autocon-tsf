# Current-Source Clean-Interpreter Replication

Status: completed same-machine clean-interpreter validation replication. It is
explicitly not an independent-machine or from-source-binary replication, and
it does not read target test rows.

The existing `acf_infonce_clean_install_20261003` Python 3.9 environment was
preflighted against the locked CUDA/Mamba wheels. Because current pretraining
and target-adaptation scripts have legitimately evolved since the 2026-10-03
checksum list, this replication writes a new manifest of SHA-256 hashes for
the current MambaTS model, pretraining/finetuning/guard scripts, and the four
PEMS source/target data files before training.

Using the current PEMS-BAY H192/10% protocol under the clean interpreter:

| Arm | Validation MSE |
| --- | ---: |
| ACF active | 2.760915 |
| Exact null | 2.820535 |
| Active vs null | -2.114% |

The active source relation anchor fraction was 1.0 in each pretraining epoch.
Both source pretraining and frozen-core target adaptation completed under the
78C/85% guard; observed peak temperature was 53C. The audit verifies that
current source/data hashes match the manifest and that neither arm emitted a
test metric.

This validates the current experiment's execution on a separately installed
interpreter and CUDA/Mamba package set. It must not be presented as an
independent-machine result.

Artifacts:

- `artifacts/reproducibility/current_clean_interpreter_validation_20261006/manifest.json`
- `artifacts/reproducibility/current_clean_interpreter_validation_20261006/audit.json`
- `scripts/run_mambatsv3_current_clean_env_validation_replication.py`
- `scripts/audit_mambatsv3_current_clean_env_validation_replication.py`
