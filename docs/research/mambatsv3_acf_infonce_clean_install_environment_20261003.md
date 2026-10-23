# Clean-Install Environment Notes

The validated clean-install environment uses Python 3.9.25, PyTorch
2.8.0+cu128, CUDA runtime/compiler 12.8, `mamba-ssm 2.3.1`, and
`causal-conv1d 1.6.2.post1`. The core Python requirements are listed in
`reproducibility/clean_install_requirements_20261003.txt`.

The cp39 Mamba and causal-conv binary wheels in
`artifacts/reproducibility_wheels_20261003/` are local artifacts repacked from
the locked original distribution, with SHA-256 hashes recorded in the
requirements file and clean-install manifest. Upstream Mamba's ABI-specific
wheel URL was unreachable during the clean-install attempt; therefore this is
not a from-source binary build.

Before running a frozen replication, execute:

```bash
python scripts/verify_clean_install_environment.py \
  --output artifacts/reproducibility/clean_environment_check.json
```

The verifier checks wheel hashes, fixed Mamba/PyTorch versions, CUDA
availability, and an actual CUDA Mamba forward smoke. An independent machine
replication remains required for a stronger submission-level reproducibility
claim.

The complete installed dependency freeze is
`reproducibility/clean_install_pip_freeze_20261003.txt` (SHA-256
`cb6dfe938735a25f879fb7bfd17d2fc8b6af60b238d44e62a4c217c1449d35a3`).

For an independent machine with a copied repository and data artifacts, run
`reproducibility/bootstrap_clean_install_20261003.sh /absolute/env/path` to
create and preflight the environment, then run
`reproducibility/run_independent_machine_replication_20261003.sh
/absolute/env/path/bin/python`. Both scripts first validate the frozen source,
dataset, and binary-artifact checksums. The launcher preserves the
validation-first test gate and GPU temperature/fan guard.

The launcher was executed end-to-end on the same clean-install host as a
rehearsal using seed 5612. It verified checksums and preflight, completed all
six guarded stages, and obtained active/null held-out MSE 2.631365/2.667106
(-1.340%). This validates the launcher workflow but is not an independent
machine replication and remains excluded from the primary effect estimate.
