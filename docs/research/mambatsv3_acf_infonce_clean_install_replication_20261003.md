# Clean-Install Replication

This replication uses a newly created Python 3.9 conda environment rather than
a clone of the project environment. PyTorch, NumPy, pandas, scikit-learn,
transformers, CUDA 12.8 nvcc, and the CUDA runtime were installed into that
environment. `mamba-ssm` and `causal-conv1d` use locally repacked, SHA-256
locked cp39 binary artifacts from the locked original distribution because the
upstream ABI-specific wheel URL was unreachable during installation.

The environment passes a GPU `Mamba(d_model=24, d_state=8, d_conv=2)` forward
smoke. This is a clean-install plus binary-artifact replication, not an
independent-machine or from-source binary build.

The frozen PEMS-BAY H192/10% active/null protocol is rerun with new seed 5611.
Source and target settings, the validation-first test gate, and GPU guard are
identical to the primary confirmation. The replication is not pooled with its
three-seed effect estimate.
