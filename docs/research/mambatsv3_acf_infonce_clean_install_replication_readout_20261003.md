# Clean-Install Replication Readout

The clean-install PEMS-BAY H192/10% replication used a newly created Python
3.9 conda environment with a separately installed PyTorch CUDA runtime and
CUDA 12.8 compiler. It installed SHA-256 locked local cp39 binary artifacts
for `mamba-ssm 2.3.1` and `causal-conv1d 1.6.2.post1`; the upstream ABI wheel
URL was unreachable during the attempt, so this is not a from-source binary
build or an independent-machine replication.

With new seed 5611, validation chose active (2.767148) over exact-null
(2.815984), after which the read-only test results were 2.643805 versus
2.682384 (-1.438%). GPU Mamba smoke, source pretraining, target fine-tuning,
and validation-first test readers all completed under the temperature/fan
guard.

This supports clean-install execution reproducibility with locked binary
artifacts. It is excluded from the primary three-seed effect estimate and does
not close the independent-machine replication gate.
