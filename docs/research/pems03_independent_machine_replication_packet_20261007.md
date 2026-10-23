# PEMS03 Independent-Machine Replication Packet

The strict PEMS03 H192/25% Channel-Patch protocol now has a portable runner
for a genuinely independent GPU host. It rebuilds the protocol from explicit
arguments rather than local historical events: PEMS04/08/07 sources, PEMS04/08
initial checkpoints, PEMS03 target exclusion, frozen target core, three fresh
seeds, and validation-only chronology. Every stage runs through the 78°C/85%
GPU guard and records code, data, and initialization-checkpoint SHA-256 hashes.

The runner intentionally does not self-certify independence. The operator must
supply a machine identifier, and a reviewer must explicitly supply the same
verified identifier to the audit command after confirming that host is physically
independent from this workstation.

On the independent host, with the repository, named data files, and initial
checkpoints present at the same relative paths:

```bash
python scripts/run_pems03_channel_patch_independent_replication.py \
  --python /path/to/python --machine-id HOST_IDENTIFIER

python scripts/audit_pems03_channel_patch_independent_replication.py \
  --artifact-root artifacts/reproducibility/pems03_channel_patch_independent_machine_20261007 \
  --verified-independent-machine-id HOST_IDENTIFIER
```

No test reader is part of this packet. Results are admissible only after the
host independence, source hashes, and generated artifact are manually reviewed.

Current local packet manifest:
`artifacts/reproducibility/pems03_independent_machine_packet_20261007/manifest.json`

It is explicitly marked `packet_only` and `unverified_operator_claim`; it is
not independent-machine evidence. It was subsequently executed on the verified
independent host `zhong-Precision-Tower-7810` (RTX 3060 Ti, Torch 2.6/CUDA
12.6, Mamba 2.2.4), which is distinct from the local `aceserver2` host.
The three remote seeds reproduce Channel-Patch versus forecast-only by
`-0.406%`, 3/3 wins, one-SE ratio upper `0.99662`, with no test read and peak
GPU telemetry of 64°C/36% fan. See
`artifacts/reproducibility/pems03_channel_patch_independent_machine_20261007/independent_audit.json`.
