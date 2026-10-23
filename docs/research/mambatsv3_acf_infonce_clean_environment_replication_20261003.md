# Clean-Interpreter Replication

This is a same-machine independent-interpreter replication of the frozen
PEMS-BAY H192/10%-label local-cluster ACF-InfoNCE confirmation. It is not a
second-machine reproduction and is excluded from the primary three-seed effect
estimate.

## Locked Contract

| Item | Value |
| --- | --- |
| Interpreter | `/home/zhong/anaconda3/envs/acf_infonce_repro_20261003/bin/python` |
| Environment | Python 3.9.25, PyTorch 2.8.0+cu128, mamba-ssm 2.3.1 |
| Target | PEMS-BAY, H=192, chronological 60/20/20, 10% target labels |
| Sources | PEMS07, PEMS08, PEMS04 |
| Pairing | `channel_coverage` support 0.5, `contrastive_cluster`, local packing |
| Arms | active relation weights 0.1/0.1; exact-null weights 0/0 |
| Seed | 5610 |
| Test gate | Read test only if active validation MSE <= exact-null validation MSE |

All source and target training remains GPU-guarded at 78C / 85% fan with four
CPU threads. Source, target, and dataset SHA-256 values are stored in the
replication manifest.
