# Graph-Spatial MambaTSv3 AutoCon Screen

## Purpose

This draft protocol tests a narrow hypothesis: METR-LA and PEMS-BAY may need their supplied directed sensor topology, rather than another topology-free channel mixer. The treatment adds exactly one sparse, row-normalized graph propagation to the observed seasonal window before the existing single MambaTSv3 scan. The same encoded state still supplies both the forecast and AutoCon representation losses.

The protocol separates three effects: ordinary temporal forecasting (`DLinear`, PatchTST, TimeMixer, iTransformer, and plain MambaTSv3), topology without Mamba/AutoCon (`GraphDLinear`), and topology inside the MambaTSv3+AutoCon candidate (paired graph off/on).

## Architecture Selection

This is deliberately not another generic Mamba variant. The official
[MambaTS implementation](https://github.com/XiudingCai/MambaTS-pytorch)
identifies variable-aware ordering/scanning as a distinct mechanism. The local
fixed-order v10 approximation and trainable-order v11 adaptation both failed
their ETTh1/H336 preflights, while the local S-Mamba/TimesMamba-inspired
bidirectional variable scan and the TimeMachine-inspired multi-path scan were
also stopped on accuracy or latency grounds (see
[`current status`](mamba_current_status.md)).

The official [ms-Mamba implementation](https://github.com/meric-karadag/ms-mamba)
uses multiple parallel Mamba blocks with different internal sampling rates.
That is a valid multi-scale reference, but it adds parallel SSM work and does
not address the observed METR-LA versus PEMS-BAY topology gap. It is therefore
not compatible with this screen's single-scan speed hypothesis. In contrast,
the supplied directed sensor graph is specific, observable input metadata for
both datasets. The treatment uses it once, before the unchanged MambaTSv3
scan, so graph off/on remains an interpretable one-factor comparison.

## Fixed Evidence Boundary

All 48 jobs are validation-only (`L=96`, `H=12`, three paired seeds `2511--2513`, METR-LA and PEMS-BAY). `skip_test=true` is mandatory. The graph contracts bind each traffic CSV's exact column order to the official directed adjacency through a safe NPZ plus SHA-256.

The plan is draft and requires human launch approval. It must not be executed by changing the plan status alone or by constructing an approval sidecar after the fact.

## Preflight Result

Prediction-only FP32 GPU preflight used RTX 5090 D, batch 32, 20 warm-up and 100 timed device-resident forwards. Graph Mamba measured `0.297 ms` on METR-LA and `0.339 ms` on PEMS-BAY; GraphDLinear measured `0.124/0.161 ms`. The prior same-shape preflight reported PatchTST `0.364/0.497 ms`, TimeMixer `0.986/1.330 ms`, and iTransformer `0.222/0.330 ms`.

Therefore the candidate remains faster than PatchTST and TimeMixer, but is slower than iTransformer on both datasets. Any future paper must state this limitation and cannot claim universal inference superiority. Resource records are under `artifacts/exploratory/metro_pemsbay_graph_spatial_preflight_20260927/`.

## Decision Rule

The graph-on candidate can be considered for a separately registered held-out test only when it improves validation MSE by at least 1% with at least two of three paired wins on each dataset, has no paired degradation greater than 2%, is no worse than both its graph-off control and GraphDLinear, and remains within 2% of the strongest external validation baseline. Failing any condition stops this branch.
