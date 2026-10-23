# Baseline Coverage Dashboard (2026-10-01)

This dashboard is a paper-planning artifact. It distinguishes frozen held-out evidence from validation-only evidence, resource-only measurements, and stopped partial baselines.

## Dataset Coverage

| Dataset/domain | Main candidate evidence | Strong baselines | Evidence level | Current interpretation |
|---|---|---|---|---|
| PEMS03/04/07/08 | MambaTSv3AutoCon H12 | DLinear, NLinear, PatchTST, TimesNet, TimeMixer, iTransformer, Mamba2 | Held-out, 28/28 noninferior cells | Strongest short-term traffic evidence; limited to registered PEMS protocol |
| METR-LA/PEMS-BAY | MambaTSv3AutoCon fusion H12 | DLinear, NLinear, PatchTST, TimeMixer, iTransformer, plain MambaTSv3 | Held-out, 12/12 noninferior cells | Prediction-connected fusion is supported on this non-graph traffic release |
| Weather | MambaTSv3AutoCon H96/H336/H720 | PatchTST, TimesNet, iTransformer, Mamba2 | Held-out, 12/12 noninferior cells | Strong non-traffic success domain; no universal claim |
| Electricity/Traffic | Relation-auxiliary-only MambaTSv3 H96 | DLinear, NLinear, RLinear, PatchTST, TimesNet, iTransformer, Mamba2, Mamba-1 | Held-out CPU-4 rerun, 13/14 noninferior cells | Strong high-dimensional evidence; Traffic/iTransformer is an explicit failure cell |
| ETTm1/ETTm2 | Stride-24 MambaTSv3 H96 | DLinear, RLinear, PatchTST, iTransformer, Mamba-1, TimesNet, Mamba2 | Validation-only, 24/24 per dataset | High-frequency transfer evidence; long-horizon failures remain |
| ETTh1/ETTh2 | Stride-24 MambaTSv3 and TimeMixer screens | RLinear, PatchTST, iTransformer, TimeMixer | Validation-only | Explicit low-dimensional and long-horizon failure boundary |
| Solar | Variable-axis Mamba and temporal MambaTSv3 | DLinear, NLinear, RLinear, PatchTST, TimesNet, iTransformer, Mamba2, TimeMixer | Validation-only | Variable-axis architecture is promising; temporal candidate has iTransformer failure boundary |
| Exchange-Rate/ILI | MambaTSv3, blend and classical baseline screens | DLinear, NLinear, PatchTST, TimesNet, iTransformer, classical models | Validation-only | Negative transfer boundaries; no selective success reporting |
| METR-LA/PEMS-BAY graph baselines | MambaTSv3 candidate plus STGCN, GraphWaveNet, AGCRN, optimized DCRNN | Canonical-style local graph implementations | Validation-only, complete 12/12 per registered matrix for STGCN/GraphWaveNet/AGCRN/DCRNN | Graph baseline coverage is complete at validation level; no held-out graph ranking |

## Model Coverage

| Model family | Data/horizon coverage | Current role | Main limitation |
|---|---|---|---|
| MambaTSv3AutoCon | Held-out PEMS, METR-LA/PEMS-BAY, Weather, Electricity/Traffic; broad validation | Principal candidate | Not universally better on low-dimensional or long-horizon datasets; one high-dimensional failure cell |
| Mamba-1 / Mamba-2 | PEMS, Electricity, Traffic, ETT and other validation screens | Plain architecture controls | Candidate is not uniformly faster than plain Mamba-1 |
| TimeMixer | ETTm2, ETTh2, Solar and traffic validation/resource screens | Strong multi-scale MLP comparator | Candidate loses at some long horizons and remains validation-only outside releases |
| PatchTST / TimesNet / iTransformer | Main held-out releases and broad validation | Strong external baselines | Linear models remain faster on many fixed shapes; Traffic/iTransformer is the strongest counterexample in the CPU-4 release |
| RLinear / NLinear / DLinear | Broad current-source matrices | Linear normalization and forecasting controls | Candidate cannot claim universal speed or accuracy dominance |
| STGCN / GraphWaveNet / AGCRN / DCRNN | METR-LA/PEMS-BAY H12 validation-only | Graph-temporal baseline coverage | Local canonical-style implementations; held-out graph release absent |
| MambaACF / variable-token / multi-scale / Mamba-2-mixer branches | Exploratory screens | Architecture mechanism probes | Most branches stopped by cross-domain or resource gates; Mamba-2 and independent-channel v2 are slower on matched high-dimensional resource shapes |

## Evidence Gates

1. `Held-out`: only the three frozen releases may support principal test claims.
2. `Validation-only`: useful for model selection and failure boundaries; never interchangeable with held-out evidence.
3. `Resource-only`: fixed-shape forward or parameter evidence; never training throughput or universal speed.
4. `Stopped`: retain the negative result and do not tune after observing the failure.
5. `AutoCon`: train-time nonzero relation loss proves participation, not an independent causal accuracy gain.

## Current Paper Scope

The defensible scope is a train-only AutoCon-connected MambaTSv3 forecaster that is competitive on the named PEMS, METR-LA/PEMS-BAY, Weather, Electricity and Traffic held-out releases, with bounded fixed-shape efficiency advantages over selected nonlinear baselines and explicit low-dimensional, long-horizon, graph-model and comparator-specific limitations.

The current high-dimensional H96 resource measurement preserves the same boundary: the relation-auxiliary-only candidate is faster than PatchTST/TimesNet on Electricity and Traffic, faster than iTransformer on Traffic, but slower than DLinear/NLinear/RLinear and iTransformer on Electricity. This is a comparator-specific resource observation, not a universal speed claim.

Source documents: [publication readiness audit](publication_readiness_audit_20260930.md), [paper evidence matrix](paper_evidence_matrix_20260928.md), [current status](mamba_current_status.md).
