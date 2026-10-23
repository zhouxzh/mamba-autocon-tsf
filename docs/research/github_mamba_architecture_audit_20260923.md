# GitHub Mamba Forecasting Structure Audit

## Purpose

This note records which public Mamba forecasting structures have a real
implementation counterpart in this repository, and whether another branch is
worth a bounded experiment. It is an architecture-selection note, not an
accuracy claim.

## Reviewed structures

| Public implementation | Core idea | Local counterpart | Current decision |
| --- | --- | --- | --- |
| [TimeMachine](https://github.com/Atik-Ahamed/TimeMachine) | Four Mamba paths combine global/local and channel-mixing/channel-independent cues | `MambaTM1/2` | Stopped: slower and not competitive on the H96 multi-dataset screen. |
| [S-Mamba](https://github.com/wzhwzhwzh0921/S-D-Mamba) / [TimesMamba](https://github.com/shaowen310/TimesMamba) | Bidirectional scan over variable tokens, optionally with FFN and RevIN | `MambaSM` | Stopped: ETTh1 signal only; weather, Exchange and ILI regressions and no latency advantage. |
| [Bi-Mamba4TS](https://github.com/Leopold2333/Bi-Mamba4TS) | SRA-selected channel-mixing/independent patch tokens, bidirectional Mamba+ and a token FFN | `BiPatchMamba` | Stopped before training: native Mamba-1 implementation of both public token axes misses the Traffic H96 latency gate. The original custom Mamba+ kernel cannot be built on this host without `nvcc`; no equivalence claim is made. |
| [MambaTS](https://github.com/XiudingCai/MambaTS-pytorch) | Variable-aware time scan and variable permutation training | `MambaTS` v1-v11 | v10 fixed-ACF VAST-lite and v11 learned-VPT/greedy-order adaptations both stopped on their ETTh1/H336 preflights; neither is an official reproduction. |
| [ms-Mamba](https://github.com/meric-karadag/ms-mamba) | Parallel Mamba blocks at distinct internal SSM sampling rates (Delta), then aligned fusion | No exact counterpart | Stopped before training: its implementation runs one main selective scan plus one external-Delta selective scan for every extra scale and disables the standard fused Mamba fast path. Even the minimal two-branch form cannot plausibly meet this project's `<=10%` latency gate; the published four-branch form is further outside the efficiency target. |
| [DPEM](https://github.com/HRFarsighted/DPEM) | Separate temporal/spatial Mamba views followed by cross-attention fusion | No counterpart | Excluded for this objective: cross-attention fusion introduces a quadratic token interaction and changes the speed target rather than improving the existing fast scan. |
| [SST](https://github.com/XiongxiaoXu/SST) | Mamba global expert plus local Transformer expert and adaptive routing | No counterpart | Excluded for this objective: its Transformer expert directly conflicts with the requirement to beat efficient baselines in computation. It is a useful accuracy comparator, not an efficient Mamba candidate. |

The existing `MambaACF.mamba_scales` feature is not an ms-Mamba equivalent: it
pools/downsamples input branches before scanning. ms-Mamba instead keeps the
same input resolution and changes the internal state-space sampling rates.

## 2026-09-27 MambaTS VPT/VAST implementation check

The official MambaTS repository was inspected at commit
`3b6797d9bf5178a490e1bb3c3e9f4d223f2af51d`. Its `VPT_mode=1` training path
randomly permutes variables and records transition costs. Its evaluation path
constructs an ordering from the accumulated adjacency matrix with the supplied
ATSP solver, then restores output channels. The earlier local random
`mamba_ts_variable_permutation` flag covered only the training shuffle, not
this learned inference order.

`MambaTSv10` adds a narrower, explicitly non-equivalent test: a fixed order is
constructed once from signed ACF profiles fitted on the training split and is
used in both train and evaluation. This preserves one fused Mamba scan and
avoids an inference-time solver. The mechanism is recorded as VAST-lite, not
VPT/VAST or an official MambaTS reproduction; its preflight contract is
[`mambatsv10_vastlite_etth1_h336_preflight_20260927.json`](mambatsv10_vastlite_etth1_h336_preflight_20260927.json).

`MambaTSv11` is the next bounded local candidate. It restores public VPT's
two material properties: a distinct random variable path for each training
sample and a detached per-sample pure-MSE transition-cost accumulator. Its
inference order is decoded by a deterministic greedy minimum-cost path rather
than the repository's optional simulated-annealing/local-search/Lin-Kernighan
solvers. The one Mamba scan, output restoration, and AutoCon-connected
forecast residual remain unchanged. This is therefore a mechanism-faithful
adaptation with a declared solver substitution, not an official reproduction.

## 2026-09-27 v11 preflight outcome

The frozen ETTh1/H336 three-seed validation-only preflight completed with all
six v11 runs reporting VPT transition coverage `1.0` and nonzero transition
counts. Nevertheless, active/null/PatchTST mean validation MSE was
`1.342472/1.340283/1.279776`: active lost to its exact null in two of three
seeds and to PatchTST in all three. The pre-registered gates therefore failed.
No test metric was produced. This closes the local v11 adaptation rather than
testing or judging the public repository's external ATSP solver. See
[`v11 readout`](mambatsv11_vpt_etth1_h336_preflight_readout_20260927.md).

## 2026-09-23 implementation-level ms-Mamba check

The public implementation was inspected at commit `HEAD` on 2026-09-23. Its
`HierarchicalMamba` obtains `dt` from a custom `MambaMain`, then calls separate
`MambaWithExternalDt` blocks with `dt * 2`, `dt * 4`, and `dt * 8` before
averaging four outputs. Both custom blocks explicitly avoid the normal fused
Mamba path in order to expose or inject `dt`. The active local environment
uses `mamba_ssm==2.3.1`; its public `Mamba` API does not expose an external
per-token Delta input while retaining the fused forward path.

This is a structural cost finding, not a benchmark claim: it rules out
implementing ms-Mamba as the next candidate because the required mechanism
adds at least one full selective scan and loses the optimized path. A future
project whose primary objective is accuracy rather than latency can revisit it
with a separately pre-registered throughput protocol. No model code or
forecast result was produced from this inspection.

## Candidate gate

An ms-Mamba-inspired candidate is only justified after the ongoing baseline
matrix establishes a target coordinate. Its smallest valid screen must:

1. use two Delta branches and a single learned fusion, with an exact one-branch
   control and the same forecast head;
2. retain the current AutoCon representation and contrastive loss, plus a
   `lambda=0` null control, so a backbone change cannot be mislabeled as an
   AutoCon gain;
3. run only on a selected core data coordinate and three validation seeds;
4. require lower mean validation MSE than the one-branch control in at least
   two seeds and no more than a 10% prediction-latency increase; and
5. stop immediately if it loses either gate. No test metric can select its
   scales, fusion, or loss weight.

This gate prevents a costly multi-branch implementation from becoming another
unbounded architecture search. The ongoing six-model matrix is the source for
choosing the coordinate and relevant accuracy baseline.
