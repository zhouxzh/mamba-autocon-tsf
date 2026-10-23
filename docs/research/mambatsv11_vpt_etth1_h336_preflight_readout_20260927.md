# MambaTS v11 VPT ETTh1 H336 Preflight Readout

## Decision

Stop the MambaTS v11 VPT branch. The registered validation-only continuation
rule required both a stable active-versus-null improvement and non-inferiority
to PatchTST. Neither condition passed. No test metric was generated, read, or
used.

## Frozen Scope

- Dataset: ETTh1, multivariate `C=7`, `L=96`, `H=336`.
- Seeds: `2221`, `2222`, `2223`; exact pairing across phases.
- Candidate: one shared Mamba scan with per-sample random VPT paths in
  training, detached pure-MSE directed edge-cost accumulation, greedy learned
  evaluation path, channel restoration, and direct AutoCon forecast fusion.
- Comparator: same-contract PatchTST.
- Training: 8-epoch budget, patience 3, GPU, `skip_test=true` for all jobs.

The plan and authorization are
[`preflight plan`](mambatsv11_vpt_etth1_h336_preflight_20260927.json) and
[`launch sidecar`](mambatsv11_vpt_etth1_h336_preflight_20260927_launch_sidecar.json).
This is a local adaptation of public MambaTS VPT: the official external ATSP
decoder is replaced with a deterministic greedy decoder, so it is not an
official reproduction.

## Results

| Seed | v11 null validation MSE | v11 active validation MSE | PatchTST validation MSE | Active versus null |
| ---: | ---: | ---: | ---: | ---: |
| 2221 | 1.354645 | 1.359018 | 1.281443 | +0.323% |
| 2222 | 1.349009 | 1.352200 | 1.277799 | +0.237% |
| 2223 | 1.317195 | 1.316197 | 1.280085 | -0.076% |
| Mean | 1.340283 | 1.342472 | 1.279776 | +0.163% |

The active model wins against its exact lambda-zero control in `1/3` seed
pairs and against PatchTST in `0/3`. Its mean validation MSE is `4.899%`
higher than PatchTST. Therefore the AutoCon gate and the accuracy gate are
both false.

## Integrity Checks

All 9 registered jobs completed successfully. The strict reader confirmed a
single identical source snapshot across null, active, and PatchTST phases;
matching configuration and seeds; finite validation losses; and no
`metrics.npy`, `original_metrics.npy`, or `stream_metrics.json` artifacts.

All six v11 runs reached VPT transition coverage `1.0` with cumulative edge
count `201061`. The negative result is consequently not explained by inactive
VPT bookkeeping. It shows that this VPT-plus-greedy-order adaptation does not
improve this ETTh1/H336 coordinate under the frozen contract.

Machine-readable validation rows and summary are retained under
[`v11 preflight readout artifacts`](../../artifacts/exploratory/mambatsv11_vpt_etth1_h336_preflight_20260927/readout/).

## Boundary

Do not tune the VPT loss source, edge normalization, greedy decoder, lambda,
seeds, or training budget after observing these results. Do not run a v11
resource comparison, multi-dataset extension, or held-out test release from
this branch. The result does not refute the official MambaTS VPT/VAST method,
because this repository does not execute its external ATSP solver.
