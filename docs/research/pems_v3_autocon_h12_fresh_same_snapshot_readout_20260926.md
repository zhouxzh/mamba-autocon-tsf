# PEMS H=12 fresh same-snapshot readout (2026-09-26)

## Scope

- Dataset: PEMS03, PEMS04, PEMS07, PEMS08.
- Horizon: `H=12`; input length `96`, label length `48`.
- Protocol: multivariate (`M`), MeanStd normalization, `d_model=32`, `d_ff=64`, one encoder layer, 20 epochs, early stopping patience 5, CUDA:0.
- Seeds: 2026, 2027, 2028.
- Split: validation-only exploratory screen. `--skip_test` was enabled; no test metrics are available.
- Source and plan: [`pems_v3_autocon_h12_fresh_same_snapshot_20260926.json`](pems_v3_autocon_h12_fresh_same_snapshot_20260926.json), SHA256 `2f46b60624e34e2a9e2260f3ace7c5ffeca5f6f1c1994081e89bac226b0fa1e9`.

All 120 jobs completed successfully. The four phase validation files report the expected number of unique jobs and zero validation errors. This is a complete validation screen, not a test-set release.

## Validation MSE means

The baseline means below are averaged over the three fresh seeds. The best external baseline per dataset is shown for the comparison gate.

| Dataset | DLinear | NLinear | PatchTST | TimesNet | TimeMixer | iTransformer | Mamba2 | MambaTSv3 plain | AutoCon null | AutoCon active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PEMS03 | 0.119718 | 0.119676 | 0.096695 | 0.063845 | 0.065467 | 0.076857 | 0.068243 | 0.060219 | 0.060033 | **0.060018** |
| PEMS04 | 0.137955 | 0.137975 | 0.119142 | 0.081965 | 0.080217 | 0.101055 | 0.081846 | 0.074886 | 0.074663 | **0.074632** |
| PEMS07 | 0.106067 | 0.106033 | 0.086311 | 0.081262 | 0.065095 | 0.071815 | 0.068107 | 0.059942 | 0.059525 | **0.059442** |
| PEMS08 | 0.135108 | 0.135046 | 0.115442 | 0.128293 | 0.092892 | 0.098952 | 0.097532 | 0.088215 | 0.088567 | **0.088577** |

Against the best external baseline (TimesNet for PEMS03 and TimeMixer for PEMS04/07/08), the active candidate improves the dataset means by approximately `5.99%`, `6.96%`, `8.68%`, and `4.64%`, respectively. This is a useful short-horizon signal, but it does not establish long-horizon or test-set superiority.

## AutoCon ablation

The active run uses `AutoCon_lambda=0.001`. The null run uses the same model and training protocol with an explicitly effective lambda of zero. The paired active-minus-null changes are:

| Dataset | Active vs null mean change | Seed-level wins (lower MSE) |
|---|---:|---:|
| PEMS03 | -0.0243% | 2/3 |
| PEMS04 | -0.0413% | 2/3 |
| PEMS07 | -0.1390% | 3/3 |
| PEMS08 | +0.0113% | 1/3 |

The contrastive representation objective is active: the active runs have effective lambda `0.001` and nonzero AutoCon loss. The null runs retain diagnostic AutoCon values but have effective lambda `0.0`. The separate `mamba_acf_relation_loss` was disabled in this screen, so these results support an AutoCon representation-learning effect, not an independent ACF-relation-loss claim.

The improvement over the plain MambaTSv3 model is larger on PEMS03/04/07, but PEMS08 is worse. This is an ablation signal, not yet a universal claim.

## GPU resource screen

Representative prediction-only measurements used PEMS07 (`B=32`, `L=96`, `C=883`, `H=12`, CUDA:0, 10 warmup and 30 timed iterations). These are forward-pass measurements and exclude training, data loading, and AutoCon loss computation.

| Model | Parameters | Mean latency (ms) | P95 latency (ms) | Peak allocated (MB) |
|---|---:|---:|---:|---:|
| NLinear | 1,164 | 0.095 | 0.098 | 68.3 |
| TimeMixer | 158,643 | 0.601 | 0.614 | 97.0 |
| Mamba2 | 158,839 | 0.753 | 0.770 | 73.6 |
| iTransformer | 12,172 | 0.915 | 0.929 | 57.3 |
| TimesNet | 1,296,639 | 2.492 | 2.517 | 80.3 |
| MambaTSv3AutoCon | 66,491 | 2.189 | 2.204 | 647.8 |
| PatchTST | 13,740 | 4.025 | 4.031 | 412.4 |

The candidate is not the fastest and has substantially higher peak allocation in this representative configuration. The resource artifact is [`model_resources.csv`](../../artifacts/exploratory/pems_v3_resource_screen_20260926/model_resources.csv). Because this was a direct configuration benchmark, it should be treated as a resource screen rather than a separately trained efficiency experiment.

## Current decision

This screen supports the following narrow statement:

> Under a frozen, same-source PEMS H=12 validation protocol, MambaTSv3AutoCon is a strong accuracy candidate and beats the best external baseline on all four dataset means. The AutoCon component provides a small but consistent average benefit on three datasets, while the fourth is neutral/slightly negative.

It does **not** yet support a paper claim of general superiority, long-horizon robustness, test-set performance, or speed/memory superiority. Before a confirmatory paper table, the next required evidence is: test-set evaluation, H=96 (and preferably H=192/336) under the same source snapshot, paired confidence intervals across more seeds, and a targeted reduction of the candidate's GPU memory/latency overhead.

