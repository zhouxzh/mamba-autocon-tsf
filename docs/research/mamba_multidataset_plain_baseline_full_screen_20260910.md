# Multidataset plain-baseline full screen (2026-09-10)

## Executive conclusion

The fresh six-baseline matrix completed all `216/216` validation-only jobs:
ETTh1, ETTm1, Electricity, and Traffic; horizons 96/336/720; seeds
2021/2022/2023; DLinear, NLinear, PatchTST, TimesNet, Mamba-1, and Mamba-2.
Every manifest record is `ok`, every coordinate is present exactly once, and
the phase validator reports `valid=true`, `expected_jobs=216`, and
`status_counts.ok=216`.

This result does **not** support a universal plain-Mamba accuracy claim:

- On ETTh1, PatchTST is best at all three horizons. Mamba-1 is `+5.48%`,
  `+7.32%`, and `+7.62%` above the strongest non-Mamba mean at H96/H336/H720;
  Mamba-2 is `+7.18%`, `+9.39%`, and `+10.52%`.
- On ETTm1, PatchTST remains best at H96/H336. At H720 Mamba-1 is `-1.33%`
  relative to DLinear and Mamba-2 is `-0.66%`, but both remain above PatchTST.
- On Electricity, Mamba-2 beats DLinear/NLinear/PatchTST by `4.06--12.36%`
  depending on horizon, but TimesNet remains better by `3.63--6.78%`.
- On Traffic, Mamba-1/2 beat DLinear and NLinear by roughly `1.3--3.4%`,
  but lose to PatchTST by `9.72--16.98%` and to TimesNet by `3.41--15.71%`.

The defensible finding is therefore dataset-dependent: Mamba-2 is a useful
high-dimensional plain baseline and can be competitive with linear models,
but the current plain Mamba backbone does not meet the requested “no worse
than all baselines” accuracy bar. The AutoCon/ACF candidate must be evaluated
against this frozen coordinate contract; it cannot inherit a claim from plain
Mamba.

## Frozen contract

Plan: [`mamba_multidataset_plain_baseline_full_screen_20260910.json`](mamba_multidataset_plain_baseline_full_screen_20260910.json)

Artifacts:

- source and launcher root: `artifacts/exploratory/mamba_multidataset_plain_baseline_full_screen_20260910`
- phase root: `artifacts/exploratory/mamba_multidataset_plain_baseline_full_screen_20260910/plain_baselines`
- machine-readable readout: `artifacts/exploratory/mamba_multidataset_plain_baseline_full_screen_20260910/plain_baselines/summary/`
- coverage audit: `artifacts/audit_multidataset_plain_baseline_full_screen_20260910`

All rows use M features, `seq_len=96`, the same optimizer and short
validation budget, and `skip_test=true`. The primary metric is the minimum
pure-forecast `val_loss` from each checkpoint's `loss_logs.json`. No test
loader, `metrics.npy`, `original_metrics.npy`, or stream test artifact was
used. The paired unit is dataset x horizon x seed; overlapping windows and
channels are not treated as independent replicates.

The independent summary command is:

```bash
conda run --no-capture-output -n mamba python scripts/summarize_multidataset_baseline_screen.py \
  --plan docs/research/mamba_multidataset_plain_baseline_full_screen_20260910.json \
  --phase-root artifacts/exploratory/mamba_multidataset_plain_baseline_full_screen_20260910/plain_baselines \
  --output-dir artifacts/exploratory/mamba_multidataset_plain_baseline_full_screen_20260910/plain_baselines/summary
```

The summary contains 216 seed-level rows, 72 dataset/horizon/model groups,
240 exact-coordinate paired comparisons, and 36 descriptive best-model cells.
The full group table and every paired delta are in `summary/summary.csv` and
`summary/paired_deltas.csv`; the rendered readout is
`summary/readout.md`.

## Mean best validation MSE

Each cell is mean +/- sample SD over the three seeds. Lower is better only
within a dataset and horizon; these are validation values, not test metrics.

| Dataset | Horizon | DLinear | NLinear | PatchTST | TimesNet | Mamba-1 | Mamba-2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh1 | 96 | 0.695156 | 0.697799 | **0.685596** | 0.768010 | 0.723166 | 0.734823 |
| ETTh1 | 336 | 1.284755 | 1.284927 | **1.278423** | 1.341457 | 1.371943 | 1.398460 |
| ETTh1 | 720 | 1.554533 | 1.555215 | **1.551895** | 1.663838 | 1.670177 | 1.715085 |
| ETTm1 | 96 | 0.410020 | 0.408890 | **0.399746** | 0.440102 | 0.419661 | 0.411823 |
| ETTm1 | 336 | 0.686746 | 0.686254 | **0.664856** | 0.692058 | 0.691148 | 0.690630 |
| ETTm1 | 720 | 1.007128 | 1.006950 | **0.976294** | 0.987300 | 0.993742 | 1.000525 |
| Electricity | 96 | 0.201873 | 0.201958 | 0.194715 | **0.170231** | 0.180997 | 0.176997 |
| Electricity | 336 | 0.219740 | 0.219831 | 0.217591 | **0.203430** | 0.216569 | 0.210811 |
| Electricity | 720 | 0.270270 | 0.270352 | 0.267944 | **0.237409** | 0.266701 | 0.253516 |
| Traffic | 96 | 0.506108 | 0.506352 | **0.420740** | 0.425349 | 0.492186 | 0.489107 |
| Traffic | 336 | 0.453556 | 0.453779 | **0.403727** | 0.429357 | 0.446614 | 0.444016 |
| Traffic | 720 | 0.488807 | 0.489076 | 0.433200 | **0.424684** | 0.478547 | 0.475311 |

The bold value is the best of the six plain models in that coordinate. This
is a descriptive validation ranking, not a statistical claim.

## Training wall-time observation

The short-budget end-to-end elapsed means (seconds) show that plain Mamba is
not automatically faster than the linear baselines. For example, at H720,
Mamba-2 takes about `17.2s/31.4s/85.1s/129.2s` on
ETTh1/ETTm1/Electricity/Traffic, while TimesNet takes
`39.5s/139.6s/154.3s/149.6s`. These values include data loading and this
short training schedule; they are not synchronized prediction latency,
train-step throughput, or a publication-grade efficiency result. A fair
efficiency claim still requires matched parameter reporting, synchronized
forward and train-step protocols, peak allocated/reserved memory, and a
fixed hardware/software snapshot.

## Interpretation and next gate

1. Keep Mamba-1 and Mamba-2 as separate baselines; do not pool them into one
   “Mamba” number.
2. Use Mamba-2 as the primary plain-Mamba control for high-dimensional
   Electricity/Traffic, while retaining Mamba-1 for the architecture-family
   ablation.
3. Evaluate `MambaACF`/`MambaACFAutoCon` on the same four datasets and horizon
   grid, with a validation-only AutoCon lambda selection phase and the same
   baseline coordinates. The current PEMS v3/v4/v5 screens cannot substitute
   for this matrix.
4. Do not release test metrics until the candidate, lambda, and resource
   protocol are frozen. A result that only matches DLinear/NLinear but remains
   behind PatchTST/TimesNet is not sufficient for the proposed paper claim.

## Provenance checks

- plan SHA256: `095a56ae5d73d02fec593c1c2271bed66b269f24a1e0435baaa2fac7f9762ccb`
- manifest SHA256: `c58d99d11d844b54b99d00a98eb28c65453bfc3900ef349de2cc361d97426749`
- phase validation SHA256: `5d1e38889ee18a1299ed3847b04b8750003655419bc82a80fe5c2c62f7faed32`
- source tree hash in every manifest row: `e491e7014b2512e2d159c4460c97f879196c39b666cba94633a547148901e161`
- test metrics produced: `false`

## Manual review checklist

- [ ] Confirm the six-model matrix uses one source snapshot and the four data hashes recorded in the plan.
- [ ] Confirm validation MSE is read only from `loss_logs.json:val_loss` and no test artifact is present.
- [ ] Review the Traffic/PatchTST and Electricity/TimesNet gaps before selecting the proposed ACF-Mamba candidate.
- [ ] Require a same-contract AutoCon/ACF phase before making any accuracy or speed statement.
- [ ] Treat this screen as baseline coverage evidence, not as a publication-ready result.
