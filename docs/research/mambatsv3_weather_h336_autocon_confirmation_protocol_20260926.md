# MambaTS v3 Weather H336 AutoCon confirmation protocol (2026-09-26)

## Motivation

The current-source Weather H96 screen passed both the exact AutoCon null and
same-contract strong-baseline gates. H96 alone is insufficient evidence for a
long-term forecasting contribution. This validation-only phase first asks
whether the small active-vs-null effect survives at H336 before allocating a
full H336 baseline matrix.

## Fixed contract

- Dataset: Weather multivariate, SHA-256
  `773b97c4495ed044b390241cea9a9c39b79c80b2436dbcc0fbd32e289c86abf5`;
- Input/label/prediction lengths: `96/48/336`; 21 variables, `MeanStd`;
- Candidate: MambaTS v3, one layer, `d_model=32`, `d_ff=64`, patch `16/8`,
  causal embedding, Mamba state/convolution/expand `16/4/2`, one scale;
- Training: batch 32, eight epochs maximum, patience 3, `lr=.001` cosine,
  `weight_decay=1e-5`, seeds `2021/2022/2023`;
- Exact paired arms: `MambaTSv3AutoCon` with `AutoCon_lambda=0` and `.001`.

All jobs use GPU `cuda:0`, `--skip_test`, and `--no_save`. The test split is
not run or inspected.

## Gates

Run the current-source H336 strong-baseline matrix only if active has no higher
mean best validation MSE than its exact null and wins at least two of three
seed pairs with no numerical or contract failure. If the active arm fails,
stop this candidate at H336. A passing null gate licenses baseline comparison,
not held-out testing or a general efficiency claim.

