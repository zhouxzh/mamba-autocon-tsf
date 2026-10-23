# MambaTS v3 Weather AutoCon replication protocol (2026-09-26)

## Question

The earlier Weather H96 pilot was the only current candidate with both a
three-seed strong-baseline accuracy signal and a favorable matched forward
latency observation. Its design lacked an exact AutoCon lambda-zero arm. This
validation-only replication asks whether the signed AutoCon representation
objective has a reproducible causal contribution under the current source.

## Fixed contract

- Weather M features, file hash
  `773b97c4495ed044b390241cea9a9c39b79c80b2436dbcc0fbd32e289c86abf5`;
- H96, `seq_len=96`, `label_len=48`, `MeanStd`, batch 32, eight epochs,
  patience 3, learning rate .001/cosine, seeds 2021/2022/2023;
- MambaTS v3: `d_model=32`, `d_ff=64`, one layer, Mamba state 16/convolution
  4/expand 2, patch length/stride 16/8, causal embedding, one scale;
- paired arms: `AutoCon_lambda=0` and `.001`. All other arguments, data split,
  random seed and optimizer settings are identical.

Every run has `--skip_test --no_save`; no test data is used for selection.

## Continuation rule

Continue to a new same-snapshot Weather strong-baseline matrix only if the
active arm has lower mean validation MSE than lambda-zero, wins at least 2/3
seed pairs, and does not trigger numerical/representation-contract failures.
Otherwise stop the MambaTS v3 Weather branch. Even a passing result does not
establish a speed claim: it only licenses baselines and end-to-end resource
measurement under one source snapshot.
