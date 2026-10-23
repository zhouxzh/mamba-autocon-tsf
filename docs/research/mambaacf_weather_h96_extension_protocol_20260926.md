# MambaACF Weather H96 extension protocol (2026-09-26)

## Question

The current-source MambaACF ACF-memory candidate is competitive on the
high-dimensional Electricity and Traffic H96 validation coordinates. This
extension asks whether that signal transfers to the independent Weather H96
dataset without changing architecture or choosing a result from the test set.

## Frozen contract

- dataset: `dataset/weather/weather.csv`, SHA-256
  `773b97c4495ed044b390241cea9a9c39b79c80b2436dbcc0fbd32e289c86abf5`;
- split/features: repository Weather M-feature default, `seq_len=96`,
  `label_len=48`, `pred_len=96`, `MeanStd`;
- budget: 8 epochs, patience 3, batch 32, Adam learning rate .001, cosine
  schedule, seeds `2021`, `2022`, `2023`, GPU `cuda:0`;
- fixed candidate: Mamba-2 MambaACF, `d_model=32`, `d_ff=64`, one scale,
  causal embedding, static train-only ACF memory with
  `train_peaks_available`, hard selected-lag relation target (`lag_mask=true`,
  global non-overlap pairs beginning at lag 96);
- arms: exact lambda-zero relation null and active signed relation loss with
  `AutoCon_lambda=.001`.

No test predictions, test losses, hyperparameter selection, or model changes
are allowed in this phase. Existing Weather DLinear, PatchTST, TimesNet,
iTransformer, and Mamba-2 rows are contextual strong-baseline references only:
they were produced in a prior frozen source snapshot, so any cross-snapshot
comparison is exploratory until a full same-snapshot matrix is regenerated.

## Decision rule

The active arm may proceed to a fresh same-snapshot baseline matrix only if it
is not worse than its exact null in at least 2/3 seeds and its mean is not more
than 2% above the retained best Weather baseline. Otherwise, stop this
architecture on Weather. Regardless of outcome, this phase alone cannot make a
paper accuracy or speed claim.
